"""
Persistent Multi-Turn Conversation History & Compaction Engine for YellowAI (Ollama).

Ensures:
1. Every turn sees previous turns' actions, observations, dialogue, and outcomes (no amnesia).
2. Past runs are persisted to disk (`state/conversation_history.json`) and loaded at startup
   so new runs retain full knowledge of prior runs.
3. Context window & 12GB VRAM stay bounded via:
   - Stripping base64 image payloads from older turns beyond `keep_recent_images`
     while keeping their textual observations/actions.
   - Automatic Compaction: when turn count or Ollama `prompt_eval_count` exceeds thresholds,
     older turns are condensed into a dense episode summary (via Ollama LLM or deterministic
     fallback), while preserving the Pinned Goal Anchor and the most recent turns verbatim.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

import config


class ConversationHistoryManager:
    """Manages multi-turn conversation history, cross-run continuity, and compaction."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_turns_before_compact: Optional[int] = None,
        keep_recent_turns: Optional[int] = None,
        compact_token_threshold: Optional[int] = None,
        keep_recent_images: Optional[int] = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else config.CONVERSATION_HISTORY_FILE
        self.max_turns_before_compact = (
            max_turns_before_compact
            if max_turns_before_compact is not None
            else getattr(config, "OLLAMA_HISTORY_MAX_TURNS", 20)
        )
        self.keep_recent_turns = (
            keep_recent_turns
            if keep_recent_turns is not None
            else getattr(config, "OLLAMA_HISTORY_KEEP_RECENT", 6)
        )
        self.compact_token_threshold = (
            compact_token_threshold
            if compact_token_threshold is not None
            else getattr(config, "OLLAMA_HISTORY_COMPACT_TOKENS", 10000)
        )
        self.keep_recent_images = (
            keep_recent_images
            if keep_recent_images is not None
            else getattr(config, "OLLAMA_KEEP_RECENT_IMAGES", 1)
        )

        # Persistent state across runs
        self.run_count: int = 0
        self.current_run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.past_runs: List[Dict[str, Any]] = []          # High-level summaries of previous runs
        self.compacted_summaries: List[str] = []           # Compacted episode summaries of older turns
        self.active_turns: List[Dict[str, Any]] = []       # Recent turn-by-turn records (user + assistant)
        self.last_prompt_tokens: int = 0

        self.load()
        self.run_count += 1
        self.save()

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.run_count = int(data.get("run_count", 0))
                self.past_runs = data.get("past_runs", [])[-12:]
                self.compacted_summaries = data.get("compacted_summaries", [])[-12:]
                self.active_turns = data.get("active_turns", [])
                self.last_prompt_tokens = int(data.get("last_prompt_tokens", 0))
        except Exception as e:
            print(f"[ConversationHistory] Warning: Could not load history ({e})")

    def save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_count": self.run_count,
                "current_run_id": self.current_run_id,
                "last_prompt_tokens": self.last_prompt_tokens,
                "past_runs": self.past_runs[-12:],
                "compacted_summaries": self.compacted_summaries[-12:],
                "active_turns": self.active_turns,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ConversationHistory] Warning: Could not save history ({e})")

    def record_turn(
        self,
        turn_id: int,
        area: str,
        goal: str,
        user_summary: str,
        assistant_response: Dict[str, Any],
        buttons_executed: Optional[List[str]] = None,
        outcome: str = "PENDING",
        screenshot_path: str = "",
        prompt_tokens: int = 0,
    ) -> None:
        """Records a completed turn into active conversation history."""
        if prompt_tokens > 0:
            self.last_prompt_tokens = prompt_tokens

        compact_assistant = {
            "intent": assistant_response.get("intent", ""),
            "target_poi": assistant_response.get("target_poi"),
            "buttons": buttons_executed or assistant_response.get("flat_actions", []),
            "battle_action": assistant_response.get("battle_action"),
            "reasoning": assistant_response.get("reasoning", ""),
            "dialogue_text": assistant_response.get("dialogue_text"),
            "micro_step": assistant_response.get("micro_step"),
        }

        self.active_turns.append({
            "run_id": self.current_run_id,
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "area": area,
            "goal": goal,
            "user_summary": user_summary,
            "assistant_json": json.dumps(compact_assistant, ensure_ascii=False),
            "buttons": compact_assistant["buttons"],
            "outcome": outcome,
            "screenshot_path": screenshot_path,
        })
        self.save()

    def update_last_turn_outcome(self, buttons_executed: List[str], outcome: str) -> None:
        """Updates the most recent turn with actual executed buttons and pixel/progress outcome."""
        if not self.active_turns:
            return
        last = self.active_turns[-1]
        last["buttons"] = buttons_executed
        last["outcome"] = outcome
        try:
            parsed = json.loads(last.get("assistant_json", "{}"))
            parsed["executed_buttons"] = buttons_executed
            parsed["outcome"] = outcome
            last["assistant_json"] = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        self.save()

    def needs_compaction(self) -> bool:
        """Determines whether active conversation history should be compacted."""
        if len(self.active_turns) > self.max_turns_before_compact:
            return True
        if self.last_prompt_tokens >= self.compact_token_threshold and len(self.active_turns) > self.keep_recent_turns:
            return True
        return False

    def _deterministic_compact(self, turns_to_compact: List[Dict[str, Any]]) -> str:
        """Fallback deterministic compaction if LLM compaction fails or is offline."""
        if not turns_to_compact:
            return ""
        t_start = turns_to_compact[0].get("turn_id", "?")
        t_end = turns_to_compact[-1].get("turn_id", "?")
        areas: List[str] = []
        dialogues: List[str] = []
        actions: List[str] = []
        for t in turns_to_compact:
            area = t.get("area", "")
            if area and (not areas or areas[-1] != area):
                areas.append(area)
            btns = ",".join(t.get("buttons", [])[:4])
            actions.append(f"T{t.get('turn_id','?')}[{btns}->{t.get('outcome','?')}]")
            try:
                aj = json.loads(t.get("assistant_json", "{}"))
                dt = aj.get("dialogue_text")
                if dt and str(dt).upper() != "NULL":
                    dialogues.append(str(dt)[:100])
            except Exception:
                pass
        parts = [f"Episode T{t_start}-T{t_end}: Areas=[{' -> '.join(areas[:6])}]"]
        parts.append(f"Actions: {' ; '.join(actions[-8:])}")
        if dialogues:
            parts.append(f"Dialogue seen: {' | '.join(dialogues[-3:])}")
        return " | ".join(parts)

    def compact_if_needed(
        self,
        ollama_base_url: Optional[str] = None,
        ollama_model_id: Optional[str] = None,
        pinned_goal: str = "",
        summarizer_fn: Optional[Callable[[str], str]] = None,
        completed_goals: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Compacts oldest turns when turn count or token count exceeds limits.
        Preserves `keep_recent_turns` verbatim and NEVER discards the pinned goal anchor.
        """
        if not self.needs_compaction():
            return None

        split_idx = max(1, len(self.active_turns) - self.keep_recent_turns)
        to_compact = self.active_turns[:split_idx]
        remaining = self.active_turns[split_idx:]
        if not to_compact:
            return None

        transcript_lines: List[str] = []
        for t in to_compact:
            transcript_lines.append(
                f"Turn {t.get('turn_id')} (Run {t.get('run_id')}, Area={t.get('area')}, Goal={t.get('goal')}): "
                f"Obs={t.get('user_summary','')[:220]} | Action={t.get('assistant_json','')[:220]} | Outcome={t.get('outcome','')}"
            )
        transcript_text = "\n".join(transcript_lines)

        summary_text = ""
        if summarizer_fn is not None:
            try:
                summary_text = (summarizer_fn(transcript_text) or "").strip()
            except Exception as e:
                print(f"[ConversationHistory] Custom summarizer failed ({e}); using fallback.")
        elif ollama_base_url and ollama_model_id:
            done_text = ", ".join((completed_goals or [])[-8:]) or "None"
            prompt = (
                "Compact these past Pokémon Yellow Recharged gameplay turns into a concise, high-signal episode summary "
                "(max 6 bullet points) so the AI remembers everything important without wasting context tokens.\n"
                "MUST PRESERVE: areas visited, NPCs talked to, items/Pokémon gained or lost, doors/paths that worked or failed.\n"
                "CRITICAL RULE: Clearly mark ANY finished/completed goals as ALREADY COMPLETED so the AI never repeats them!\n"
                f"Already Completed Goals (DONE): {done_text}\n"
                f"Current Active Goal (keep aligned): {pinned_goal}\n\n"
                f"Turns to compact:\n{transcript_text}"
            )
            try:
                resp = requests.post(
                    f"{ollama_base_url.rstrip('/')}/api/chat",
                    json={
                        "model": ollama_model_id,
                        "think": False,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": "You compress agent gameplay logs concisely and accurately."},
                            {"role": "user", "content": prompt},
                        ],
                        "options": {"temperature": 0.1, "num_ctx": getattr(config, "OLLAMA_NUM_CTX", 16384)},
                    },
                    timeout=35.0,
                )
                resp.raise_for_status()
                summary_text = resp.json().get("message", {}).get("content", "").strip()
            except Exception as e:
                print(f"[ConversationHistory] Ollama compaction fallback ({e})")

        if not summary_text:
            summary_text = self._deterministic_compact(to_compact)

        t_start = to_compact[0].get("turn_id", "?")
        t_end = to_compact[-1].get("turn_id", "?")
        header = f"[Compacted Turns {t_start}-{t_end} | Run {to_compact[0].get('run_id', '?')}]"
        full_summary = f"{header}\n{summary_text}"

        self.compacted_summaries.append(full_summary)
        # If compacted summaries themselves grow too numerous, fold oldest two into one meta-summary
        if len(self.compacted_summaries) > 10:
            merged_head = " || ".join(self.compacted_summaries[:3])
            self.compacted_summaries = [f"[Earlier Adventure Summary]\n{merged_head[:1200]}"] + self.compacted_summaries[3:]

        self.active_turns = remaining
        self.last_prompt_tokens = 0
        self.save()
        print(f"[ConversationHistory] Compacted {len(to_compact)} older turns (kept {len(remaining)} recent turns verbatim).")
        return full_summary

    def finalize_run(self, current_area: str, active_goal: str, team_summary: str, badges: int) -> None:
        """Snapshots the current run into `past_runs` so future runs immediately know what happened."""
        run_turns = [t for t in self.active_turns if t.get("run_id") == self.current_run_id]
        if not run_turns and not self.compacted_summaries:
            return
        last_turn = run_turns[-1].get("turn_id", 0) if run_turns else 0
        recent_actions = " -> ".join(
            f"T{t.get('turn_id')}:{'/'.join(t.get('buttons', [])[:2])}({t.get('outcome','?')})"
            for t in run_turns[-5:]
        )
        run_entry = {
            "run_id": self.current_run_id,
            "run_number": self.run_count,
            "ended_at": datetime.now().isoformat(),
            "last_turn": last_turn,
            "area": current_area,
            "active_goal": active_goal,
            "team": team_summary,
            "badges": badges,
            "recent_actions": recent_actions,
        }
        # Update existing entry for current_run_id if already saved, else append
        for i, pr in enumerate(self.past_runs):
            if pr.get("run_id") == self.current_run_id:
                self.past_runs[i] = run_entry
                self.save()
                return
        self.past_runs.append(run_entry)
        self.past_runs = self.past_runs[-12:]
        self.save()

    @staticmethod
    def _is_goal_done(goal_text: str, completed_goals: Optional[List[str]]) -> bool:
        if not goal_text or not completed_goals:
            return False
        g_low = goal_text.strip().lower()
        for cg in completed_goals:
            c_low = (cg or "").strip().lower()
            if c_low and (g_low == c_low or g_low in c_low or c_low in g_low):
                return True
        return False

    def build_ollama_messages(
        self,
        system_prompt: str,
        current_user_prompt: str,
        current_images: List[str],
        pinned_goal_block: str = "",
        completed_goals: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Builds a true multi-turn `messages` list for Ollama `/api/chat`:
        - System prompt (with Pinned Goal Anchor + Past Run Continuity + Compacted Summaries)
        - Multi-turn alternating user/assistant history from `active_turns` (text-only so images don't overflow VRAM)
        - Current turn user message WITH current screenshot(s)
        """
        sys_sections = [system_prompt]
        if pinned_goal_block:
            sys_sections.append(pinned_goal_block)

        if self.past_runs:
            prev_lines = ["[Knowledge From Previous Runs]:"]
            for pr in self.past_runs[-4:]:
                if pr.get("run_id") == self.current_run_id:
                    continue
                pr_goal = pr.get("active_goal", "")
                status_tag = (
                    " (ALREADY COMPLETED / DROPPED - DO NOT REPEAT)"
                    if self._is_goal_done(pr_goal, completed_goals)
                    else " (historical goal from that run)"
                )
                prev_lines.append(
                    f"- Run #{pr.get('run_number','?')} ({pr.get('run_id','?')}): "
                    f"Ended in '{pr.get('area','?')}', Badges={pr.get('badges',0)}/8, "
                    f"Team=[{pr.get('team','None')}], EarlierGoal='{pr_goal}'{status_tag}, "
                    f"LastActions=[{pr.get('recent_actions','')}]"
                )
            if len(prev_lines) > 1:
                sys_sections.append("\n".join(prev_lines))

        if completed_goals:
            sys_sections.append(
                "[COMPLETED / DROPPED GOALS - NEVER ATTEMPT THESE AGAIN]:\n"
                + "\n".join(f"- DONE: {g}" for g in completed_goals[-8:])
            )

        if self.compacted_summaries:
            sys_sections.append("[Compacted History of Earlier Turns]:\n" + "\n\n".join(self.compacted_summaries[-4:]))

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(sys_sections)}
        ]

        # Include recent turns verbatim as multi-turn chat history (without old base64 images)
        recent = self.active_turns[-self.keep_recent_turns:]
        for t in recent:
            t_goal = t.get("goal", "")
            done_note = " [THIS EARLIER GOAL IS NOW COMPLETED/DROPPED]" if self._is_goal_done(t_goal, completed_goals) else ""
            u_msg = (
                f"[Past Turn {t.get('turn_id')} | Run {t.get('run_id')} | Area: {t.get('area')}]{done_note}\n"
                f"{t.get('user_summary', '')}\n"
                f"(Actual Outcome: {t.get('outcome', 'UNKNOWN')}, Executed Buttons: {t.get('buttons', [])})"
            )
            messages.append({"role": "user", "content": u_msg})
            messages.append({"role": "assistant", "content": t.get("assistant_json", "{}")})

        # Current live turn with images
        messages.append({
            "role": "user",
            "content": current_user_prompt,
            "images": current_images,
        })
        return messages

    def reset(self) -> None:
        self.run_count = 1
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.past_runs = []
        self.compacted_summaries = []
        self.active_turns = []
        self.last_prompt_tokens = 0
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass
        self.save()
