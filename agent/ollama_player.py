"""
Compact Local Ollama VLM Player for YellowAI (`qwen3.5:9b`).
- 100% local via Ollama `/api/chat` (zero cloud APIs)
- Multi-turn conversation history + automatic compaction across runs
- DuckDuckGo internet search tool calling & on-demand `web_search_query`
- Lossless sign/speech transcription & strategic thinking persistence
- Strict GBA button outputs: UP, DOWN, LEFT, RIGHT, A, B, SELECT, START
"""

import base64
from enum import Enum
import io
import json
import re
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from pydantic import BaseModel, Field

import config
from agent.prompts import SYSTEM_INSTRUCTION, build_turn_prompt
from agent.web_search import WebSearchClient
from core.conversation_history import ConversationHistoryManager
from core.dialogue_recorder import DialogueRecorder
from core.goal_tracker import GoalTracker
from core.thinking_store import ThinkingStore

VALID_GBA_BUTTONS = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT", "START"}


class GBAButton(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    A = "A"
    B = "B"
    SELECT = "SELECT"
    START = "START"


class TurnDecision(BaseModel):
    reasoning: str = Field(..., description="Brief visual observation and rationale for this turn's button presses.")
    actions: List[GBAButton] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="REQUIRED: 1 to 5 GBA buttons to press this turn (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT).",
    )
    current_area: Optional[str] = Field(
        default=None,
        description="Observed location name (e.g. 'Red Bedroom 2F', 'Red House 1F', 'Pallet Town', 'Oak Lab', 'Route 1'). Populate every turn when identifiable.",
    )
    screen_kind: Optional[str] = Field(
        default=None,
        description="Observed screen category: OVERWORLD, START_MENU, TRAINER_CARD, PARTY_MENU, BAG_MENU, POKEDEX, DIALOGUE, BATTLE, or TITLE_OR_MENU.",
    )
    dialogue_text: Optional[str] = Field(
        default=None,
        description="Verbatim text of any visible sign, speech bubble, battle box, or menu status so Python archives it.",
    )
    discovered_fact: Optional[str] = Field(
        default=None,
        description="Optional verified fact read from Start Menu / Trainer Card / Bag / Party / Pokédex / Sign.",
    )
    ready_to_plan_goal: Optional[bool] = Field(
        default=None,
        description="Set true after finishing Start Menu / sign inspection to synthesize your next objective from online search.",
    )
    completed_goal: Optional[str] = Field(
        default=None,
        description="Set to 'CURRENT' (or the finished goal text) to permanently DROP an already-completed goal.",
    )
    active_goal_update: Optional[str] = Field(
        default=None,
        description="Optional concrete new active goal if current goal was dropped or discovered via menus/signs/web.",
    )
    next_step: Optional[str] = Field(default=None, description="Short 1-line immediate physical next step.")
    grand_goal_update: Optional[str] = Field(default=None, description="Optional update to overarching grand goal.")
    web_search_query: Optional[str] = Field(
        default=None,
        description="Optional internet search query if unsure how to progress, find an item/location, or beat a battle/puzzle.",
    )
    strategic_insight: Optional[str] = Field(
        default=None,
        description="Optional 1-sentence reusable lesson to remember across future runs.",
    )


class GoalSynthesis(BaseModel):
    confirmed_completed_goals: List[str] = Field(
        default_factory=list,
        description="Any goals proven ALREADY completed based on inspected Badges, Party, Bag, Pokédex, or Signs.",
    )
    new_active_goal: str = Field(..., description="Concrete, single next objective to accomplish next.")
    next_step: str = Field(..., description="Immediate physical next step to start pursuing `new_active_goal`.")
    completion_criteria: str = Field(..., description="Clear observable condition showing `new_active_goal` is done.")
    grand_goal: Optional[str] = Field(default=None, description="Updated overarching goal if appropriate.")


class OllamaPlayer:
    """Single-backend local Ollama multimodal agent for Pokémon Recharged Yellow."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model_id: str = config.OLLAMA_MODEL_ID,
        temperature: float = 0.2,
        timeout: float = config.OLLAMA_TIMEOUT,
        num_ctx: int = config.OLLAMA_NUM_CTX,
        think_mode: str = config.OLLAMA_THINK_MODE,
        enable_web_search: bool = config.OLLAMA_ENABLE_WEB_SEARCH,
        history_manager: Optional[ConversationHistoryManager] = None,
        web_client: Optional[WebSearchClient] = None,
        thinking_store: Optional[ThinkingStore] = None,
        goal_tracker: Optional[GoalTracker] = None,
        dialogue_recorder: Optional[DialogueRecorder] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.temperature = temperature
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.think_mode = (think_mode or "auto").lower().strip()
        self.enable_web_search = enable_web_search
        self.consecutive_errors = 0

        self.history_manager = history_manager or ConversationHistoryManager()
        self.web_client = web_client or WebSearchClient()
        self.thinking_store = thinking_store or ThinkingStore()
        self.goal_tracker = goal_tracker or GoalTracker()
        self.dialogue_recorder = dialogue_recorder or DialogueRecorder()

    @staticmethod
    def _normalize_button(btn: Any) -> Optional[str]:
        s = str(btn or "").upper().strip()
        aliases = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT", "ENTER": "START", "BACKSPACE": "SELECT"}
        s = aliases.get(s, s)
        return s if s in VALID_GBA_BUTTONS else None

    @classmethod
    def flatten_actions(cls, result_data: Dict[str, Any]) -> List[str]:
        raw = result_data.get("actions")
        buttons: List[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    norm = cls._normalize_button(item.get("button"))
                    rep = int(item.get("repeat", 1) or 1)
                    if norm:
                        buttons.extend([norm] * max(1, min(5, rep)))
                else:
                    norm = cls._normalize_button(item)
                    if norm:
                        buttons.append(norm)
        elif isinstance(raw, str):
            for tok in re.split(r"[\s,->]+", raw):
                norm = cls._normalize_button(tok)
                if norm:
                    buttons.append(norm)
        return buttons[: config.MAX_BATCH_ACTIONS] if buttons else ["A"]

    @staticmethod
    def _encode_image_b64(img: Image.Image, quality: int = 85) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_json(raw_text: str) -> Dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return {}

    def is_server_reachable(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3.0).status_code == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code != 200:
                return False
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(self.model_id == n or n.startswith(self.model_id) for n in names)
        except Exception:
            return False

    def _should_think(self, turn_id: int, stuck_alert: str) -> bool:
        if self.think_mode == "always":
            return True
        if self.think_mode == "never":
            return False
        # Periodic reflection turns; fast turns stay responsive and retry without think if truncated
        return turn_id > 0 and (turn_id % 10 == 0)

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        think: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        num_predict = (
            getattr(config, "OLLAMA_MAX_TOKENS_THINK", 900)
            if think
            else getattr(config, "OLLAMA_MAX_TOKENS_FAST", 450)
        )
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": "15m",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        if tools:
            payload["tools"] = tools
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def run_web_research_session(
        self,
        objective_question: str,
        current_area: str = "",
        image: Optional[Image.Image] = None,
        max_tool_rounds: int = 2,
    ) -> Dict[str, Any]:
        """Runs Ollama native tool calling (`web_search`, `fetch_webpage`) to figure out next steps."""
        if not self.enable_web_search:
            return {"summary": "", "used_tools": False}

        user_msg: Dict[str, Any] = {
            "role": "user",
            "content": (
                f"Game: Pokémon Recharged Yellow (GBA Gen-3 Emerald-engine remake of Pokémon Yellow)\n"
                f"Area: {current_area or self.goal_tracker.current_area}\n"
                f"Verified Facts: {', '.join(self.goal_tracker.discovered_facts[-5:]) or 'None yet'}\n"
                f"Question: {objective_question}\n\n"
                "Use `web_search` to find reliable walkthrough details, then give a concise actionable summary."
            ),
        }
        if image is not None:
            user_msg["images"] = [self._encode_image_b64(image, quality=80)]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "Research Pokémon Recharged Yellow concisely using web_search/fetch_webpage."},
            user_msg,
        ]
        tools = self.web_client.get_ollama_tool_schemas()
        used_tools = False
        final_text = ""

        try:
            for _ in range(max_tool_rounds + 1):
                data = self._chat(messages=messages, think=False, tools=tools)
                msg = data.get("message", {})
                final_text = (msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    break
                messages.append(msg)
                used_tools = True
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    t_name = fn.get("name", "")
                    t_args = fn.get("arguments", {})
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except Exception:
                            t_args = {}
                    out = self.web_client.execute_tool_call(t_name, t_args or {})
                    messages.append({"role": "tool", "name": t_name, "content": out[:3500]})

            if final_text:
                self.web_client.add_verified_fact(topic=objective_question[:60], summary=final_text[:500])
                self.thinking_store.add_insight(
                    insight=final_text[:280],
                    area=current_area or self.goal_tracker.current_area,
                    category="WEB_GUIDE",
                    source="WEB_SEARCH",
                )
        except Exception as e:
            print(f"[OllamaPlayer] Web research note: {e}")

        return {"summary": final_text, "used_tools": used_tools}

    def synthesize_goal_from_discoveries(
        self,
        image: Optional[Image.Image] = None,
        extra_context: str = "",
    ) -> Dict[str, Any]:
        """
        Combines Start Menu discoveries (Badges, Party, Bag, Pokédex), recorded signs/dialogue,
        and live internet search to drop already-finished goals and synthesize the next goal.
        """
        profile = self.goal_tracker.verified_profile
        area = self.goal_tracker.current_area
        recent_diag = self.dialogue_recorder.format_recall_prompt(area=area, recent_n=8, area_n=5)

        query = (
            f"Pokémon Recharged Yellow walkthrough: Current Area='{area}', "
            f"Badges='{profile.get('badges')}', Party='{profile.get('party')}', "
            f"Bag='{profile.get('bag')}', Pokédex='{profile.get('pokedex')}'. "
            "What goals are already done and what should the player do next?"
        )
        web_res = self.run_web_research_session(objective_question=query, current_area=area, image=image, max_tool_rounds=2)

        synth_prompt = (
            "Synthesize our exact verified state and next objective in Pokémon Recharged Yellow.\n\n"
            f"- Current Location: {area}\n"
            f"- Verified Menu Profile: {json.dumps(profile, ensure_ascii=False)}\n"
            f"- Other Discovered Facts: {self.goal_tracker.discovered_facts[-8:]}\n"
            f"- Already Dropped Goals: {self.goal_tracker.completed_goals[-10:]}\n"
            f"- Current Candidate Goal: {self.goal_tracker.active_goal}\n"
            f"{recent_diag}\n"
            f"[Online Research Summary]:\n{web_res.get('summary', '') or 'No new web results.'}\n"
            f"{extra_context}\n\n"
            "Rules:\n"
            "1. List a goal in `confirmed_completed_goals` ONLY if it was our active_goal OR is directly proven done by "
            "Verified Menu Profile / Signs (e.g. actual Badge count or item in Bag). NEVER mark unvisited gyms, towns, badges, or starter Pokémon complete!\n"
            "2. Choose ONE clear, immediate `new_active_goal` matching the exact room/town visible right now."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "Synthesize grounded Pokémon Recharged Yellow goals from menus, signs, and web search. Output JSON."},
            {"role": "user", "content": synth_prompt},
        ]
        try:
            data = self._chat(messages=messages, think=False, json_schema=GoalSynthesis.model_json_schema())
            parsed = self._parse_json(data.get("message", {}).get("content", ""))
            if parsed and parsed.get("new_active_goal"):
                self.goal_tracker.apply_synthesized_goal(
                    new_active_goal=parsed.get("new_active_goal"),
                    next_step=parsed.get("next_step"),
                    completion_criteria=parsed.get("completion_criteria"),
                    grand_goal=parsed.get("grand_goal"),
                    confirmed_completed_goals=parsed.get("confirmed_completed_goals"),
                )
                return parsed
        except Exception as e:
            print(f"[OllamaPlayer] Goal synthesis fallback: {e}")
        return {}

    def decide_action(
        self,
        turn_id: int,
        image: Image.Image,
        previous_image: Optional[Image.Image] = None,
        discovery_directive: str = "",
        action_history: str = "",
        stuck_alert: str = "",
        emulator_verified: bool = False,
        screenshot_path: str = "",
    ) -> Dict[str, Any]:
        """Captures multi-turn context + screenshot and returns validated GBA button presses."""
        if not emulator_verified:
            raise RuntimeError("[SAFEGUARD BLOCKED] Refusing to query Ollama without verified emulator.")

        area = self.goal_tracker.current_area
        goal = self.goal_tracker.active_goal
        pinned_goal = self.goal_tracker.format_pinned_anchor()

        self.history_manager.compact_if_needed(
            ollama_base_url=self.base_url,
            ollama_model_id=self.model_id,
            pinned_goal=goal,
            completed_goals=self.goal_tracker.completed_goals,
        )

        insights_block = self.thinking_store.format_insights_prompt(area=area, goal=goal, limit=4)
        dialogue_block = self.dialogue_recorder.format_recall_prompt(area=area, query=goal, recent_n=5, area_n=4)
        cached_facts = self.web_client.get_relevant_cached_facts(f"{area} {goal}", limit=3)
        web_block = "[Internet Knowledge Cache]:\n" + "\n".join(f"- {f}" for f in cached_facts) if cached_facts else ""

        prompt_text = build_turn_prompt(
            turn_id=turn_id,
            pinned_goal_block=pinned_goal,
            discovery_directive=discovery_directive,
            stuck_alert=stuck_alert,
            action_history=action_history,
            thinking_insights_block=insights_block,
            dialogue_archive_block=dialogue_block,
            web_facts_block=web_block,
        )

        images = []
        if previous_image is not None and stuck_alert:
            images.append(self._encode_image_b64(previous_image, quality=75))
        images.append(self._encode_image_b64(image, quality=85))

        messages = self.history_manager.build_ollama_messages(
            system_prompt=SYSTEM_INSTRUCTION,
            current_user_prompt=prompt_text,
            current_images=images,
            completed_goals=self.goal_tracker.completed_goals,
        )

        schema = TurnDecision.model_json_schema()
        use_think = self._should_think(turn_id, stuck_alert)

        try:
            msg: Dict[str, Any] = {}
            raw_text = ""
            thinking_text = ""
            result: Dict[str, Any] = {}

            try:
                data = self._chat(messages=messages, think=use_think, json_schema=schema)
                msg = data.get("message", {})
                raw_text = (msg.get("content") or "").strip()
                thinking_text = (msg.get("thinking") or "").strip()
                result = self._parse_json(raw_text)
            except requests.exceptions.Timeout:
                if not (use_think or len(images) > 1):
                    raise
                print(f"[OllamaPlayer] Turn {turn_id} timed out with think={use_think}/imgs={len(images)}; retrying fast mode...")

            # If thinking hit token limit (empty content) or timed out, retry cleanly with think=False
            if not result and (use_think or len(images) > 1):
                if thinking_text and not raw_text:
                    print(f"[OllamaPlayer] Turn {turn_id} exhausted token budget in <think>; retrying fast mode (think=False)...")
                use_think = False
                fast_messages = list(messages[:-1]) + [
                    {
                        "role": "user",
                        "content": prompt_text,
                        "images": [images[-1]],
                    }
                ]
                data = self._chat(messages=fast_messages, think=False, json_schema=schema)
                msg = data.get("message", {})
                raw_text = (msg.get("content") or "").strip()
                result = self._parse_json(raw_text)

            if not result:
                raise ValueError(f"Invalid JSON from Ollama: {raw_text[:160]}")

            # Execute on-demand DuckDuckGo lookup if requested by model
            web_q = result.get("web_search_query")
            if self.enable_web_search and isinstance(web_q, str) and web_q.strip() and web_q.strip().upper() != "NULL":
                clean_q = web_q.strip()
                print(f"[OllamaPlayer] Web search requested: '{clean_q}'")
                search_res = self.web_client.search(clean_q, max_results=4)
                snippets = [f"- {r.get('title','')}: {r.get('snippet','')}" for r in search_res.get("results", [])]
                if snippets:
                    joined = "\n".join(snippets)
                    url0 = search_res["results"][0].get("url", "")
                    self.web_client.add_verified_fact(topic=clean_q[:60], summary=joined[:500], source_url=url0)
                    followup = messages + [
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": f"[WEB SEARCH RESULTS for '{clean_q}']:\n{joined}\n\nNow output updated JSON (`web_search_query`: null).",
                        },
                    ]
                    data2 = self._chat(messages=followup, think=False, json_schema=schema)
                    parsed2 = self._parse_json(data2.get("message", {}).get("content", ""))
                    if parsed2:
                        result, data = parsed2, data2

            buttons = self.flatten_actions(result)
            result["flat_actions"] = buttons
            result["think_used"] = use_think

            prompt_tok = int(data.get("prompt_eval_count", 0) or 0)
            if thinking_text:
                self.thinking_store.record_trace(
                    run_id=self.history_manager.current_run_id,
                    turn_id=turn_id,
                    area=area,
                    goal=goal,
                    thinking_text=thinking_text,
                    thinking_tokens=max(1, len(thinking_text.split())),
                    decision_summary=result.get("reasoning", ""),
                    buttons=buttons,
                )

            insight = result.get("strategic_insight")
            if isinstance(insight, str) and insight.strip() and insight.strip().upper() != "NULL":
                self.thinking_store.add_insight(insight=insight.strip(), area=area, category="GAMEPLAY", source="TURN")

            self.history_manager.record_turn(
                turn_id=turn_id,
                area=result.get("current_area") or area,
                goal=goal,
                user_summary=f"Area={area} | StuckAlert={stuck_alert[:90]}",
                assistant_response=result,
                buttons_executed=buttons,
                outcome="PENDING",
                screenshot_path=screenshot_path,
                prompt_tokens=prompt_tok,
            )
            self.consecutive_errors = 0
            return result

        except Exception as e:
            self.consecutive_errors += 1
            print(f"[OllamaPlayer] Turn {turn_id} fallback ({self.consecutive_errors}): {e}")
            return {
                "actions": ["B"],
                "flat_actions": ["B"],
                "reasoning": f"Local Ollama fallback: {e}",
                "fallback": True,
            }
