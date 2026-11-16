"""
Lossless Sign, Speech, & On-Screen Text Recorder for YellowAI.
Stores every unique sign, NPC speech bubble, and dialogue line permanently (untruncated)
in a single searchable file (`state/signs_and_speech.json`), linked to run_id, turn_id,
area, category (SIGN/NPC_SPEECH/DIALOGUE), timestamp, and screenshot path.
"""

from difflib import SequenceMatcher
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class DialogueRecorder:
    """Permanent, untruncated Python recorder for all signs, NPC speech, and story text."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path else config.SIGNS_AND_SPEECH_FILE
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                self.entries = data["entries"]
        except Exception as e:
            print(f"[DialogueRecorder] Warning: Could not read {self.storage_path}: {e}")

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_entries": len(self.entries),
                "last_updated": datetime.now().isoformat(),
                "entries": self.entries,
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[DialogueRecorder] Warning: Could not save ({e})")

    @staticmethod
    def classify_text_kind(text: str, screen_mode: str = "OVERWORLD") -> str:
        """Classifies recorded text as SIGN, NPC_SPEECH, BATTLE_TEXT, or SYSTEM_TEXT."""
        low = text.lower()
        if screen_mode == "BATTLE":
            return "BATTLE_TEXT"
        if any(k in low for k in ["tips!", "sign", "trainer tips", "house", "lab", "gym", "route ", "city", "town", "notice", "mailbox"]):
            if ":" not in text[:25]:
                return "SIGN"
        if ":" in text[:30] or any(k in low for k in ["oak", "mom", "prof.", "nurse", "clerk", "rival"]):
            return "NPC_SPEECH"
        return "DIALOGUE"

    def record(
        self,
        text: str,
        turn_id: int,
        run_id: str = "default",
        area: str = "Unknown",
        screen_mode: str = "OVERWORLD",
        screenshot_path: str = "",
        kind: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Records any sign, NPC speech, or on-screen dialogue permanently without truncation.
        Merges consecutive multi-page speech lines from the same conversation, and deduplicates
        back-to-back identical frames without losing distinct occurrences.
        """
        if not text or not isinstance(text, str):
            return None
        clean = " ".join(text.strip().split())
        if len(clean) < 2 or clean.upper() == "NULL":
            return None

        clean_area = (area or "Unknown").strip()
        resolved_kind = kind or self.classify_text_kind(clean, screen_mode=screen_mode)

        if self.entries:
            last = self.entries[-1]
            last_text = last.get("text", "")
            last_turn = int(last.get("turn_id", -999))
            last_run = last.get("run_id", "")

            # Exact or near-duplicate of immediately preceding frame while dialog is still on screen
            sim = SequenceMatcher(None, clean.lower(), last_text.lower()).ratio()
            if sim >= 0.88 or clean.lower() in last_text.lower():
                last["last_seen_turn"] = turn_id
                last["seen_count"] = int(last.get("seen_count", 1)) + 1
                if screenshot_path:
                    last["last_screenshot"] = screenshot_path
                self._save()
                return None

            # Multi-page speech bubble continuation in same run/area within 3 turns
            if (
                last_run == run_id
                and last.get("area") == clean_area
                and 0 <= (turn_id - last_turn) <= 3
                and resolved_kind != "BATTLE_TEXT"
                and last.get("kind") != "BATTLE_TEXT"
            ):
                # Append cleanly without truncating
                last["text"] = f"{last_text} | {clean}"
                last["turn_id"] = turn_id
                last["last_seen_turn"] = turn_id
                last["timestamp"] = datetime.now().isoformat()
                if screenshot_path:
                    last["last_screenshot"] = screenshot_path
                self._save()
                return last

        entry = {
            "id": len(self.entries) + 1,
            "run_id": run_id,
            "turn_id": turn_id,
            "last_seen_turn": turn_id,
            "seen_count": 1,
            "area": clean_area,
            "screen_mode": screen_mode,
            "kind": resolved_kind,
            "text": clean,
            "screenshot_path": screenshot_path,
            "timestamp": datetime.now().isoformat(),
        }
        self.entries.append(entry)
        self._save()
        return entry

    def search(self, query: str = "", area: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Searches all permanently recorded signs and speech across all runs."""
        if not self.entries:
            return []
        q_tokens = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
        area_low = (area or "").lower().strip()

        scored: List[tuple] = []
        for idx, entry in enumerate(self.entries):
            e_area = entry.get("area", "").lower()
            e_text = entry.get("text", "").lower()
            score = 0
            if area_low and (area_low in e_area or e_area in area_low):
                score += 5
            if q_tokens:
                e_tokens = set(re.findall(r"[a-z0-9]+", f"{e_area} {e_text}"))
                score += len(q_tokens & e_tokens) * 2
            if entry.get("kind") == "SIGN":
                score += 1
            if not q_tokens and not area_low:
                score += 1
            if score > 0:
                scored.append((score, idx, entry))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def get_recent(self, limit: int = 8, include_battle: bool = False) -> List[Dict[str, Any]]:
        """Returns the most recent recorded signs and dialogue entries."""
        filtered = self.entries if include_battle else [e for e in self.entries if e.get("kind") != "BATTLE_TEXT"]
        return filtered[-limit:]

    def format_recall_prompt(self, area: str = "", query: str = "", recent_n: int = 5, area_n: int = 5) -> str:
        """Formats recent and area-relevant recorded signs/speech for the LLM turn prompt."""
        recent = self.get_recent(limit=recent_n, include_battle=False)
        area_matches = self.search(query=query, area=area, limit=area_n) if (area or query) else []

        seen_ids = set()
        lines: List[str] = []
        if recent:
            lines.append(f"[Signs & Speech Archive (Total Recorded Across Runs: {len(self.entries)}) - Recent]:")
            for e in recent:
                seen_ids.add(e.get("id"))
                lines.append(f"- [{e.get('kind','TEXT')} | {e.get('area','?')}] (T{e.get('turn_id','?')}): \"{e.get('text','')}\"")

        extra = [e for e in area_matches if e.get("id") not in seen_ids and e.get("kind") != "BATTLE_TEXT"]
        if extra:
            lines.append(f"[Past Signs/Speech Recalled for '{area or query}']:")
            for e in extra:
                lines.append(f"- [{e.get('kind','TEXT')} | {e.get('area','?')}] (Run {e.get('run_id','?')}): \"{e.get('text','')}\"")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clears the signs and speech archive (used only on explicit --reset)."""
        self.entries = []
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass
