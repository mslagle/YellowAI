"""
Thinking Trace & Distilled Strategic Insight Store for YellowAI.

Architectural Decision on Saving Thinking Logic / Tokens Across Runs:
1. Raw thinking traces (`message.thinking`) are archived to `state/thinking_traces.jsonl`
   for auditing, debugging, and post-run analysis, along with token counts and outcomes.
2. Raw thinking tokens are NOT re-injected verbatim into future turns or runs, because
   multi-hundred-token scratchpads rapidly exhaust 12GB VRAM KV caches and induce reasoning loops.
3. Instead, verified strategic takeaways ("Thinking Insights") from successful rescues, battles,
   web discoveries, and navigation breakthroughs are distilled into `state/thinking_insights.json`
   and selectively recalled into future runs by area/objective relevance!
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class ThinkingStore:
    """Persists raw thinking traces and distilled reusable insights across runs."""

    def __init__(
        self,
        traces_path: Optional[Path] = None,
        insights_path: Optional[Path] = None,
    ):
        self.traces_path = Path(traces_path) if traces_path else config.THINKING_TRACES_FILE
        self.insights_path = Path(insights_path) if insights_path else config.THINKING_INSIGHTS_FILE
        self.insights: List[Dict[str, Any]] = []
        self._load_insights()

    def _load_insights(self) -> None:
        if not self.insights_path.exists():
            return
        try:
            with open(self.insights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("insights"), list):
                self.insights = data["insights"]
        except Exception as e:
            print(f"[ThinkingStore] Warning: Could not load insights ({e})")

    def _save_insights(self) -> None:
        try:
            self.insights_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_insights": len(self.insights),
                "last_updated": datetime.now().isoformat(),
                "insights": self.insights,
            }
            with open(self.insights_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ThinkingStore] Warning: Could not save insights ({e})")

    def record_trace(
        self,
        run_id: str,
        turn_id: int,
        area: str,
        goal: str,
        thinking_text: str,
        thinking_tokens: int,
        decision_summary: str,
        buttons: List[str],
        outcome: str = "PENDING",
    ) -> None:
        """Appends a raw thinking trace record to disk without polluting future context windows."""
        if not getattr(config, "OLLAMA_SAVE_THINKING_TRACES", True):
            return
        clean_thinking = (thinking_text or "").strip()
        if not clean_thinking:
            return
        record = {
            "run_id": run_id,
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "area": area,
            "goal": goal,
            "thinking_tokens": thinking_tokens,
            "thinking": clean_thinking,
            "decision": decision_summary,
            "buttons": buttons,
            "outcome": outcome,
        }
        try:
            self.traces_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.traces_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ThinkingStore] Warning: Could not write thinking trace ({e})")

    def add_insight(
        self,
        insight: str,
        area: str = "General",
        category: str = "STRATEGY",
        source: str = "THINKING",
        confidence: float = 1.0,
    ) -> bool:
        """
        Stores a concise, distilled strategic insight for future runs.
        Deduplicates near-identical insights and caps the active insight store at 50 items.
        """
        clean = " ".join((insight or "").strip().split())
        if len(clean) < 10:
            return False

        low = clean.lower()
        for existing in self.insights:
            ex_text = existing.get("insight", "").lower()
            if low == ex_text or low in ex_text or ex_text in low:
                existing["uses"] = int(existing.get("uses", 1)) + 1
                existing["confidence"] = min(5.0, float(existing.get("confidence", 1.0)) + 0.25)
                existing["last_reinforced"] = datetime.now().isoformat()
                self._save_insights()
                return False

        self.insights.append({
            "id": len(self.insights) + 1,
            "area": (area or "General").strip(),
            "category": category.upper().strip(),
            "source": source.upper().strip(),
            "insight": clean[:350],
            "confidence": confidence,
            "uses": 1,
            "created_at": datetime.now().isoformat(),
            "last_reinforced": datetime.now().isoformat(),
        })
        if len(self.insights) > 50:
            # Keep highest-confidence and most recent insights
            self.insights.sort(key=lambda x: (float(x.get("confidence", 1.0)), x.get("id", 0)))
            self.insights = self.insights[-50:]
        self._save_insights()
        return True

    def get_relevant_insights(self, area: str = "", goal: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves the most relevant distilled thinking insights for the current area & goal."""
        if not self.insights:
            return []
        ctx_tokens = {w for w in re.findall(r"[a-z0-9]+", f"{area} {goal}".lower()) if len(w) > 2}
        area_low = (area or "").lower().strip()

        scored: List[tuple] = []
        for idx, item in enumerate(self.insights):
            i_area = item.get("area", "").lower()
            i_text = item.get("insight", "").lower()
            score = float(item.get("confidence", 1.0))
            if area_low and (area_low in i_area or i_area in area_low or i_area == "general"):
                score += 4.0
            if ctx_tokens:
                item_tokens = set(re.findall(r"[a-z0-9]+", f"{i_area} {i_text}"))
                score += len(ctx_tokens & item_tokens) * 1.5
            scored.append((score, idx, item))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def format_insights_prompt(self, area: str = "", goal: str = "", limit: int = 5) -> str:
        """Formats distilled insights from past thinking/runs for injection into the prompt."""
        relevant = self.get_relevant_insights(area=area, goal=goal, limit=limit)
        if not relevant:
            return ""
        lines = ["[Distilled Wisdom From Past Thinking & Runs]:"]
        for item in relevant:
            lines.append(f"- ({item.get('area', 'General')}) {item.get('insight', '')}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.insights = []
        for p in (self.traces_path, self.insights_path):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
