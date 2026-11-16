"""
Goal Tracker & Discovery-First State Manager for YellowAI (Local Ollama).
No hardcoded playthrough script: discovers verified game state directly from in-game menus
(Start Menu, Trainer Card, Party, Bag, Pokédex), signs, NPC dialogue, and online web search,
while permanently dropping completed goals so finished tasks are never repeated.
"""

from difflib import SequenceMatcher
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

_STOPWORDS = {
    "the", "and", "to", "in", "of", "for", "a", "an", "on", "at", "from",
    "into", "with", "then", "go", "walk", "head", "get", "check", "current",
}

DISCOVERY_MENU_TARGETS = ["TRAINER_CARD", "PARTY_MENU", "BAG_MENU", "POKEDEX"]


class GoalTracker:
    """Persistent Grand Goal, Active Goal, Menu-Verified Profile, and Dropped-Goal Tombstones."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = Path(state_path) if state_path else config.GOAL_STATE_FILE
        self.grand_goal: str = (
            "Complete Pokémon Recharged Yellow: earn all 8 Kanto Badges, defeat the Elite Four and Champion."
        )
        self.active_goal: str = (
            "Check START menu (Trainer Card, Party, Bag, Pokédex) and nearby signs/NPCs, then search online to set the next objective."
        )
        self.next_step: str = "If on Title Screen press A/START; once in-game, press START to inspect status or read nearby signs."
        self.completion_criteria: str = "Verify status via Start Menu/signs and synthesize next concrete objective."
        self.current_area: str = "Unknown"
        self.area_history: List[str] = []
        self.discovered_facts: List[str] = []
        self.verified_profile: Dict[str, Any] = {
            "badges": "Unknown (check Trainer Card)",
            "party": "Unknown (check Party menu)",
            "bag": "Unknown (check Bag)",
            "pokedex": "Unknown (check Pokédex)",
        }
        self.completed_goals: List[str] = []
        self.turns_without_change: int = 0

        # Per-run menu inspection state (resets each run unless already mid-audit)
        self.inspected_menus_this_run: List[str] = []
        self.discovery_done_this_run: bool = False
        self.load()

    @staticmethod
    def _tokens(text: str) -> set:
        return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 1}

    @classmethod
    def _goals_match(cls, g1: str, g2: str) -> bool:
        s1 = (g1 or "").strip().lower()
        s2 = (g2 or "").strip().lower()
        if not s1 or not s2:
            return False
        if s1 == s2 or s1 in s2 or s2 in s1:
            return True
        t1, t2 = cls._tokens(s1), cls._tokens(s2)
        if t1 and t2 and (len(t1 & t2) / float(min(len(t1), len(t2)))) >= 0.6:
            return True
        return SequenceMatcher(None, s1, s2).ratio() >= 0.72

    def is_goal_completed(self, goal_text: str) -> bool:
        return any(self._goals_match(goal_text, cg) for cg in self.completed_goals)

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.grand_goal = data.get("grand_goal", self.grand_goal)
                self.active_goal = data.get("active_goal", self.active_goal)
                self.next_step = data.get("next_step", self.next_step)
                self.completion_criteria = data.get("completion_criteria", self.completion_criteria)
                self.current_area = data.get("current_area", self.current_area)
                self.area_history = list(data.get("area_history", []))[-8:]
                self.discovered_facts = list(data.get("discovered_facts", []))[-25:]
                if isinstance(data.get("verified_profile"), dict):
                    self.verified_profile.update(data["verified_profile"])
                self.completed_goals = list(data.get("completed_goals", []))[-60:]
                self.turns_without_change = int(data.get("turns_without_change", 0))
                if self.is_goal_completed(self.active_goal):
                    self.active_goal = "Inspect Start menu / signs and search online for next uncompleted objective."
        except Exception as e:
            print(f"[GoalTracker] Warning: Could not load goal state ({e})")

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "grand_goal": self.grand_goal,
                "active_goal": self.active_goal,
                "next_step": self.next_step,
                "completion_criteria": self.completion_criteria,
                "current_area": self.current_area,
                "area_history": self.area_history[-8:],
                "discovered_facts": self.discovered_facts[-25:],
                "verified_profile": self.verified_profile,
                "completed_goals": self.completed_goals[-60:],
                "turns_without_change": self.turns_without_change,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[GoalTracker] Warning: Could not save goal state ({e})")

    def complete_goal(self, goal_text: Optional[str]) -> Optional[str]:
        """Permanently drops a finished goal so it cannot remain active or be re-added."""
        clean = (goal_text or "").strip()
        if not clean or clean.upper() == "NULL":
            return None
        if clean.upper() in {"CURRENT", "ACTIVE", "DONE", "FINISHED", "ALL"}:
            clean = self.active_goal
        if not clean:
            return None

        if not self.is_goal_completed(clean):
            self.completed_goals.append(clean)
            self.completed_goals = self.completed_goals[-60:]

        if self._goals_match(clean, self.active_goal) or self.is_goal_completed(self.active_goal):
            self.active_goal = (
                f"Determine next step in {self.current_area} (check signs/NPCs, Start menu, or web search)."
            )
            self.next_step = "Read nearby signs, check Start menu if needed, or search online for next step."
            self.completion_criteria = "Next objective verified and set."
            self.turns_without_change = 0

        self.save()
        return clean

    def record_menu_observation(
        self,
        screen_kind: Optional[str],
        discovered_fact: Optional[str] = None,
        dialogue_text: Optional[str] = None,
    ) -> None:
        """Captures verified facts from Start Menu screens (Trainer Card, Party, Bag, Pokédex)."""
        sk = (screen_kind or "").upper().strip()
        if sk in DISCOVERY_MENU_TARGETS and sk not in self.inspected_menus_this_run:
            self.inspected_menus_this_run.append(sk)

        combined = " | ".join(
            p.strip() for p in (discovered_fact or "", dialogue_text or "") if p and p.strip() and p.strip().upper() != "NULL"
        )
        if not combined:
            return

        if sk == "TRAINER_CARD" or "badge" in combined.lower() or "money" in combined.lower():
            self.verified_profile["badges"] = combined[:180]
            if "TRAINER_CARD" not in self.inspected_menus_this_run:
                self.inspected_menus_this_run.append("TRAINER_CARD")
        if sk == "PARTY_MENU" or any(w in combined.lower() for w in ("party:", "team:", "lv", "hp ")):
            self.verified_profile["party"] = combined[:180]
            if "PARTY_MENU" not in self.inspected_menus_this_run:
                self.inspected_menus_this_run.append("PARTY_MENU")
        if sk == "BAG_MENU" or any(w in combined.lower() for w in ("bag:", "key item", "poké ball", "poke ball", "potion", "parcel")):
            self.verified_profile["bag"] = combined[:180]
            if "BAG_MENU" not in self.inspected_menus_this_run:
                self.inspected_menus_this_run.append("BAG_MENU")
        if sk == "POKEDEX" or "pokédex" in combined.lower() or "pokedex" in combined.lower() or "seen" in combined.lower():
            self.verified_profile["pokedex"] = combined[:180]
            if "POKEDEX" not in self.inspected_menus_this_run:
                self.inspected_menus_this_run.append("POKEDEX")

    def get_uninspected_menus(self) -> List[str]:
        return [m for m in DISCOVERY_MENU_TARGETS if m not in self.inspected_menus_this_run]

    def format_discovery_directive(self, discovery_turn: int, max_turns: int) -> str:
        """Builds instructions guiding the VLM to inspect in-game menus before planning."""
        remaining = self.get_uninspected_menus()
        inspected = ", ".join(self.inspected_menus_this_run) or "None yet"
        return (
            f"[DISCOVERY AUDIT MODE ({discovery_turn}/{max_turns}) - LEARN STATE FROM IN-GAME MENUS]\n"
            f"- Inspected so far this run: {inspected}\n"
            f"- Remaining to inspect: {', '.join(remaining) if remaining else 'All core menus checked -> press B to return to Overworld and set ready_to_plan_goal=true'}\n"
            "- Instructions: If on Title/Continue screen, press A/START to load into the world first.\n"
            "  Once in-game, press START to open the Start Menu, then navigate with UP/DOWN and A to inspect "
            "Trainer Card (<PLAYER> for Badges), Pokémon (Party levels/HP), Bag (Items/Key Items), and Pokédex.\n"
            "  Transcribe what each menu shows into `discovered_fact` and set `screen_kind`.\n"
            "  Once done (or if you have enough info), press B back to Overworld and set `ready_to_plan_goal`: true!"
        )

    def apply_synthesized_goal(
        self,
        new_active_goal: Optional[str],
        next_step: Optional[str],
        completion_criteria: Optional[str] = None,
        grand_goal: Optional[str] = None,
        confirmed_completed_goals: Optional[List[str]] = None,
    ) -> None:
        """Drops any confirmed-completed goals and commits a freshly researched objective."""
        for done in (confirmed_completed_goals or []):
            if isinstance(done, str) and done.strip():
                self.complete_goal(done.strip())

        if grand_goal and grand_goal.strip() and grand_goal.strip().upper() != "NULL":
            self.grand_goal = grand_goal.strip()

        if new_active_goal and new_active_goal.strip() and not self.is_goal_completed(new_active_goal.strip()):
            self.active_goal = new_active_goal.strip()
        if next_step and next_step.strip() and not self.is_goal_completed(next_step.strip()):
            self.next_step = next_step.strip()
        if completion_criteria and completion_criteria.strip():
            self.completion_criteria = completion_criteria.strip()
        self.turns_without_change = 0
        self.save()

    def update_from_turn(
        self,
        area: Optional[str] = None,
        discovered_fact: Optional[str] = None,
        dialogue_text: Optional[str] = None,
        screen_kind: Optional[str] = None,
        active_goal_update: Optional[str] = None,
        next_step: Optional[str] = None,
        grand_goal_update: Optional[str] = None,
        screen_changed: bool = True,
    ) -> None:
        progress = False
        if area and area.strip() and area.strip().upper() != "NULL":
            a_clean = area.strip()
            if a_clean != self.current_area:
                self.current_area = a_clean
                if not self.area_history or self.area_history[-1] != a_clean:
                    self.area_history.append(a_clean)
                    self.area_history = self.area_history[-8:]
                progress = True

        self.record_menu_observation(screen_kind=screen_kind, discovered_fact=discovered_fact, dialogue_text=dialogue_text)

        if discovered_fact and discovered_fact.strip() and discovered_fact.strip().upper() != "NULL":
            fact = " ".join(discovered_fact.strip().split())
            if not any(fact.lower() == f.lower() for f in self.discovered_facts):
                self.discovered_facts.append(fact)
                self.discovered_facts = self.discovered_facts[-25:]
                progress = True

        if grand_goal_update and grand_goal_update.strip() and grand_goal_update.strip().upper() != "NULL":
            self.grand_goal = grand_goal_update.strip()

        if active_goal_update and active_goal_update.strip() and active_goal_update.strip().upper() != "NULL":
            candidate = active_goal_update.strip()
            if not self.is_goal_completed(candidate):
                if candidate != self.active_goal:
                    progress = True
                self.active_goal = candidate

        if next_step and next_step.strip() and next_step.strip().upper() != "NULL":
            if not self.is_goal_completed(next_step.strip()):
                self.next_step = next_step.strip()

        self.turns_without_change = 0 if (progress or screen_changed) else (self.turns_without_change + 1)
        self.save()

    def is_drifting(self, threshold: int = 10) -> bool:
        return self.turns_without_change >= threshold

    def format_pinned_anchor(self) -> str:
        lines = [
            "[PINNED GOAL ANCHOR - LEARNED FROM MENUS, SIGNS & ONLINE SEARCH]",
            f"- Grand Goal: {self.grand_goal}",
            f"- Current Location: {self.current_area}",
            f"- Active Goal: >>> {self.active_goal} <<<",
            f"- Immediate Step: {self.next_step}",
            f"- Done When: {self.completion_criteria}",
            (
                "- Verified Menu Profile: "
                f"Badges=[{self.verified_profile.get('badges')}] | "
                f"Party=[{self.verified_profile.get('party')}] | "
                f"Bag=[{self.verified_profile.get('bag')}] | "
                f"Pokédex=[{self.verified_profile.get('pokedex')}]"
            ),
        ]
        if self.discovered_facts:
            lines.append("- Other Verified Facts: " + " | ".join(self.discovered_facts[-6:]))
        if self.area_history:
            lines.append("- Recent Areas: " + " -> ".join(self.area_history[-5:]))
        if self.completed_goals:
            lines.append("- Already Completed / Dropped Goals (DO NOT REPEAT): " + " | ".join(self.completed_goals[-6:]))
        if self.turns_without_change >= 8:
            lines.append(
                f"- [STUCK / DRIFT NOTICE]: Screen unchanged or no progress for {self.turns_without_change} turns. "
                "Try checking START menu, reading a nearby sign/NPC, pressing B to close menus, or setting `web_search_query`."
            )
        return "\n".join(lines)

    def reset(self) -> None:
        if self.state_path.exists():
            try:
                self.state_path.unlink()
            except Exception:
                pass
        self.__init__(state_path=self.state_path)
        self.save()
