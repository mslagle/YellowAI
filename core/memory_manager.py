"""
Memory Manager for YellowAI.
Maintains hierarchical short, medium, and long-term memory with JSON persistence.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

DEFAULT_AREA_WAYPOINTS: Dict[str, List[str]] = {
    "Pallet Town": [
        "Triggering Professor Oak: If Oak's Lab is empty, walk NORTH into the tall grass exit at the top of Pallet Town to trigger Professor Oak's Pikachu capture event.",
        "Oak's Lab: SOUTHEAST (bottom-right) large building with green/teal door. Inside, walk to the green Starter Table on the right and press A on the Poké Ball to claim Starter Pikachu!",
        "Mom's House (Red's House): NORTHWEST (top-left) house with red door. Talk to Mom at the dining table to say goodbye / heal your party before leaving town.",
        "Daisy's House (Rival's House): NORTHEAST (top-right) house. Talk to Daisy at the table to receive the Town Map.",
        "Route 1: North exit through the 2-tile gap into tall grass (leads NORTH to Viridian City)."
    ],
    "Route 1": [
        "1st Ledge (South): The opening is immediately LEFT of the Route 1 signpost! Walk LEFT to the grass gap and walk UP onto the road.",
        "2nd Barrier (Ledge & Pine Trees): Middle of Route 1 has a brown ledge on the left and 3 pine trees on the right. If trapped below the ledge, step DOWN to the dirt road, walk EAST (RIGHT) past the 3 pine trees, then walk UP around the east side of the trees!",
        "Ledge Rule: Ledges can only be hopped going SOUTH. You CANNOT jump UP through brown ledges. Do NOT press DOWN into ledges or you will jump back down to previous sections!",
        "Path to Viridian City: Continue north along the dirt road and paths through Route 1 to reach Viridian City."
    ],
    "Viridian City": [
        "Poké Mart: Blue-roof building on east/right (collect Oak's Parcel at counter).",
        "Pokémon Center: Red-roof building on southwest/left (free party healing).",
        "Route 22: West path (Pokémon League gate / early Rival battle).",
        "Route 2: North gate toward Viridian Forest & Pewter City."
    ],
    "Route 22": [
        "East Path: Returns to Viridian City.",
        "West Path: Pokémon League Reception Gate (Rival battle encounter)."
    ],
    "Route 2 / Viridian Forest": [
        "South Gate: Leads into Viridian Forest from Route 2.",
        "North Gate: Leads out of Viridian Forest to Pewter City."
    ],
    "Pewter City": [
        "Pewter Gym (Brock): Building with pillars on northwest/top-left (Boulder Badge).",
        "Pokémon Center: Red-roof building in center.",
        "Poké Mart: Blue-roof building in center-east.",
        "Route 3: East exit toward Mt. Moon."
    ]
}

AREA_CONNECTIONS: Dict[str, List[str]] = {
    "Pallet Town": ["Route 1 (North to Viridian City)"],
    "Route 1": ["Pallet Town (South)", "Viridian City (North)"],
    "Viridian City": ["Route 1 (South)", "Route 22 (West - League Gate/Rival/Wild Mankey & Nidoran)", "Route 2 (North - Viridian Forest)"],
    "Route 22": ["Viridian City (East)", "Pokémon League Reception Gate (West)"],
    "Route 2 / Viridian Forest": ["Viridian City (South)", "Pewter City (North - Brock's Gym)"],
    "Pewter City": ["Route 2 (South)", "Route 3 (East - Mt. Moon)"]
}


class MemoryManager:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.turn_history: List[Dict[str, Any]] = []
        
        # Multi-quest tracking & active quest management
        self.quests: List[str] = ["Leave Red's bedroom, go downstairs, and talk to Mom"]
        self.priority_quest: Optional[str] = "Leave Red's bedroom, go downstairs, and talk to Mom"
        self.current_area: str = "Pallet Town (Red's Bedroom)"
        self.waypoints: Dict[str, List[str]] = dict(DEFAULT_AREA_WAYPOINTS)
        self.short_term_memory: str = "In Red's bedroom (2F upstairs). Need to walk UP and RIGHT to the staircase in the top-right corner to go downstairs."
        self.medium_term_memory: str = "Pallet Town: Go downstairs, talk to Mom, head outside, meet Professor Oak, choose starter Pokémon, and defeat Rival."
        self.long_term_memory: str = "Grand Goal: Defeat all 8 Kanto Gym Leaders, conquer the Elite Four, and become Pokémon Champion."
        # Topological route breadcrumbs & immediate micro-step
        self.route_trail: List[str] = [
            "Red's Bedroom (2F)"
        ]
        self.micro_step: Optional[str] = "Walk UP and RIGHT to staircase in top-right corner to go downstairs"

        # Structured Party, Supplies, Badges, and ROM Knowledge Notes
        self.team: List[Dict[str, Any]] = []
        self.supplies: Dict[str, Any] = {
            "poke_balls": 0,
            "potions": 0,
            "town_map": False,
            "money": 3000
        }
        self.badges: int = 0
        self.last_items_update: Optional[str] = None
        self.knowledge_notes: List[str] = [
            "Yellow Recharged: Poké Balls available immediately at Viridian Mart with no parcel required to unlock."
        ]
        
        self.load()

    def update_micro_step(self, new_step: Optional[str]):
        """Sets or updates the immediate physical micro-step."""
        if new_step and new_step.strip() and new_step.strip().upper() != "NULL":
            self.micro_step = new_step.strip()
            self.save()
        elif new_step and new_step.strip().upper() == "NULL":
            self.micro_step = None
            self.save()

    def add_route_node(self, landmark_name: str):
        """Pushes a landmark to route_trail, maintaining max 4 items and avoiding duplicates."""
        if not landmark_name or not landmark_name.strip():
            return
        node = landmark_name.strip()
        if not self.route_trail or self.route_trail[-1] != node:
            self.route_trail.append(node)
            if len(self.route_trail) > 4:
                self.route_trail = self.route_trail[-4:]
            self.save()

    @property
    def active_quest(self) -> Optional[str]:
        """Returns the highest priority quest for backward compatibility."""
        return self.priority_quest

    @active_quest.setter
    def active_quest(self, value: Optional[str]):
        if value and value.strip():
            self.set_priority_quest(value.strip())
        else:
            self.priority_quest = None

    def add_waypoint(self, area: str, poi: str):
        """Dynamically adds a discovered landmark or obstacle to the specified area's waypoints."""
        if not area or not poi or not poi.strip():
            return
        poi_clean = poi.strip()
        if area not in self.waypoints:
            self.waypoints[area] = []
        if poi_clean not in self.waypoints[area]:
            self.waypoints[area].append(poi_clean)
            print(f"[MemoryManager] Discovered new POI in {area}: '{poi_clean}'")
            self.save()

    def add_knowledge_note(self, note: str):
        """Appends a verified game/ROM knowledge note."""
        if not note or not note.strip():
            return
        n_clean = note.strip()
        if n_clean not in self.knowledge_notes:
            self.knowledge_notes.append(n_clean)
            if len(self.knowledge_notes) > 8:
                self.knowledge_notes = self.knowledge_notes[-8:]
            print(f"[MemoryManager] Added ROM Knowledge Note: '{n_clean}'")
            self.save()

    def update_supplies(
        self,
        poke_balls: Optional[int] = None,
        potions: Optional[int] = None,
        money: Optional[int] = None,
        town_map: Optional[bool] = None
    ):
        """Updates quantities in the supplies tracker."""
        changed = False
        if poke_balls is not None:
            self.supplies["poke_balls"] = max(0, poke_balls)
            changed = True
        if potions is not None:
            self.supplies["potions"] = max(0, potions)
            changed = True
        if money is not None:
            self.supplies["money"] = max(0, money)
            changed = True
        if town_map is not None:
            self.supplies["town_map"] = town_map
            changed = True
        if changed:
            self.save()

    def apply_items_update(self, update_text: str):
        """Parses natural language item update from model (e.g. '+5 Poké Balls', 'Bought 5 Poké Balls')."""
        if not update_text or not isinstance(update_text, str) or update_text.strip().upper() == "NULL":
            return
        txt = update_text.lower().strip()
        if self.last_items_update == txt:
            return  # Prevent duplicate addition on back-to-back turns
        self.last_items_update = txt
        import re
        ball_match = re.search(r'(\d+)\s*(?:pok[eé]\s*ball|ball)', txt)
        if ball_match:
            qty = int(ball_match.group(1))
            if '+' in txt or 'bought' in txt or 'got' in txt:
                self.supplies["poke_balls"] = self.supplies.get("poke_balls", 0) + qty
            else:
                self.supplies["poke_balls"] = qty
            self.save()
            print(f"[MemoryManager] Supplies updated -> Poké Balls: {self.supplies['poke_balls']}")

        potion_match = re.search(r'(\d+)\s*potion', txt)
        if potion_match:
            qty = int(potion_match.group(1))
            if '+' in txt or 'bought' in txt or 'got' in txt:
                self.supplies["potions"] = self.supplies.get("potions", 0) + qty
            else:
                self.supplies["potions"] = qty
            self.save()
            print(f"[MemoryManager] Supplies updated -> Potions: {self.supplies['potions']}")

    def apply_party_update(self, update_text: str):
        """Parses natural language party update (e.g. 'Caught Pidgey Lv3', 'Received Pikachu Lv5', 'Pikachu reached Lv7')."""
        if not update_text or not isinstance(update_text, str) or update_text.strip().upper() == "NULL":
            return
        txt = update_text.strip()
        import re
        catch_match = re.search(
            r'(?:caught|added|got|received|chose|obtained|claimed|starter)\s+(?:starter\s+)?([A-Za-z]+)(?:\s*(?:lv|level)\s*(\d+))?',
            txt,
            re.IGNORECASE
        )
        if catch_match:
            species = catch_match.group(1).capitalize()
            default_lvl = 5 if species.lower() == "pikachu" else 3
            level = int(catch_match.group(2)) if catch_match.group(2) else default_lvl
            if not any(p.get("species", "").lower() == species.lower() for p in self.team):
                is_lead = len(self.team) == 0
                moves = ["ThunderShock", "Growl"] if species.lower() == "pikachu" else []
                self.team.append({"species": species, "level": level, "lead": is_lead, "key_moves": moves})
                print(f"[MemoryManager] Team member added: {species} Lv{level}")
                self.save()

    def add_quest(self, quest: str, is_priority: bool = False):
        """Adds a quest to the log. Optionally sets it as highest priority."""
        if not quest or not quest.strip():
            return
        q = quest.strip()
        if q not in self.quests:
            self.quests.append(q)
        if is_priority or self.priority_quest is None:
            self.priority_quest = q
        self.save()

    def complete_quest(self, query: str) -> Optional[str]:
        """
        Marks a quest as completed and removes it from the quest log.
        Promotes the next available quest if the completed one was highest priority.
        """
        if not query or not query.strip():
            return None
        q_lower = query.strip().lower()

        target_q = None
        for q in self.quests:
            if q.lower() == q_lower or q_lower in q.lower() or q.lower() in q_lower:
                target_q = q
                break

        if target_q:
            self.quests.remove(target_q)
            if self.priority_quest == target_q:
                self.priority_quest = self.quests[0] if self.quests else None
            self.save()
            return target_q
        elif query.strip().upper() in ["ALL", "COMPLETED"]:
            if self.priority_quest:
                old = self.priority_quest
                if old in self.quests:
                    self.quests.remove(old)
                self.priority_quest = self.quests[0] if self.quests else None
                self.save()
                return old

        return None

    def set_priority_quest(self, query: str):
        """Sets a quest as highest priority, adding it if not already present."""
        if not query or not query.strip():
            return
        q_clean = query.strip()
        for q in self.quests:
            if q.lower() == q_clean.lower() or q_clean.lower() in q.lower():
                self.priority_quest = q
                self.save()
                return
        self.quests.append(q_clean)
        self.priority_quest = q_clean
        self.save()

    def set_active_quest(self, new_quest: str):
        """Backward-compatible alias for adding/prioritizing a quest."""
        if new_quest and new_quest.strip():
            self.add_quest(new_quest.strip(), is_priority=True)

    def clear_active_quest(self):
        """Backward-compatible alias for completing top priority quest."""
        if self.priority_quest:
            self.complete_quest(self.priority_quest)

    def set_current_area(self, area: str):
        """Sets the active town or route."""
        if area and area.strip():
            self.current_area = area.strip()
            self.save()

    def _auto_detect_area(self, text: str):
        """Detects if memory text indicates a new town or route."""
        if not text:
            return
        for area_name in self.waypoints.keys():
            if area_name.lower() in text.lower():
                if self.current_area != area_name:
                    print(f"[MemoryManager] Area changed: {self.current_area} -> {area_name}")
                    self.current_area = area_name
                break

    def update_short_term(self, new_memory: str):
        if new_memory and new_memory.strip():
            self.short_term_memory = new_memory.strip()
            self._auto_detect_area(new_memory)
            self.save()

    def update_medium_term(self, new_memory: str):
        if new_memory and new_memory.strip():
            self.medium_term_memory = new_memory.strip()
            self._auto_detect_area(new_memory)
            self.save()

    def update_long_term(self, new_memory: str):
        if new_memory and new_memory.strip():
            self.long_term_memory = new_memory.strip()
            self.save()

    def log_turn(self, turn_id: int, action: str, thought: str = None, reasoning: str = None):
        entry = {
            "turn": turn_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reasoning": reasoning or "",
            "thought": thought or ""
        }
        self.turn_history.append(entry)
        # Keep last 50 turns in memory to avoid unbounded growth in RAM
        if len(self.turn_history) > 100:
            self.turn_history = self.turn_history[-100:]

    def get_recent_actions(self, n: int = 6) -> str:
        """Returns string representation of the last N actions taken."""
        if not self.turn_history:
            return "No previous actions."
        recent = self.turn_history[-n:]
        return " -> ".join([f"T{e['turn']}:{e['action']}" for e in recent])

    def get_memory_context_prompt(self) -> str:
        """Formats the memory tiers, quest log, party, supplies, and area waypoints into concise prompt text."""
        lines = ["[Session Memory]"]

        # Secondary Quests (Top priority quest is prominently displayed in turn prompt header)
        secondary_quests = [q for q in self.quests if q != self.priority_quest]
        if secondary_quests:
            lines.append(f"- Secondary Quests: {', '.join(secondary_quests)}")

        # Strategy 2 & 3: Route Trail and Immediate Micro-Step
        if self.route_trail:
            lines.append(f"- Recent Route: {' -> '.join(self.route_trail)}")
        if self.micro_step:
            lines.append(f"- Immediate Micro-Step: >>> {self.micro_step} <<<")

        # Structured Party & Supplies Summary
        party_str = ", ".join([f"{p.get('species','Pokemon')} Lv{p.get('level','?')}" for p in self.team]) or "None"
        lines.append(f"- Party: [{party_str}] | Badges: {self.badges}/8")
        supplies_str = f"Poké Balls: {self.supplies.get('poke_balls', 0)} | Potions: {self.supplies.get('potions', 0)} | Money: ¥{self.supplies.get('money', 0)}"
        lines.append(f"- Supplies: [{supplies_str}]")

        lines.append(f"- Strategy: {self.long_term_memory}")
        lines.append(f"- Milestone: {self.medium_term_memory}")
        lines.append(f"- Immediate: {self.short_term_memory}")

        # Regional Exits from Current Area
        exits = AREA_CONNECTIONS.get(self.current_area, [])
        if exits:
            lines.append(f"- Regional Exits: {', '.join(exits)}")

        # Local landmarks for current area
        waypoints = self.waypoints.get(self.current_area, [])
        if waypoints:
            lines.append(f"\n[Local Waypoints ({self.current_area})]:")
            for wp in waypoints:
                lines.append(f"- {wp}")

        # Verified ROM Knowledge Notes
        if self.knowledge_notes:
            lines.append("\n[Learned ROM Notes]:")
            for kn in self.knowledge_notes[-3:]:
                lines.append(f"- {kn}")

        return "\n".join(lines)

    def save(self):
        """Persists the current memory state to disk."""
        data = {
            "quests": self.quests,
            "priority_quest": self.priority_quest,
            "active_quest": self.priority_quest,
            "current_area": self.current_area,
            "route_trail": self.route_trail,
            "micro_step": self.micro_step,
            "team": self.team,
            "supplies": self.supplies,
            "badges": self.badges,
            "knowledge_notes": self.knowledge_notes,
            "waypoints": self.waypoints,
            "short_term_memory": self.short_term_memory,
            "medium_term_memory": self.medium_term_memory,
            "long_term_memory": self.long_term_memory,
            "last_updated": datetime.now().isoformat()
        }
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryManager] Error saving memory: {e}")

    def load(self):
        """Loads memory state from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.quests = data.get("quests", [])
                    self.priority_quest = data.get("priority_quest", None)

                    # Backward compatibility fallback
                    legacy_quest = data.get("active_quest", None)
                    if legacy_quest and not self.quests:
                        self.quests = [legacy_quest]
                        self.priority_quest = legacy_quest

                    self.current_area = data.get("current_area", self.current_area)
                    self.route_trail = data.get("route_trail", self.route_trail)
                    self.micro_step = data.get("micro_step", self.micro_step)
                    self.team = data.get("team", self.team)
                    self.supplies = data.get("supplies", self.supplies)
                    self.badges = data.get("badges", self.badges)
                    self.knowledge_notes = data.get("knowledge_notes", self.knowledge_notes)
                    self.waypoints = data.get("waypoints", self.waypoints)
                    self.short_term_memory = data.get("short_term_memory", self.short_term_memory)
                    self.medium_term_memory = data.get("medium_term_memory", self.medium_term_memory)
                    self.long_term_memory = data.get("long_term_memory", self.long_term_memory)
                print(f"[MemoryManager] Loaded saved memory from {self.storage_path}")
            except Exception as e:
                print(f"[MemoryManager] Error loading memory: {e}, using defaults.")

    def reset(self):
        """Resets memory to clean starting slate."""
        self.quests = ["Leave Red's bedroom, go downstairs, and talk to Mom"]
        self.priority_quest = "Leave Red's bedroom, go downstairs, and talk to Mom"
        self.current_area = "Pallet Town (Red's Bedroom)"
        self.route_trail = [
            "Red's Bedroom (2F)"
        ]
        self.micro_step = "Walk UP and RIGHT to staircase in top-right corner to go downstairs"
        self.team = []
        self.supplies = {
            "poke_balls": 0,
            "potions": 0,
            "town_map": False,
            "money": 3000
        }
        self.badges = 0
        self.knowledge_notes = [
            "Yellow Recharged: Poké Balls available immediately at Viridian Mart with no parcel required to unlock."
        ]
        self.waypoints = dict(DEFAULT_AREA_WAYPOINTS)
        self.short_term_memory = "Just started a fresh game in Red's bedroom (2F upstairs). Need to walk UP and RIGHT to the staircase in the top-right corner to go downstairs."
        self.medium_term_memory = "Pallet Town: Go downstairs, talk to Mom, head outside, meet Professor Oak, choose starter Pokémon, and defeat Rival."
        self.long_term_memory = "Grand Goal: Defeat all 8 Kanto Gym Leaders, conquer the Elite Four, and become Pokémon Champion."
        self.turn_history.clear()
        self.save()
        print("[MemoryManager] Reset memory to clean starting slate.")
