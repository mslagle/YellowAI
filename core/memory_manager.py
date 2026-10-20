"""
Memory Manager for YellowAI.
Maintains hierarchical short, medium, and long-term memory with JSON persistence,
canonical storyline milestone tracking (sticky tombstones + walkthrough hints),
semantic quest deduplication, and a rolling NPC/sign dialogue log.
"""

from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import config

# Canonical Kanto Storyline Milestones (Ordered DAG)
# Provides deterministic progression tombstones and bite-sized walkthrough hints for local VLMs.
STORYLINE_MILESTONES: List[Dict[str, Any]] = [
    {
        "id": "leave_bedroom",
        "title": "Leave Red's bedroom, go downstairs, and talk to Mom",
        "keywords": ["bedroom", "downstairs", "staircase", "2f"],
        "walkthrough_hint": "In Red's 2F bedroom, walk UP and RIGHT to the staircase in the top-right corner to go downstairs to 1F."
    },
    {
        "id": "talk_to_mom_and_exit_house",
        "title": "Exit Mom's house via the south red exit mat into Pallet Town",
        "keywords": ["exit mom", "leave mom", "leave house", "south exit mat", "exit house"],
        "walkthrough_hint": "In Red's 1F living room, optionally talk to Mom at the table, then walk DOWN onto the red exit carpet at the bottom-center and press DOWN to step outside into Pallet Town."
    },
    {
        "id": "trigger_oak_grass",
        "title": "Walk NORTH into the tall grass exit of Pallet Town to trigger Professor Oak",
        "keywords": ["trigger professor oak", "tall grass", "meet oak", "north into the tall grass", "find professor oak"],
        "walkthrough_hint": "In Pallet Town, walk NORTH through the narrow 2-tile gap into the tall grass at the top of town. Professor Oak will automatically stop you, catch Pikachu, and escort you to his lab."
    },
    {
        "id": "claim_pikachu_and_beat_rival",
        "title": "Claim Starter Pikachu from Oak's table, defeat Rival Blue, and exit Oak's Lab",
        "keywords": ["claim starter", "starter pikachu", "poke ball on the table", "choose starter", "defeat rival in oak"],
        "walkthrough_hint": "Inside Oak's Lab, stand next to the Poké Ball on the table to the right of Professor Oak and press A to claim Pikachu. Then walk DOWN toward the exit to battle Rival Blue and leave the lab."
    },
    {
        "id": "reach_viridian_mart",
        "title": "Travel NORTH through Route 1 to Viridian City Poké Mart and pick up Oak's Parcel",
        "keywords": ["viridian mart", "pick up oak", "get oak's parcel", "collect oak's parcel", "viridian city poke mart"],
        "walkthrough_hint": "Exit Pallet Town NORTH onto Route 1, walk NORTH through Route 1 into Viridian City, and enter the blue-roofed Poké Mart on the east side to receive Oak's Parcel from the clerk."
    },
    {
        "id": "deliver_oaks_parcel",
        "title": "Deliver Oak's Parcel back to Professor Oak in Pallet Town to get the Pokédex",
        "keywords": [
            "deliver oak", "return oak", "oak's parcel", "oaks parcel", "parcel to oak",
            "parcel back", "take the parcel", "return the parcel", "bring the parcel",
            "deliver the parcel", "get the pokedex"
        ],
        "walkthrough_hint": "Leave Viridian Mart, walk SOUTH through Viridian City and Route 1 (hopping south over ledges) back to Pallet Town, enter Oak's Lab (southeast building), and talk to Professor Oak to hand over the Parcel and receive the Pokédex."
    },
    {
        "id": "reach_pewter_city",
        "title": "Travel NORTH through Viridian City, Route 2, and Viridian Forest to reach Pewter City",
        "keywords": ["viridian forest", "reach pewter", "travel to pewter", "through route 2"],
        "walkthrough_hint": "With the Pokédex obtained, walk NORTH from Pallet Town -> Route 1 -> Viridian City -> Route 2 -> Viridian Forest (follow the path to the north exit gate) -> Pewter City."
    },
    {
        "id": "defeat_brock",
        "title": "Challenge Pewter Gym and defeat Leader Brock for the Boulder Badge",
        "keywords": ["pewter gym", "defeat brock", "leader brock", "boulder badge", "boulderbadge"],
        "walkthrough_hint": "In Pewter City, heal at the Pokémon Center, then enter Pewter Gym (northwest building with pillars) and defeat Brock (Rock/Ground types—use Fighting, Water, Grass, or non-Electric attacks)."
    },
    {
        "id": "reach_cerulean_city",
        "title": "Travel EAST through Route 3, Mt. Moon, and Route 4 to reach Cerulean City",
        "keywords": ["mt. moon", "mt moon", "route 3", "route 4", "reach cerulean"],
        "walkthrough_hint": "Exit Pewter City to the EAST onto Route 3, enter Mt. Moon, defeat Team Rocket at the fossil site, exit onto Route 4, and walk EAST to Cerulean City."
    },
    {
        "id": "bill_and_misty",
        "title": "Defeat Rival on Nugget Bridge, help Bill on Route 25 for the S.S. Ticket, and beat Misty at Cerulean Gym",
        "keywords": ["nugget bridge", "route 25", "s.s. ticket", "ss ticket", "cerulean gym", "defeat misty"],
        "walkthrough_hint": "In Cerulean City, go NORTH across Nugget Bridge (Route 24) to Bill's cottage on Route 25 to get the S.S. Ticket, then defeat Misty at Cerulean Gym (use Pikachu's Electric moves)."
    }
]

STOPWORDS = {
    "the", "and", "to", "in", "of", "for", "a", "an", "on", "at", "from", "into",
    "with", "then", "go", "walk", "head", "back", "your", "talk", "get", "reach"
}

BATTLE_SPAM_PATTERNS = [
    " appeared!", " Go! ", " used ", "is paralyzed", "can't move",
    "critical hit", "super effective", "not very effective", "gained ",
    "exp. points", "fainted!", "enemy ", "foe ", "unleashed ", "missed!",
    "stat fell", "stat rose", "defense fell", "attack fell", "speed fell",
    "has no energy", "what will ", "do next?", "sent out ", "got away safely"
]

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

        # Multi-quest tracking, canonical milestone tombstones & active quest management
        self.completed_milestones: List[str] = []
        self.quests: List[str] = [STORYLINE_MILESTONES[0]["title"]]
        self.priority_quest: Optional[str] = STORYLINE_MILESTONES[0]["title"]
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

        # Structured Party, Supplies, Badges, ROM Knowledge Notes, and Rolling Dialogue History
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
        self.dialogue_history: List[Dict[str, Any]] = []

        self.load()

    # ------------------------------------------------------------------
    # Canonical Milestone & Semantic Deduplication Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize_quest(text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 1}

    @classmethod
    def _match_canonical_milestone(cls, text: str) -> Optional[Dict[str, Any]]:
        """Maps free-text quest string to a canonical STORYLINE_MILESTONES entry if it matches."""
        if not text:
            return None
        t_low = text.strip().lower()
        for m in STORYLINE_MILESTONES:
            if t_low == m["title"].lower() or t_low == m["id"].lower():
                return m
            for kw in m["keywords"]:
                if kw in t_low:
                    return m
        return None

    @classmethod
    def _is_similar_quest(cls, q1: str, q2: str) -> bool:
        """Returns True if two quest strings represent the same semantic objective."""
        if not q1 or not q2:
            return False
        s1 = q1.strip().lower()
        s2 = q2.strip().lower()
        if s1 == s2 or s1 in s2 or s2 in s1:
            return True

        m1 = cls._match_canonical_milestone(s1)
        m2 = cls._match_canonical_milestone(s2)
        if m1 and m2 and m1["id"] == m2["id"]:
            return True

        t1 = cls._tokenize_quest(s1)
        t2 = cls._tokenize_quest(s2)
        if t1 and t2:
            overlap = len(t1 & t2) / float(min(len(t1), len(t2)))
            if overlap >= 0.65:
                return True

        return SequenceMatcher(None, s1, s2).ratio() >= 0.72

    def get_active_milestone(self) -> Optional[Dict[str, Any]]:
        """Returns the current active storyline milestone (either matching priority_quest or first uncompleted)."""
        if self.priority_quest:
            matched = self._match_canonical_milestone(self.priority_quest)
            if matched and matched["id"] not in self.completed_milestones:
                return matched
        for m in STORYLINE_MILESTONES:
            if m["id"] not in self.completed_milestones:
                return m
        return None

    def mark_milestone_completed(self, milestone_id: str) -> bool:
        """
        Marks a canonical milestone AND all earlier prerequisites in STORYLINE_MILESTONES as completed.
        Purges any active quests matching those completed milestones and promotes the next milestone.
        """
        target_idx = -1
        for idx, m in enumerate(STORYLINE_MILESTONES):
            if m["id"] == milestone_id:
                target_idx = idx
                break
        if target_idx < 0:
            return False

        changed = False
        for idx in range(target_idx + 1):
            m_id = STORYLINE_MILESTONES[idx]["id"]
            if m_id not in self.completed_milestones:
                self.completed_milestones.append(m_id)
                changed = True

        # Purge all quests in self.quests that correspond to any completed milestone
        remaining_quests: List[str] = []
        for q in self.quests:
            m = self._match_canonical_milestone(q)
            if m and m["id"] in self.completed_milestones:
                changed = True
                continue
            remaining_quests.append(q)
        self.quests = remaining_quests

        # If priority_quest was completed or removed, promote next quest or next uncompleted milestone
        if not self.quests:
            next_m = self.get_active_milestone()
            if next_m:
                self.quests = [next_m["title"]]
                self.priority_quest = next_m["title"]
                changed = True
            else:
                self.priority_quest = None
        elif self.priority_quest not in self.quests:
            self.priority_quest = self.quests[0]
            changed = True

        if changed:
            self.save()
        return changed

    def _deduplicate_quest_list(self):
        """Cleans any legacy duplicate or already-completed quests from self.quests."""
        cleaned: List[str] = []
        for q in self.quests:
            if not q or not q.strip():
                continue
            m = self._match_canonical_milestone(q)
            if m and m["id"] in self.completed_milestones:
                continue
            canonical_text = m["title"] if m else q.strip()
            if not any(self._is_similar_quest(canonical_text, existing) for existing in cleaned):
                cleaned.append(canonical_text)
        self.quests = cleaned
        if not self.quests:
            next_m = self.get_active_milestone()
            if next_m:
                self.quests = [next_m["title"]]
        if self.quests and (not self.priority_quest or self.priority_quest not in self.quests):
            self.priority_quest = self.quests[0]

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
        """Appends a verified game/ROM knowledge note (separate from rolling NPC dialogue history)."""
        if not note or not note.strip():
            return
        n_clean = note.strip()
        if n_clean not in self.knowledge_notes:
            self.knowledge_notes.append(n_clean)
            if len(self.knowledge_notes) > 12:
                self.knowledge_notes = self.knowledge_notes[-12:]
            print(f"[MemoryManager] Added ROM Knowledge Note: '{n_clean}'")
            self.save()

    def record_dialogue(
        self,
        turn_id: int,
        text: str,
        area: Optional[str] = None,
        screen_mode: str = "OVERWORLD"
    ) -> bool:
        """
        Records an overworld NPC or signpost dialogue into the rolling dialogue_history log.
        - Filters out battle status spam.
        - Merges multi-page speech bubbles from the same conversation (within 3 turns).
        - Deduplicates repeated inspections of the same sign/NPC.
        - Checks for deterministic location & quest milestone triggers.
        """
        if not text or not isinstance(text, str):
            return False
        clean = " ".join(text.strip().split())
        if len(clean) < 3 or clean.upper() == "NULL":
            return False

        # Filter out battle spam
        if screen_mode == "BATTLE":
            return False
        low = clean.lower()
        if any(pat.lower() in low for pat in BATTLE_SPAM_PATTERNS):
            return False

        # Always check location & milestone triggers even if deduplicated
        self.ingest_sign_or_dialogue_location(clean)
        self._check_dialogue_milestone_triggers(clean)

        cur_area = (area or self.current_area or "Unknown").strip()
        max_history = getattr(config, "MAX_DIALOGUE_HISTORY", 20)

        if self.dialogue_history:
            last = self.dialogue_history[-1]
            last_text = last.get("text", "")
            last_turn = int(last.get("turn", -999))

            # Check if identical or substring of recent entry
            sim = SequenceMatcher(None, low, last_text.lower()).ratio()
            if sim >= 0.82 or low in last_text.lower():
                last["turn"] = turn_id
                self.save()
                return False

            # Multi-page speech bubble continuation from the same conversation (within 3 turns in same area)
            if 0 <= (turn_id - last_turn) <= 3 and last.get("area") == cur_area:
                combined = f"{last_text} {clean}".strip()
                if len(combined) <= 340:
                    last["text"] = combined
                    last["turn"] = turn_id
                    self.save()
                    return True

            # Check if identical to 2nd-most-recent entry
            if len(self.dialogue_history) >= 2:
                prev2 = self.dialogue_history[-2].get("text", "").lower()
                if SequenceMatcher(None, low, prev2).ratio() >= 0.85:
                    return False

        entry = {
            "turn": turn_id,
            "area": cur_area,
            "text": clean[:340]
        }
        self.dialogue_history.append(entry)
        if len(self.dialogue_history) > max_history:
            self.dialogue_history = self.dialogue_history[-max_history:]
        self.save()
        return True

    def _check_dialogue_milestone_triggers(self, text: str):
        """Deterministically marks storyline milestones completed when key NPC dialogue appears."""
        low = text.lower()
        # Receiving Oak's Parcel at Viridian Mart
        if any(k in low for k in ["oak's parcel", "oaks parcel", "received oak", "take this parcel", "custom poké ball", "custom poke ball"]):
            if "deliver_oaks_parcel" not in self.completed_milestones:
                self.mark_milestone_completed("reach_viridian_mart")
                self.add_quest(STORYLINE_MILESTONES[5]["title"], is_priority=True)

        # Delivering Oak's Parcel & Receiving Pokédex at Oak's Lab
        if any(k in low for k in ["obtained the pokédex", "obtained the pokedex", "received the pokédex", "received the pokedex", "delivered oak", "my custom pok", "oh! that's the"]):
            self.mark_milestone_completed("deliver_oaks_parcel")

        # Defeating Brock -> Boulder Badge
        if any(k in low for k in ["boulderbadge", "boulder badge"]):
            self.mark_milestone_completed("defeat_brock")
            if self.badges < 1:
                self.badges = 1
                self.save()

    def sync_scene_milestones(self, area_type: str, screen_mode: str):
        """
        Deterministically advances storyline milestone tombstones based on verified visual scenes
        so completed quests are never re-added when knowledge_notes roll over.
        """
        if area_type == "LIVING_ROOM_1F":
            self.mark_milestone_completed("leave_bedroom")
        elif area_type in ("OUTDOOR", "OAKS_LAB", "POKECENTER", "POKEMART", "GATEHOUSE"):
            self.mark_milestone_completed("talk_to_mom_and_exit_house")

        if area_type == "OAKS_LAB" and self.team:
            # Player has already claimed Pikachu inside Oak's Lab
            self.mark_milestone_completed("trigger_oak_grass")

        if area_type == "OUTDOOR" and self.team:
            # Player is outside with Pikachu in party -> Oak grass + Lab starter/rival are complete
            self.mark_milestone_completed("claim_pikachu_and_beat_rival")

        if "pewter" in self.current_area.lower():
            self.mark_milestone_completed("reach_pewter_city")

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

        # Check for milestone items in item updates too
        if "parcel" in txt and ("delivered" in txt or "gave" in txt or "handed" in txt):
            self.mark_milestone_completed("deliver_oaks_parcel")
        elif "pokedex" in txt or "pokédex" in txt:
            self.mark_milestone_completed("deliver_oaks_parcel")
        elif "parcel" in txt:
            if "deliver_oaks_parcel" not in self.completed_milestones:
                self.mark_milestone_completed("reach_viridian_mart")

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
                if species.lower() == "pikachu":
                    self.mark_milestone_completed("trigger_oak_grass")
                self.save()

    def add_quest(self, quest: str, is_priority: bool = False):
        """
        Adds a quest to the log with semantic deduplication and sticky tombstone protection.
        Rejects quests that match an already-completed storyline milestone.
        """
        if not quest or not quest.strip() or quest.strip().upper() == "NULL":
            return
        q_raw = quest.strip()

        # Check if this quest maps to a canonical storyline milestone
        milestone = self._match_canonical_milestone(q_raw)
        if milestone:
            if milestone["id"] in self.completed_milestones:
                # Never resurrect a completed milestone!
                return
            q = milestone["title"]
        else:
            q = q_raw

        # Semantic deduplication against existing quests
        existing_match = None
        for existing in self.quests:
            if self._is_similar_quest(q, existing):
                existing_match = existing
                break

        if existing_match is None:
            self.quests.append(q)
            target_quest = q
        else:
            target_quest = existing_match

        if is_priority or self.priority_quest is None:
            self.priority_quest = target_quest
        self.save()

    def complete_quest(self, query: str) -> Optional[str]:
        """
        Marks a quest as completed, records its canonical milestone tombstone (if applicable),
        and removes ALL matching/duplicate variants from self.quests.
        Promotes the next available quest (or next uncompleted storyline milestone).
        """
        if not query or not query.strip() or query.strip().upper() == "NULL":
            return None
        q_clean = query.strip()

        if q_clean.upper() in ["ALL", "COMPLETED", "CURRENT"]:
            if self.priority_quest:
                q_clean = self.priority_quest
            else:
                return None

        # Check if query maps to a canonical milestone -> record sticky tombstone
        milestone = self._match_canonical_milestone(q_clean)
        if milestone:
            self.mark_milestone_completed(milestone["id"])

        # Remove ALL quests in self.quests that match or are semantically similar to query
        removed: List[str] = []
        remaining: List[str] = []
        for q in self.quests:
            if self._is_similar_quest(q_clean, q):
                removed.append(q)
            else:
                remaining.append(q)

        self.quests = remaining

        # If quest list is now empty, seed the next uncompleted canonical milestone
        if not self.quests:
            next_m = self.get_active_milestone()
            if next_m:
                self.quests = [next_m["title"]]

        if not self.quests:
            self.priority_quest = None
        elif self.priority_quest not in self.quests:
            self.priority_quest = self.quests[0]

        self.save()
        if removed:
            return removed[0]
        if milestone:
            return milestone["title"]
        return None

    def set_priority_quest(self, query: str):
        """Sets a quest as highest priority, deduplicating and checking tombstones first."""
        if not query or not query.strip() or query.strip().upper() == "NULL":
            return
        self.add_quest(query.strip(), is_priority=True)

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

    def ingest_sign_or_dialogue_location(self, dialogue_text: str) -> Optional[str]:
        """
        Parses dialogue text from an on-screen sign or NPC to determine location.
        Recognizes Pallet Town, Route 1, Viridian City, Route 22, Route 2, Viridian Forest, Pewter City, etc.
        Updates self.current_area and appends landmark knowledge note.
        """
        if not dialogue_text or not isinstance(dialogue_text, str) or dialogue_text.strip().upper() == "NULL":
            return None
        t_clean = dialogue_text.strip()
        t_lower = t_clean.lower()

        # Known regional landmarks and town names in Yellow Recharged
        area_patterns = [
            ("Viridian Forest", ["viridian forest"]),
            ("Pewter City", ["pewter city", "pewter gym", "brock"]),
            ("Route 22", ["route 22", "pokemon league", "pokémon league"]),
            ("Route 2", ["route 2"]),
            ("Viridian City", ["viridian city", "viridian mart", "viridian center", "eternally green"]),
            ("Route 1", ["route 1"]),
            ("Pallet Town", ["pallet town", "oak's lab", "oaks lab", "oak pokemon research", "red's house", "blue's house", "daisy's house"]),
        ]

        matched_area = None
        for area_name, patterns in area_patterns:
            if any(p in t_lower for p in patterns):
                matched_area = area_name
                break

        if matched_area:
            prev = self.current_area
            self.current_area = matched_area
            self.add_route_node(matched_area)
            if matched_area == "Pewter City":
                self.mark_milestone_completed("reach_pewter_city")
            note = f"Signpost Read: Location identified as '{matched_area}' (Sign: \"{t_clean[:60]}\")"
            self.add_knowledge_note(note)
            print(f"[MemoryManager] Signpost Location Ingested: '{matched_area}' (was: '{prev}')")
            self.save()
            return matched_area

        return None

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
        if len(self.turn_history) > 100:
            self.turn_history = self.turn_history[-100:]

    def get_recent_actions(self, n: int = 6) -> str:
        """Returns string representation of the last N actions taken."""
        if not self.turn_history:
            return "No previous actions."
        recent = self.turn_history[-n:]
        return " -> ".join([f"T{e['turn']}:{e['action']}" for e in recent])

    def get_memory_context_prompt(self, full_dialogue: bool = False) -> str:
        """
        Formats the memory tiers, quest log, completed tombstones, walkthrough hint,
        party, supplies, area waypoints, and rolling dialogue log into concise prompt text.
        """
        lines = ["[Session Memory]"]

        # Active Walkthrough Hint from Canonical Milestone DAG
        active_m = self.get_active_milestone()
        if active_m and active_m.get("walkthrough_hint"):
            lines.append(f"- Walkthrough Guide: {active_m['walkthrough_hint']}")

        # Completed Milestones (Sticky Tombstones so model never repeats finished quests)
        if self.completed_milestones:
            lines.append(f"- Completed Milestones (DO NOT REPEAT): {', '.join(self.completed_milestones[-6:])}")

        # Secondary Quests (Top priority quest is prominently displayed in turn prompt header)
        secondary_quests = [q for q in self.quests if q != self.priority_quest]
        if secondary_quests:
            lines.append(f"- Secondary Quests: {', '.join(secondary_quests)}")

        # Route Trail and Immediate Micro-Step
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

        # Rolling NPC & Sign Dialogue Log (4 on routine turns, up to 20 on stuck/rescue turns)
        if self.dialogue_history:
            recent_count = (
                getattr(config, "MAX_DIALOGUE_HISTORY", 20)
                if full_dialogue
                else getattr(config, "DIALOGUE_PROMPT_RECENT_COUNT", 4)
            )
            selected_dialogues = self.dialogue_history[-recent_count:]
            lines.append(f"\n[Recent Dialogue Log (last {len(selected_dialogues)} of {len(self.dialogue_history)})]:")
            for d in selected_dialogues:
                lines.append(f"- [T{d.get('turn', '?')} @ {d.get('area', '?')}]: \"{d.get('text', '')}\"")

        # Verified ROM Knowledge Notes
        if self.knowledge_notes:
            lines.append("\n[Learned ROM Notes]:")
            for kn in self.knowledge_notes[-3:]:
                lines.append(f"- {kn}")

        return "\n".join(lines)

    def save(self):
        """Persists the current memory state to disk."""
        data = {
            "completed_milestones": self.completed_milestones,
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
            "dialogue_history": self.dialogue_history,
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
        """Loads memory state from disk if exists and deduplicates any legacy quests."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.completed_milestones = data.get("completed_milestones", [])
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
                    self.dialogue_history = data.get("dialogue_history", [])
                    self.waypoints = data.get("waypoints", self.waypoints)
                    self.short_term_memory = data.get("short_term_memory", self.short_term_memory)
                    self.medium_term_memory = data.get("medium_term_memory", self.medium_term_memory)
                    self.long_term_memory = data.get("long_term_memory", self.long_term_memory)

                    # Clean up any legacy duplicates or completed quests
                    self._deduplicate_quest_list()
                print(f"[MemoryManager] Loaded saved memory from {self.storage_path}")
            except Exception as e:
                print(f"[MemoryManager] Error loading memory: {e}, using defaults.")

    def reset(self):
        """Resets memory to clean starting slate."""
        self.completed_milestones = []
        self.quests = [STORYLINE_MILESTONES[0]["title"]]
        self.priority_quest = STORYLINE_MILESTONES[0]["title"]
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
        self.dialogue_history = []
        self.waypoints = dict(DEFAULT_AREA_WAYPOINTS)
        self.short_term_memory = "Just started a fresh game in Red's bedroom (2F upstairs). Need to walk UP and RIGHT to the staircase in the top-right corner to go downstairs."
        self.medium_term_memory = "Pallet Town: Go downstairs, talk to Mom, head outside, meet Professor Oak, choose starter Pokémon, and defeat Rival."
        self.long_term_memory = "Grand Goal: Defeat all 8 Kanto Gym Leaders, conquer the Elite Four, and become Pokémon Champion."
        self.turn_history.clear()
        self.save()
        print("[MemoryManager] Reset memory to clean starting slate.")

