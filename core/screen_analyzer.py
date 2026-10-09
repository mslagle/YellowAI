"""
Screen Analyzer and Perception Engine for YellowAI.
Uses fast NumPy vector operations (< 5ms) to:
1. Classify the environment / area (Bedroom, Living Room, Pokecenter, Pokemart, Gym, Oak's Lab, Overworld, Battle, Dialogue/Menu).
2. Segment the 480x320 frame into a 15x10 tile grid (32x32 px cells matching 16x16 GBA native tiles).
3. Identify impassable obstacles (void borders, walls, beds, bookcases, desks, railings, counters, tables, NPCs).
4. Locate the player character (Red's cap, coordinates, facing direction).
5. Detect Points of Interest (POIs) such as staircases, doorway mats, counters, NPCs, and PCs.
6. Generate and save `logs/last_poi_annotated.png` for visual verification.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Tile Grid Constants for 480x320 Capture (Scaled 2x from 240x160 GBA Native)
GRID_COLS = 15      # 480 / 32 = 15 columns
GRID_ROWS = 10      # 320 / 32 = 10 rows
TILE_SIZE = 32      # 32x32 pixels per tile

# Cell Occupancy Types
CELL_WALKABLE = 0
CELL_VOID = 1
CELL_WALL = 2
CELL_OBSTACLE = 3
CELL_RAILING = 4
CELL_COUNTER = 5
CELL_NPC = 6
CELL_WARP = 7


@dataclass
class POI:
    name: str                       # e.g. "STAIRS_DOWN", "FRONT_DOOR", "MOM", "PC"
    category: str                   # "WARP", "NPC", "OBJECT", "COUNTER"
    tile_col: int                   # 0 to 14
    tile_row: int                   # 0 to 9
    pixel_x: int
    pixel_y: int
    description: str
    interaction_tile: Tuple[int, int]  # (col, row) where player should stand to interact / warp


@dataclass
class PlayerState:
    detected: bool = False
    pixel_x: int = -1
    pixel_y: int = -1
    tile_col: int = -1
    tile_row: int = -1
    facing: str = "UNKNOWN"          # "UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"


@dataclass
class SceneAnalysis:
    area_type: str                   # e.g. "BEDROOM_2F", "LIVING_ROOM_1F", "OVERWORLD", "BATTLE", etc.
    area_description: str
    player: PlayerState
    collision_grid: np.ndarray       # Shape (10, 15) with CELL_* values
    pois: List[POI] = field(default_factory=list)
    available_intents: List[str] = field(default_factory=list)
    spatial_notes: List[str] = field(default_factory=list)
    is_battle: bool = False
    is_dialogue: bool = False
    is_naming_screen: bool = False


class ScreenAnalyzer:
    """
    Fast rule-based perception engine converting raw pixels to grounded spatial and semantic game state.
    """

    def __init__(
        self,
        target_width: int = 480,
        target_height: int = 320,
        debug_poi_path: Optional[str] = "logs/last_poi_annotated.png"
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.debug_poi_path = debug_poi_path

    def analyze(self, frame: Image.Image, current_area_hint: str = "") -> SceneAnalysis:
        """
        Analyzes a captured frame and produces a structured SceneAnalysis object.
        Also saves `logs/last_poi_annotated.png`.
        """
        resized = frame.resize((self.target_width, self.target_height))
        arr = np.array(resized)

        # 1. Check special screens first (Battle, Naming, Dialogue)
        is_naming = self._is_naming_screen(arr)
        is_battle = self._is_battle_screen(arr)
        is_dialogue = self._is_dialogue_active(arr)

        # 2. Player localization
        player = self._detect_player(arr)

        # 3. Classify area
        area_type, area_desc = self._classify_area(arr, current_area_hint, is_battle, is_naming)

        # 4. Construct base collision grid
        collision_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        self._populate_void_cells(arr, collision_grid)

        # 5. Extract POIs and area-specific obstacles
        pois: List[POI] = []
        notes: List[str] = []
        available_intents: List[str] = []

        if is_battle:
            area_type = "BATTLE"
            area_desc = "Turn-Based Pokémon Battle"
            notes.append("Active Pokémon Battle: Choose battle strategy (FIGHT, BAG, POKÉMON, RUN).")
            available_intents = ["BATTLE_ACTION"]
        elif is_naming:
            area_type = "NAMING_SCREEN"
            area_desc = "Character / Rival Naming Keyboard"
            notes.append("Naming Keyboard Screen: Entering character name.")
            available_intents = ["WAIT", "ADVANCE_DIALOGUE"]
        elif is_dialogue:
            notes.append("Dialogue or text box is active on screen. Press A to advance dialogue.")
            available_intents = ["ADVANCE_DIALOGUE", "EXIT_ROOM"]
        else:
            # Map-specific collision & POI extraction
            if area_type == "BEDROOM_2F":
                self._analyze_bedroom(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "LIVING_ROOM_1F":
                self._analyze_living_room(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "OAKS_LAB":
                self._analyze_oaks_lab(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "POKECENTER":
                self._analyze_pokecenter(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "POKEMART":
                self._analyze_pokemart(arr, collision_grid, player, pois, notes, available_intents)
            else:
                self._analyze_overworld(arr, collision_grid, player, pois, notes, available_intents)

        analysis = SceneAnalysis(
            area_type=area_type,
            area_description=area_desc,
            player=player,
            collision_grid=collision_grid,
            pois=pois,
            available_intents=available_intents,
            spatial_notes=notes,
            is_battle=is_battle,
            is_dialogue=is_dialogue,
            is_naming_screen=is_naming
        )

        if self.debug_poi_path:
            self._save_annotated_image(resized, analysis)

        return analysis

    # -------------------------------------------------------------------------
    # Screen Mode & Area Classification
    # -------------------------------------------------------------------------
    def _is_naming_screen(self, arr: np.ndarray) -> bool:
        """Checks for the GBA naming keyboard screen."""
        try:
            top_blue = np.mean((arr[5:20, 10:200, 0] < 50) & (arr[5:20, 10:200, 1] > 70) & (arr[5:20, 10:200, 2] > 180))
            white_card = np.mean((arr[60:120, 120:380, 0] > 230) & (arr[60:120, 120:380, 1] > 230) & (arr[60:120, 120:380, 2] > 230))
            ok_yellow = np.mean((arr[250:290, 390:460, 0] > 180) & (arr[250:290, 390:460, 1] > 170) & (arr[250:290, 390:460, 2] < 140))
            return bool(top_blue > 0.30 and white_card > 0.40 and ok_yellow > 0.04)
        except Exception:
            return False

    def _is_battle_screen(self, arr: np.ndarray) -> bool:
        """Detects battle HUD (HP bars, battle text box, command menu)."""
        try:
            # Battle enemy HP bar at top left (green/dark bar with black border)
            # Gen 3 battle background is split into upper battle field and bottom dark banner
            # Bottom banner in battle: dark blue / black border
            bottom_box = arr[220:, :, :]
            is_dark_banner = np.mean(bottom_box[:, :, 0] < 60) > 0.6 and np.mean(bottom_box[:, :, 2] > 60) > 0.3
            # Check for green HP bar pixels (R: 110-130, G: 240-255, B: 110-130)
            hp_green = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 150) & (arr[:, :, 2] < 150)
            has_hp = np.sum(hp_green) > 50
            return bool(is_dark_banner and has_hp)
        except Exception:
            return False

    def _is_dialogue_active(self, arr: np.ndarray) -> bool:
        """Checks if a white bottom dialogue box with dark text is on screen."""
        try:
            dialogue_area = arr[230:300, 20:460, :]
            # Dialogue box interior is mostly pure white (> 240)
            white_pixels = np.mean((dialogue_area[:, :, 0] > 230) & (dialogue_area[:, :, 1] > 230) & (dialogue_area[:, :, 2] > 230))
            # Border is dark/black
            border_dark = np.mean(arr[220:225, 20:460, :] < 50) > 0.5
            return bool(white_pixels > 0.45 and border_dark)
        except Exception:
            return False

    def _classify_area(
        self,
        arr: np.ndarray,
        area_hint: str,
        is_battle: bool,
        is_naming: bool
    ) -> Tuple[str, str]:
        """Classifies the environment based on structural colors, flooring, and void layout."""
        if is_battle:
            return "BATTLE", "In a Pokémon Battle"
        if is_naming:
            return "NAMING_SCREEN", "Character Naming Screen"

        upper = arr[:160, :]
        h, w, _ = arr.shape

        # Void borders (< 25 RGB)
        void_mask = (arr[:, :, 0] < 25) & (arr[:, :, 1] < 25) & (arr[:, :, 2] < 25)
        total_void = float(np.mean(void_mask))
        south_void = float(np.mean(void_mask[h - 40:, :])) > 0.25
        north_void = float(np.mean(void_mask[:40, :])) > 0.40
        west_void = float(np.mean(void_mask[:, :40])) > 0.40
        east_void = float(np.mean(void_mask[:, w - 40:])) > 0.40

        # Flooring signatures
        # 1. Tatami floor (Tan / Yellowish-brown) & Green rug (Red's 2F Bedroom)
        tatami = (upper[:, :, 0] > 140) & (upper[:, :, 1] > 120) & (upper[:, :, 2] < 110)
        green_rug = (upper[:, :, 0] < 130) & (upper[:, :, 1] > 130) & (upper[:, :, 2] > 70)
        tatami_score = float(np.mean(tatami) + np.mean(green_rug))

        is_bedroom = (
            (tatami_score > 0.05 and (total_void > 0.25 or north_void or east_void))
            or (total_void > 0.35 and tatami_score > 0.03)
            or ("bedroom" in area_hint.lower())
        )
        if is_bedroom:
            return "BEDROOM_2F", "Pallet Town - Red's Bedroom (2F)"

        # 2. Oak's Research Lab: Yellow walls, lab floor, void on west & east wings
        is_lab = (
            (west_void and east_void and total_void > 0.08)
            or "lab" in area_hint.lower()
        )
        if is_lab:
            return "OAKS_LAB", "Pallet Town - Professor Oak's Research Lab"

        # 3. Outdoor Overworld Check: Dominant green foliage channel (G > R + 12)
        mean_rgb = arr.mean(axis=(0, 1))
        is_outdoor_foliage = bool(mean_rgb[1] > mean_rgb[0] + 12)
        if is_outdoor_foliage and total_void < 0.10:
            return "OVERWORLD", "Pallet Town / Kanto Overworld"

        # 4. 1F Living Room: Wood floor, dining table with Mom, red door mat
        red_mat = (arr[200:, :, 0] > 180) & (arr[200:, :, 1] < 100) & (arr[200:, :, 2] < 100)
        has_red_mat = np.sum(red_mat) > 150
        is_living_room = (
            (has_red_mat and total_void < 0.20)
            or ("living" in area_hint.lower() or "1f" in area_hint.lower() or "mom" in area_hint.lower())
        )
        if is_living_room:
            return "LIVING_ROOM_1F", "Pallet Town - Mom's Living Room (1F)"

        # 4. Poké Center: White/Pink floor, central healing counter with red Poké Ball emblem
        center_floor = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 210) & (arr[:, :, 2] > 210)
        red_counter_emblem = (arr[60:140, :, 0] > 190) & (arr[60:140, :, 1] < 80) & (arr[60:140, :, 2] < 80)
        if (np.mean(center_floor) > 0.25 and np.sum(red_counter_emblem) > 150) or "center" in area_hint.lower():
            return "POKECENTER", "Pokémon Center - Free Healing Station"

        # 5. Poké Mart: Blue roof/counter, shelves
        blue_counter = (arr[80:160, :200, 0] < 100) & (arr[80:160, :200, 1] > 120) & (arr[80:160, :200, 2] > 180)
        if np.sum(blue_counter) > 250 or "mart" in area_hint.lower():
            return "POKEMART", "Poké Mart - Item Shop"

        # 6. Gym
        if "gym" in area_hint.lower():
            return "GYM", "Pokémon Gym - Badge Challenge"

        # Fallback to Overworld
        return "OVERWORLD", "Kanto Overworld (Town / Route)"

    # -------------------------------------------------------------------------
    # Player Localization & Orientation
    # -------------------------------------------------------------------------
    def _detect_player(self, arr: np.ndarray) -> PlayerState:
        """Finds Red's sprite via red cap color and determines facing direction."""
        cap_mask = (arr[:, :, 0] > 175) & (arr[:, :, 1] < 105) & (arr[:, :, 2] < 105)
        y_idx, x_idx = np.where(cap_mask)

        if len(y_idx) < 15:
            return PlayerState(detected=False)

        py = int(np.median(y_idx))
        px = int(np.median(x_idx))

        col = max(0, min(GRID_COLS - 1, px // TILE_SIZE))
        row = max(0, min(GRID_ROWS - 1, py // TILE_SIZE))

        # Determine facing direction by inspecting sprite patch below cap
        h, w, _ = arr.shape
        y1, y2 = max(0, py - 6), min(h, py + 22)
        x1, x2 = max(0, px - 12), min(w, px + 12)
        patch = arr[y1:y2, x1:x2]

        # Skin tone pixels
        skin = (
            (patch[:, :, 0] > 210) &
            (patch[:, :, 1] > 140) & (patch[:, :, 1] < 220) &
            (patch[:, :, 2] > 90) & (patch[:, :, 2] < 170)
        )
        skin_count = int(np.sum(skin))

        # Dark hair / back of cap pixels
        dark_hair = (patch[:, :, 0] < 90) & (patch[:, :, 1] < 80) & (patch[:, :, 2] < 80)
        hair_count = int(np.sum(dark_hair))

        if skin_count >= 15:
            facing = "DOWN"
        elif hair_count >= 20 and skin_count < 8:
            facing = "UP"
        else:
            facing = "LEFT" if px < 240 else "RIGHT"

        return PlayerState(
            detected=True,
            pixel_x=px,
            pixel_y=py,
            tile_col=col,
            tile_row=row,
            facing=facing
        )

    # -------------------------------------------------------------------------
    # Void Cell & Grid Mapping
    # -------------------------------------------------------------------------
    def _populate_void_cells(self, arr: np.ndarray, grid: np.ndarray):
        """Marks cells covered by black void as CELL_VOID (impassable)."""
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell_patch = arr[r * TILE_SIZE:(r + 1) * TILE_SIZE, c * TILE_SIZE:(c + 1) * TILE_SIZE]
                black_ratio = np.mean(cell_patch < 25)
                if black_ratio > 0.70:
                    grid[r, c] = CELL_VOID

    # -------------------------------------------------------------------------
    # Map-Specific Analyzers
    # -------------------------------------------------------------------------
    def _analyze_bedroom(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes Red's 2F Bedroom fixtures: stairs, railing, bed, TV, PC, void borders."""
        # 1. Stairs Landing Mat (Orange mat in top-right)
        mat_pixels = (
            (arr[:, :, 0] > 190) &
            (arr[:, :, 1] > 70) & (arr[:, :, 1] < 160) &
            (arr[:, :, 2] > 50) & (arr[:, :, 2] < 130)
        )
        my, mx = np.where(mat_pixels & (np.arange(self.target_height)[:, None] < 120))
        stairs_col, stairs_row = 12, 2
        if len(my) > 60:
            stairs_col = max(0, min(GRID_COLS - 1, int(np.median(mx)) // TILE_SIZE))
            stairs_row = max(0, min(GRID_ROWS - 1, int(np.median(my)) // TILE_SIZE))

        # Add Stairs POI
        pois.append(POI(
            name="STAIRS_DOWN",
            category="WARP",
            tile_col=stairs_col,
            tile_row=stairs_row,
            pixel_x=stairs_col * TILE_SIZE + 16,
            pixel_y=stairs_row * TILE_SIZE + 16,
            description="Staircase descending to 1F Living Room",
            interaction_tile=(stairs_col, stairs_row)
        ))

        # 2. Add PC POI in top-left
        pois.append(POI(
            name="PC",
            category="OBJECT",
            tile_col=3,
            tile_row=1,
            pixel_x=3 * TILE_SIZE + 16,
            pixel_y=1 * TILE_SIZE + 16,
            description="Red's Personal Computer (Item Storage)",
            interaction_tile=(3, 2)
        ))

        # 3. Mark static bedroom obstacles:
        # Solid North Wall along Row 0
        for c in range(GRID_COLS):
            grid[0, c] = CELL_WALL

        # PC at (Col 3, Row 1)
        grid[1, 3] = CELL_OBSTACLE

        # Dressers and bookcases along North Wall at Cols 5-7, Row 1
        for c in [5, 6, 7]:
            grid[1, c] = CELL_OBSTACLE

        # Bed fixture at Cols 3-4, Rows 4-6
        for r in [4, 5, 6]:
            for c in [3, 4]:
                grid[r, c] = CELL_OBSTACLE

        # TV and stand at Col 8, Rows 3-4
        for r in [3, 4]:
            grid[r, 8] = CELL_OBSTACLE

        # 4. Staircase Safety Railing (Cols 10-11, Rows 1-3)
        # Dynamic check for bright white railing pixels (> 40 pixels > 235 RGB)
        for r in range(1, 4):
            for c in range(9, 13):
                cell_patch = arr[r * TILE_SIZE:(r + 1) * TILE_SIZE, c * TILE_SIZE:(c + 1) * TILE_SIZE]
                white_cnt = np.sum((cell_patch[:, :, 0] > 235) & (cell_patch[:, :, 1] > 235) & (cell_patch[:, :, 2] > 235))
                if white_cnt > 40 and not (c == stairs_col and r == stairs_row):
                    grid[r, c] = CELL_RAILING

        # Guaranteed structural railing coverage:
        # Col 10: Vertical safety post (Rows 1, 2, 3)
        for r in [1, 2, 3]:
            grid[r, 10] = CELL_RAILING
        # Col 11: Bottom horizontal railing (Rows 2, 3)
        for r in [2, 3]:
            grid[r, 11] = CELL_RAILING

        # Mark orange stairs landing tile as CELL_WARP
        grid[stairs_row, stairs_col] = CELL_WARP

        notes.extend([
            "Red is upstairs in his 2F Bedroom.",
            f"PRIMARY EXIT: Staircase to 1F is located at Col {stairs_col}, Row {stairs_row} (Top-Right).",
            "OBSTACLES: Bed is at Cols 3-4 Rows 4-6. TV is at Col 8 Rows 3-4. White staircase railing is at Cols 10-11 Rows 1-3.",
            "NAVIGATION RULE: Walk south/east of the TV across the open floor at Row 4 to Col 12, then walk UP into the orange stairs landing mat at Col 12, Row 2!",
            "VOID WALLS: West and South borders are impassable black void. DO NOT move into void!"
        ])

        available_intents.extend(["EXIT_ROOM", "GO_TO_POI", "WAIT"])

    def _analyze_living_room(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes 1F Living Room: front door exit, dining table with Mom, stairs up."""
        # Front door exit mat at bottom center
        exit_col, exit_row = 4, 7
        pois.append(POI(
            name="FRONT_DOOR",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=exit_col * TILE_SIZE + 16,
            pixel_y=exit_row * TILE_SIZE + 16,
            description="Front Door exit mat leading outside to Pallet Town",
            interaction_tile=(exit_col, exit_row)
        ))

        # Mom at dining table in center
        mom_col, mom_row = 6, 4
        pois.append(POI(
            name="MOM",
            category="NPC",
            tile_col=mom_col,
            tile_row=mom_row,
            pixel_x=mom_col * TILE_SIZE + 16,
            pixel_y=mom_row * TILE_SIZE + 16,
            description="Red's Mom (Restores party HP)",
            interaction_tile=(mom_col, mom_row + 1)
        ))

        # Dining table obstacle
        for r in [4, 5]:
            for c in [5, 6, 7]:
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        notes.extend([
            "Red is downstairs on 1F (Mom's Living Room).",
            "PRIMARY EXIT: Front door to Pallet Town is at the bottom-center (Red welcome mat).",
            "NPC: Mom is seated at the dining table in the center."
        ])

        available_intents.extend(["EXIT_ROOM", "TALK_TO_NPC", "GO_TO_POI", "WAIT"])

    def _analyze_oaks_lab(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes Professor Oak's Research Lab."""
        # South exit door
        exit_col, exit_row = 5, 8
        pois.append(POI(
            name="LAB_EXIT",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=exit_col * TILE_SIZE + 16,
            pixel_y=exit_row * TILE_SIZE + 16,
            description="Exit door back to Pallet Town",
            interaction_tile=(exit_col, exit_row)
        ))

        # Professor Oak
        pois.append(POI(
            name="PROFESSOR_OAK",
            category="NPC",
            tile_col=6,
            tile_row=2,
            pixel_x=6 * TILE_SIZE + 16,
            pixel_y=2 * TILE_SIZE + 16,
            description="Professor Oak",
            interaction_tile=(6, 3)
        ))

        # Starter Poké Ball Table on the right
        pois.append(POI(
            name="STARTER_TABLE",
            category="OBJECT",
            tile_col=8,
            tile_row=3,
            pixel_x=8 * TILE_SIZE + 16,
            pixel_y=3 * TILE_SIZE + 16,
            description="Starter Poké Ball table (Approach from bottom at Col 8, Row 4)",
            interaction_tile=(8, 4)
        ))

        notes.extend([
            "Red is inside Professor Oak's Research Lab.",
            "OBJECTIVE: Approach starter table from BELOW (Col 8, Row 4) and press A to choose Pikachu!",
            "EXIT: South doorway leads out to Pallet Town."
        ])

        available_intents.extend(["TALK_TO_NPC", "GO_TO_POI", "EXIT_ROOM", "WAIT"])

    def _analyze_pokecenter(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes Pokémon Center interior."""
        # Nurse Joy at counter top-center
        pois.append(POI(
            name="NURSE_JOY",
            category="NPC",
            tile_col=7,
            tile_row=2,
            pixel_x=7 * TILE_SIZE + 16,
            pixel_y=2 * TILE_SIZE + 16,
            description="Nurse Joy (Free party healing)",
            interaction_tile=(7, 3)
        ))

        # PC terminal on right wall
        pois.append(POI(
            name="PC",
            category="OBJECT",
            tile_col=12,
            tile_row=2,
            pixel_x=12 * TILE_SIZE + 16,
            pixel_y=2 * TILE_SIZE + 16,
            description="Pokémon Storage System PC",
            interaction_tile=(12, 3)
        ))

        # Exit door at south center
        pois.append(POI(
            name="CENTER_EXIT",
            category="WARP",
            tile_col=7,
            tile_row=8,
            pixel_x=7 * TILE_SIZE + 16,
            pixel_y=8 * TILE_SIZE + 16,
            description="Exit door back to town",
            interaction_tile=(7, 8)
        ))

        # Counter obstacle across rows 2-3
        for c in range(4, 11):
            grid[2, c] = CELL_COUNTER

        notes.extend([
            "Red is inside the Pokémon Center.",
            "HEALING: Walk to Nurse Joy counter at Col 7, Row 3 to heal your party.",
            "EXIT: South door at Col 7, Row 8."
        ])

        available_intents.extend(["TALK_TO_NPC", "EXIT_ROOM", "GO_TO_POI", "WAIT"])

    def _analyze_pokemart(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes Poké Mart interior."""
        pois.append(POI(
            name="MART_CLERK",
            category="NPC",
            tile_col=3,
            tile_row=3,
            pixel_x=3 * TILE_SIZE + 16,
            pixel_y=3 * TILE_SIZE + 16,
            description="Cashier Clerk (Buy / Sell items, deliver Oak's Parcel)",
            interaction_tile=(4, 3)
        ))

        pois.append(POI(
            name="MART_EXIT",
            category="WARP",
            tile_col=4,
            tile_row=8,
            pixel_x=4 * TILE_SIZE + 16,
            pixel_y=8 * TILE_SIZE + 16,
            description="Exit door back outside",
            interaction_tile=(4, 8)
        ))

        notes.extend([
            "Red is inside the Poké Mart.",
            "SHOPPING: Talk to cashier clerk at left counter to buy Poké Balls or receive Parcel.",
            "EXIT: South door at Col 4, Row 8."
        ])

        available_intents.extend(["TALK_TO_NPC", "EXIT_ROOM", "GO_TO_POI", "WAIT"])

    def _analyze_overworld(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes outdoor Overworld (Pallet Town, Route 1, Viridian City)."""
        notes.extend([
            "Red is outdoors in the Overworld.",
            "NAVIGATION: Doors have entrance threshold mats. Walk directly onto doorway tiles to enter.",
            "Ledges can only be hopped going SOUTH. You cannot jump UP over brown ledges."
        ])

        available_intents.extend([
            "EXPLORE_NORTH",
            "EXPLORE_SOUTH",
            "EXPLORE_EAST",
            "EXPLORE_WEST",
            "ENTER_BUILDING",
            "TALK_TO_NPC",
            "WAIT"
        ])

    # -------------------------------------------------------------------------
    # Visual Annotation Generator (Dual Debug #1)
    # -------------------------------------------------------------------------
    def _save_annotated_image(self, img: Image.Image, analysis: SceneAnalysis):
        """
        Saves `logs/last_poi_annotated.png` drawing:
        - Red's player bounding box and facing direction
        - POI bounding boxes and labels (Warps, NPCs, Objects)
        - Highlighted obstacle regions (beds, railings, void)
        """
        try:
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            # Draw void overlay subtly
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    cell = analysis.collision_grid[r, c]
                    x1 = c * TILE_SIZE
                    y1 = r * TILE_SIZE
                    x2 = (c + 1) * TILE_SIZE
                    y2 = (r + 1) * TILE_SIZE
                    if cell == CELL_VOID:
                        draw.rectangle([x1, y1, x2, y2], fill=(20, 20, 20, 160), outline=(80, 20, 20))
                    elif cell in (CELL_OBSTACLE, CELL_RAILING, CELL_COUNTER):
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=1)

            # Draw detected player
            p = analysis.player
            if p.detected:
                px, py = p.pixel_x, p.pixel_y
                draw.rectangle([px - 14, py - 14, px + 14, py + 14], outline=(0, 255, 0), width=2)
                draw.text((max(0, px - 28), max(0, py - 26)), f"Red ({p.facing})", fill=(0, 255, 0))

            # Draw detected POIs
            for poi in analysis.pois:
                x = poi.pixel_x
                y = poi.pixel_y
                color = (255, 165, 0) if poi.category == "WARP" else (0, 200, 255)
                draw.rectangle([x - 12, y - 12, x + 12, y + 12], outline=color, width=2)
                draw.text((max(0, x - 24), max(0, y - 22)), poi.name, fill=color)

            # Header info bar
            draw.rectangle([0, 0, self.target_width, 18], fill=(0, 0, 0, 200))
            draw.text((6, 2), f"Area: {analysis.area_type} | POIs: {len(analysis.pois)}", fill=(255, 255, 255))

            Path(self.debug_poi_path).parent.mkdir(parents=True, exist_ok=True)
            annotated.save(self.debug_poi_path)
        except Exception:
            pass

    def format_grounded_prompt(self, analysis: SceneAnalysis) -> str:
        """Formats the analysis into an unambiguous spatial grounding block for the LLM."""
        lines = [
            f"[DETECTED ENVIRONMENT]: {analysis.area_description}",
        ]

        if analysis.player.detected:
            p = analysis.player
            lines.append(f"[PLAYER STATUS]: Red is at Tile (Col {p.tile_col}, Row {p.tile_row}) facing {p.facing}.")
        else:
            lines.append("[PLAYER STATUS]: Character sprite not detected.")

        if analysis.pois:
            lines.append("[AVAILABLE POINTS OF INTEREST & DESTINATIONS]:")
            for poi in analysis.pois:
                lines.append(f"  - {poi.name} ({poi.category}): {poi.description} at Tile (Col {poi.tile_col}, Row {poi.tile_row})")

        if analysis.spatial_notes:
            lines.append("[SPATIAL GUIDANCE & PHYSICAL BOUNDARIES]:")
            for note in analysis.spatial_notes:
                lines.append(f"  * {note}")

        if analysis.available_intents:
            intents_str = ", ".join(f"'{i}'" for i in analysis.available_intents)
            lines.append(f"[VALID HIGH-LEVEL INTENTS]: [{intents_str}]")

        return "\n".join(lines)
