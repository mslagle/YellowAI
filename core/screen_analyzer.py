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
GRID_Y_OFFSET = 16  # 16px vertical alignment offset (50% tile shift matching GBA camera tile centering)


def pixel_to_tile(px: int, py: int) -> Tuple[int, int]:
    """Converts a screen pixel coordinate to a grid (col, row) with the 16px GBA camera offset."""
    col = max(0, min(GRID_COLS - 1, px // TILE_SIZE))
    row = max(0, min(GRID_ROWS - 1, (py - GRID_Y_OFFSET) // TILE_SIZE))
    return col, row


def tile_to_pixel_center(col: int, row: int) -> Tuple[int, int]:
    """Returns the (x, y) pixel center of a grid tile with the 16px GBA camera offset."""
    return col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + GRID_Y_OFFSET + TILE_SIZE // 2


def tile_bounds(col: int, row: int) -> Tuple[int, int, int, int]:
    """Returns the (x1, y1, x2, y2) pixel bounding box of a grid tile."""
    x1 = col * TILE_SIZE
    y1 = row * TILE_SIZE + GRID_Y_OFFSET
    return x1, y1, x1 + TILE_SIZE, y1 + TILE_SIZE


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
    dialogue_crop: Optional[Image.Image] = None
    dialogue_has_arrow: bool = False


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

        dialogue_crop = None
        dialogue_has_arrow = False
        if is_dialogue:
            # Crop the speech bubble in 480x320: y: 215..310, x: 12..468
            dialogue_crop = resized.crop((12, 215, 468, 310)).convert("RGB")
            try:
                crop_path = Path("logs/last_dialogue_crop.png")
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                dialogue_crop.save(crop_path)
            except Exception:
                pass

            # Detect red prompt triangle arrow inside the speech bubble
            arrow_mask = (arr[218:310, 12:468, 0] > 180) & (arr[218:310, 12:468, 1] < 70) & (arr[218:310, 12:468, 2] < 70)
            dialogue_has_arrow = bool(np.sum(arrow_mask) > 15)

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

            if is_dialogue:
                arrow_status = "Waiting for player input (Press A to advance)" if dialogue_has_arrow else "Text is rendering"
                notes.insert(0, f"Speech bubble / dialogue box is OPEN on screen [{arrow_status}]. Choose 'ADVANCE_DIALOGUE' to proceed.")
                if "ADVANCE_DIALOGUE" not in available_intents:
                    available_intents.insert(0, "ADVANCE_DIALOGUE")
                # Mark dialogue box rows as impassable
                for r in range(7, GRID_ROWS):
                    for c in range(GRID_COLS):
                        collision_grid[r, c] = CELL_OBSTACLE

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
            is_naming_screen=is_naming,
            dialogue_crop=dialogue_crop,
            dialogue_has_arrow=dialogue_has_arrow
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
            # Dialogue box interior is mostly pure white (> 220)
            white_pixels = np.mean((dialogue_area[:, :, 0] > 220) & (dialogue_area[:, :, 1] > 220) & (dialogue_area[:, :, 2] > 220))
            # Border is dark/grey (< 130)
            border_dark = np.mean(arr[220:230, 20:460, :] < 130) > 0.4
            return bool(white_pixels > 0.40 and border_dark)
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
        # Inspect rows 4-7 (y = 120 to 224) where bedroom floor is located
        mid_floor = arr[120:230, :]
        tatami = (mid_floor[:, :, 0] > 140) & (mid_floor[:, :, 1] > 120) & (mid_floor[:, :, 2] < 110)
        green_rug = (mid_floor[:, :, 0] < 140) & (mid_floor[:, :, 1] > 140) & (mid_floor[:, :, 2] < 150)
        tatami_score = float(np.mean(tatami) + np.mean(green_rug))

        # Check for orange staircase landing mat (Gen 3 / Yellow Recharged Bedroom)
        mat_orange = (
            (arr[:, :, 0] > 190) &
            (arr[:, :, 1] > 110) & (arr[:, :, 1] < 165) &
            (arr[:, :, 2] > 70) & (arr[:, :, 2] < 125)
        )
        has_bedroom_stairs = int(np.sum(mat_orange & (np.arange(h)[:, None] < 230))) > 50

        is_bedroom = (
            has_bedroom_stairs
            or (tatami_score > 0.10 and (total_void > 0.15 or north_void or east_void))
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

        # Prioritize central screen viewport where player camera is centered
        center_mask = cap_mask[80:230, 160:320]
        cy, cx = np.where(center_mask)
        if len(cy) >= 15:
            py = int(np.median(cy)) + 80
            px = int(np.median(cx)) + 160
        else:
            y_idx, x_idx = np.where(cap_mask)
            if len(y_idx) < 15:
                return PlayerState(detected=False)
            py = int(np.median(y_idx))
            px = int(np.median(x_idx))

        # Ground collision tracking by FEET:
        # In GBA Pokémon, Red's sprite is 16x32 (feet are at ground collision level ~22px below cap center).
        # We determine Red's tile based on where his feet are planted on the ground!
        feet_py = py + 22
        col, row = pixel_to_tile(px, feet_py)

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
            pixel_y=feet_py,
            tile_col=col,
            tile_row=row,
            facing=facing
        )

    # -------------------------------------------------------------------------
    # Void Cell & Grid Mapping
    # -------------------------------------------------------------------------
    def _populate_void_cells(self, arr: np.ndarray, grid: np.ndarray):
        """Marks cells covered by black void as CELL_VOID (impassable)."""
        h, w, _ = arr.shape
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1, y1, x2, y2 = tile_bounds(c, r)
                y1_c, y2_c = max(0, y1), min(h, y2)
                x1_c, x2_c = max(0, x1), min(w, x2)
                if y2_c > y1_c and x2_c > x1_c:
                    cell_patch = arr[y1_c:y2_c, x1_c:x2_c]
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
        # 1. Stairs Landing Mat (Orange mat at foot of stairs)
        # Signature: Warm orange/brown [231, 140, 99], tile average [185, 135, 88]
        # Strictly enforce y >= 112 px (Row 3 or lower): The mat is on the floor, NEVER on the North Wall (y < 112)!
        y_coords = np.arange(self.target_height)[:, None]
        mat_pixels = (
            (arr[:, :, 0] > 190) &
            (arr[:, :, 1] > 110) & (arr[:, :, 1] < 165) &
            (arr[:, :, 2] > 70) & (arr[:, :, 2] < 125) &
            (y_coords >= 112) & (y_coords < 230)
        )
        my, mx = np.where(mat_pixels)
        mat_col, mat_row = 10, 4
        if len(my) > 30:
            mat_col, mat_row = pixel_to_tile(int(np.median(mx)), int(np.median(my)))

        # Room offset relative to nominal view (where mat is at Col 10, Row 4)
        dx = mat_col - 10
        dy = mat_row - 4

        # Stairs Warp: The actual staircase steps are 1 tile WEST (LEFT) of the mat!
        stairs_col = max(0, mat_col - 1)
        stairs_row = mat_row
        stairs_px, stairs_py = tile_to_pixel_center(stairs_col, stairs_row)

        # 2. Add Stairs POI (Actual warp destination: step WEST/LEFT from the mat into the stairs)
        pois.append(POI(
            name="STAIRS_DOWN",
            category="WARP",
            tile_col=stairs_col,
            tile_row=stairs_row,
            pixel_x=stairs_px,
            pixel_y=stairs_py,
            description="Staircase descending to 1F Living Room (Head WEST/LEFT from orange mat into stairs)",
            interaction_tile=(stairs_col, stairs_row)
        ))

        # 2b. Add Landing Mat POI (Staging tile in front of the staircase)
        mat_px, mat_py = tile_to_pixel_center(mat_col, mat_row)
        pois.append(POI(
            name="STAIRS_MAT",
            category="OBJECT",
            tile_col=mat_col,
            tile_row=mat_row,
            pixel_x=mat_px,
            pixel_y=mat_py,
            description="Orange stairs landing mat (Approach here first, then head WEST into stairs)",
            interaction_tile=(mat_col, mat_row)
        ))

        # 3. Add Wall Sign POI (Town map / title sign on North Wall - NOT the stairs!)
        sign_col = max(0, min(GRID_COLS - 1, 11 + dx))
        sign_row = max(0, min(GRID_ROWS - 1, 2 + dy))
        sign_px, sign_py = tile_to_pixel_center(sign_col, sign_row)
        pois.append(POI(
            name="WALL_SIGN",
            category="OBJECT",
            tile_col=sign_col,
            tile_row=sign_row,
            pixel_x=sign_px,
            pixel_y=sign_py,
            description="Town Map / Notice Sign on North Wall (Wall fixture, not an exit)",
            interaction_tile=(sign_col, min(GRID_ROWS - 1, sign_row + 1))
        ))

        # 4. Add PC POI in top-left
        pc_col = max(0, min(GRID_COLS - 1, 1 + dx))
        pc_row = max(0, min(GRID_ROWS - 1, 2 + dy))
        pc_px, pc_py = tile_to_pixel_center(pc_col, pc_row)
        pois.append(POI(
            name="PC",
            category="OBJECT",
            tile_col=pc_col,
            tile_row=pc_row,
            pixel_x=pc_px,
            pixel_y=pc_py,
            description="Red's Personal Computer (Item Storage)",
            interaction_tile=(pc_col, min(GRID_ROWS - 1, pc_row + 1))
        ))

        # 4. Room Void Borders (Cols outside room are black void)
        col_left = max(0, 1 + dx)
        col_right = min(GRID_COLS - 1, 11 + dx)
        grid[0, :] = CELL_VOID
        if col_left > 0:
            grid[:, :col_left] = CELL_VOID
        if col_right + 1 < GRID_COLS:
            grid[:, col_right + 1:] = CELL_VOID

        # 5. North Wall & Upper Fixtures (Rows 1, 2, 3)
        for r in range(max(0, 1 + dy), min(GRID_ROWS, 4 + dy)):
            for c in range(col_left, col_right + 1):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_WALL

        # 6. Staircase Safety Railing & Steps
        # Vertical railing post along western side of stairs: Col (stairs_col - 1 = mat_col - 2)
        rail_col = stairs_col - 1
        for r in [stairs_row, stairs_row + 1]:
            if 0 <= rail_col < GRID_COLS and 0 <= r < GRID_ROWS:
                grid[r, rail_col] = CELL_RAILING

        # Bottom horizontal railing bar: Below the stairs steps: Col stairs_col (mat_col - 1), Row stairs_row + 1
        bar_col = stairs_col
        if 0 <= bar_col < GRID_COLS and 0 <= stairs_row + 1 < GRID_ROWS:
            grid[stairs_row + 1, bar_col] = CELL_RAILING

        # Wooden staircase steps / upper wall behind stairs
        if 0 <= bar_col < GRID_COLS and 0 <= stairs_row - 1 < GRID_ROWS:
            grid[stairs_row - 1, bar_col] = CELL_WALL
        if 0 <= mat_col < GRID_COLS and 0 <= stairs_row - 1 < GRID_ROWS:
            grid[stairs_row - 1, mat_col] = CELL_WALL

        # Dynamic check for bright white railing pixels near the staircase
        for r in range(max(0, stairs_row - 1), min(GRID_ROWS, stairs_row + 2)):
            for c in range(max(0, stairs_col - 2), min(GRID_COLS, mat_col + 1)):
                if (c, r) in [(mat_col, mat_row), (stairs_col, stairs_row)]:
                    continue
                x1, y1, x2, y2 = tile_bounds(c, r)
                cell_patch = arr[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if cell_patch.size > 0:
                    white_cnt = np.sum((cell_patch[:, :, 0] > 210) & (cell_patch[:, :, 1] > 210) & (cell_patch[:, :, 2] > 210))
                    if white_cnt > 35:
                        grid[r, c] = CELL_RAILING

        # 7. Bed and TV Furniture Obstacles
        # Bed: southwest corner (Cols 2-3, Rows 5-8 in nominal room)
        for r in range(max(0, 5 + dy), min(GRID_ROWS, 9 + dy)):
            for c in range(max(0, 2 + dx), min(GRID_COLS, 4 + dx)):
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # TV Console / Stand: south-central (Cols 6-7, Rows 5-6 in nominal room)
        for r in range(max(0, 5 + dy), min(GRID_ROWS, 7 + dy)):
            for c in range(max(0, 6 + dx), min(GRID_COLS, 8 + dx)):
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # 8. Mark cells: Landing mat is WALKABLE, stairs steps is CELL_WARP
        grid[mat_row, mat_col] = CELL_WALKABLE
        grid[stairs_row, stairs_col] = CELL_WARP

        notes.extend([
            "Red is upstairs in his 2F Bedroom.",
            f"PRIMARY EXIT: Staircase to 1F is located at Col {stairs_col}, Row {stairs_row}.",
            f"LANDING MAT: Orange mat is at Col {mat_col}, Row {mat_row}.",
            f"RAILING: Safety railing is below the stairs at Col {bar_col}, Row {stairs_row + 1} and vertical post at Col {rail_col}.",
            f"NAVIGATION RULE: First step onto the orange landing mat at Col {mat_col}, Row {mat_row}, then head WEST (LEFT) into the staircase at Col {stairs_col}, Row {stairs_row}!",
            "SAFETY WARNING: Do NOT move DOWN from Row 4 — the bottom railing is located at Row 5!",
            "VOID WALLS: North wall and void borders are impassable."
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
        # Front door exit mat at bottom center (dynamic red mat detection)
        red_mat = (arr[180:, :, 0] > 180) & (arr[180:, :, 1] < 100) & (arr[180:, :, 2] < 100)
        my, mx = np.where(red_mat)
        exit_col, exit_row = 4, 7
        if len(my) > 30:
            exit_col, exit_row = pixel_to_tile(int(np.median(mx)), int(np.median(my)) + 180)

        exit_px, exit_py = tile_to_pixel_center(exit_col, exit_row)
        pois.append(POI(
            name="FRONT_DOOR",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=exit_px,
            pixel_y=exit_py,
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
                    x1, y1, x2, y2 = tile_bounds(c, r)
                    if cell == CELL_VOID:
                        draw.rectangle([x1, y1, x2, y2], fill=(20, 20, 20, 160), outline=(80, 20, 20))
                    elif cell in (CELL_OBSTACLE, CELL_RAILING, CELL_COUNTER, CELL_WALL):
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

            # Draw dialogue bubble box if active
            if analysis.is_dialogue:
                draw.rectangle([12, 215, 468, 310], outline=(255, 255, 0), width=2)
                draw.text((20, 200), "ACTIVE SPEECH BUBBLE (ADVANCE_DIALOGUE)", fill=(255, 255, 0))

            # Header info bar
            draw.rectangle([0, 0, self.target_width, 18], fill=(0, 0, 0, 200))
            diag_str = " | [SPEECH BUBBLE ACTIVE]" if analysis.is_dialogue else ""
            draw.text((6, 2), f"Area: {analysis.area_type} | POIs: {len(analysis.pois)}{diag_str}", fill=(255, 255, 255))

            Path(self.debug_poi_path).parent.mkdir(parents=True, exist_ok=True)
            annotated.save(self.debug_poi_path)
        except Exception:
            pass

    def format_grounded_prompt(self, analysis: SceneAnalysis) -> str:
        """Formats the analysis into an unambiguous spatial grounding block for the LLM."""
        lines = []

        if analysis.is_dialogue:
            arrow_info = " [WAITING FOR INPUT: Press 'A' to advance]" if analysis.dialogue_has_arrow else " [Text actively rendering]"
            lines.append(f"[ACTIVE SPEECH BUBBLE / DIALOGUE ON SCREEN]{arrow_info}:")
            lines.append("  * An on-screen speech bubble / message window is OPEN at the bottom of the screen!")
            lines.append("  * Carefully inspect the attached focused speech bubble crop image.")
            lines.append("  * TRANSCRIBE the visible speech bubble text into your response field `dialogue_text`.")
            lines.append("  * PRIMARY INTENT: Choose `intent: 'ADVANCE_DIALOGUE'` to advance or dismiss this message.")
            if "ADVANCE_DIALOGUE" not in analysis.available_intents:
                analysis.available_intents.insert(0, "ADVANCE_DIALOGUE")

        lines.append(f"[DETECTED ENVIRONMENT]: {analysis.area_description}")

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
