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

        # 1. Check special screens first (Battle, Naming, Title/Main Menu, Trainer Card, Shop/Start Menu, Dialogue)
        is_naming = self._is_naming_screen(arr)
        is_battle = self._is_battle_screen(arr) and not is_naming
        is_trainer_card = self._is_trainer_card(arr) and not is_battle and not is_naming
        is_title_or_menu, title_desc, is_oak_intro = self._is_title_or_main_menu(arr)
        if is_battle or is_naming or is_trainer_card:
            is_title_or_menu = False
            is_oak_intro = False

        is_shop_menu = self._is_shop_menu(arr) and not is_battle and not is_naming and not is_title_or_menu
        is_start_menu = self._is_start_menu(arr) and not is_battle and not is_naming and not is_title_or_menu and not is_shop_menu
        is_dialogue = (
            (self._is_dialogue_active(arr) or is_oak_intro)
            and not is_battle
            and not is_naming
            and not is_trainer_card
            and not is_shop_menu
            and not (is_title_or_menu and not is_oak_intro)
        )

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

        # 2. Classify area
        area_type, area_desc = self._classify_area(
            arr,
            current_area_hint,
            is_battle=is_battle,
            is_naming=is_naming,
            is_trainer_card=is_trainer_card,
            is_title_or_menu=is_title_or_menu,
            title_desc=title_desc,
            is_shop_menu=is_shop_menu,
            is_start_menu=is_start_menu,
            is_dialogue=is_dialogue
        )

        # 3. Player localization (skip on non-map screens like Battle, Naming, Title, Trainer Card)
        if area_type in ("BATTLE", "NAMING_SCREEN", "TITLE_SCREEN", "TRAINER_CARD"):
            player = PlayerState(detected=False)
        else:
            player = self._detect_player(arr)
            # In GBA Overworld & standard indoor rooms, camera is locked to Red at center (240, 160 -> Tile 7, 4)
            if not player.detected and area_type in ("OVERWORLD", "BEDROOM_2F", "LIVING_ROOM_1F", "OAKS_LAB", "POKEMART", "POKECENTER"):
                player = PlayerState(
                    detected=True,
                    pixel_x=240,
                    pixel_y=166,
                    tile_col=7,
                    tile_row=4,
                    facing="DOWN"
                )

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
        elif is_trainer_card:
            area_type = "TRAINER_CARD"
            area_desc = "Player Trainer Card Screen"
            notes.append("Trainer Card is open. Press B or A ('ADVANCE_DIALOGUE') to close and return.")
            available_intents = ["ADVANCE_DIALOGUE", "EXIT_ROOM", "WAIT"]
        elif is_title_or_menu:
            area_type = "TITLE_SCREEN"
            notes.append(f"{area_desc}: Choose 'ADVANCE_DIALOGUE' (Press A) to proceed.")
            available_intents = ["ADVANCE_DIALOGUE", "WAIT"]
        else:
            # Map-specific collision & POI extraction
            if area_type == "BEDROOM_2F":
                self._analyze_bedroom(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "LIVING_ROOM_1F":
                self._analyze_living_room(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue)
            elif area_type == "OAKS_LAB":
                self._analyze_oaks_lab(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "POKECENTER":
                self._analyze_pokecenter(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "POKEMART":
                self._analyze_pokemart(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue, is_shop_menu=is_shop_menu)
            else:
                self._analyze_overworld(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue)

            if is_start_menu:
                notes.insert(0, "START MENU IS OPEN on the right side of the screen. Press B or choose 'ADVANCE_DIALOGUE' to close or interact with menu.")
                if "ADVANCE_DIALOGUE" not in available_intents:
                    available_intents.insert(0, "ADVANCE_DIALOGUE")

            if is_shop_menu:
                notes.insert(0, "POKÉ MART SHOP MENU IS OPEN. Use 'ADVANCE_DIALOGUE' to buy items or press B to exit the shop menu.")
                if "ADVANCE_DIALOGUE" not in available_intents:
                    available_intents.insert(0, "ADVANCE_DIALOGUE")

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
        """Detects Pokémon Battle via Enemy & Player cream HUD cards or battle banner + HP bar."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            # Recharged Yellow / Gen 3 Battle HUD cream card fill: RGB ~ [248, 248, 216]
            cream_mask = (r > 230) & (g > 230) & (b >= 190) & (b <= 232) & ((r - b) >= 12) & ((g - b) >= 12)
            ehud_cream = float(np.mean(cream_mask[28:82, 24:212]))
            phud_cream = float(np.mean(cream_mask[148:212, 268:455]))
            if ehud_cream > 0.25 and phud_cream > 0.25:
                return True

            # Fallback for dark battle banner + green HP bar
            bottom_box = arr[220:, :, :]
            is_dark_banner = np.mean(bottom_box[:, :, 0] < 60) > 0.6 and np.mean(bottom_box[:, :, 2] > 60) > 0.3
            hp_green = (g > 200) & (r < 150) & (b < 150)
            has_hp = np.sum(hp_green) > 50
            return bool(is_dark_banner and has_hp)
        except Exception:
            return False

    def _is_trainer_card(self, arr: np.ndarray) -> bool:
        """Detects the full-screen Trainer Card UI."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            top_teal = float(np.mean((r[:14, :300] < 95) & (g[:14, :300] > 135) & (b[:14, :300] > 130) & (np.abs(g[:14, :300] - b[:14, :300]) < 25)))
            bot_teal = float(np.mean((r[306:, :300] < 95) & (g[306:, :300] > 135) & (b[306:, :300] > 130) & (np.abs(g[306:, :300] - b[306:, :300]) < 25)))
            return bool(top_teal > 0.50 and bot_teal > 0.50)
        except Exception:
            return False

    def _is_title_or_main_menu(self, arr: np.ndarray) -> Tuple[bool, str, bool]:
        """Detects Title Screen, Jaizu Splash, Main Menu (Continue/New Game), and Oak Intro."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)

            # 1. Title Screen: White left margin + yellow/blue Pokémon logo at top + gold/yellow bottom bar
            left_white = float(np.mean((r[40:260, :20] > 230) & (g[40:260, :20] > 230) & (b[40:260, :20] > 230)))
            logo_yellow = float(np.mean((r[20:115, 130:450] > 230) & (g[20:115, 130:450] > 195) & (b[20:115, 130:450] < 70)))
            logo_blue = float(np.mean((r[20:115, 130:450] < 55) & (g[20:115, 130:450] < 80) & (b[20:115, 130:450] > 175)))
            if left_white > 0.70 and logo_yellow > 0.12 and logo_blue > 0.05:
                return True, "Pokémon Recharged Yellow Title Screen", False

            # 2. Jaizu Intro Splash: Uniform warm khaki/beige screen
            khaki_bg = float(np.mean((r[30:290, 20:340] > 210) & (g[30:290, 20:340] > 195) & (b[30:290, 20:340] > 130) & (b[30:290, 20:340] < 175)))
            if khaki_bg > 0.75:
                return True, "Intro Credits / Splash Screen", False

            # 3. Main Menu (Continue / New Game): Periwinkle blue left border + white option cards
            left_periwinkle = float(np.mean((b[50:250, :20] > 200) & (b[50:250, :20] > r[50:250, :20] + 65) & (r[50:250, :20] > 100)))
            menu_white = float(np.mean((r[20:160, 45:435] > 225) & (g[20:160, 45:435] > 225) & (b[20:160, 45:435] > 225)))
            if left_periwinkle > 0.65 and menu_white > 0.45:
                return True, "Main Menu (Continue / New Game)", False

            # 4. Oak Intro Screen: Mint-cyan upper backdrop + teal oval platform in center
            intro_mint = float(np.mean((r[40:140, 20:280] > 215) & (g[40:140, 20:280] > 235) & (b[40:140, 20:280] > 225)))
            intro_oval = float(np.mean((r[150:200, 150:330] < 90) & (g[150:200, 150:330] > 120) & (b[150:200, 150:330] > 115)))
            if intro_mint > 0.70 and intro_oval > 0.35:
                return True, "Professor Oak Intro Speech", True

            return False, "", False
        except Exception:
            return False, "", False

    def _is_shop_menu(self, arr: np.ndarray) -> bool:
        """Detects Poké Mart Buy/Sell item selection menu."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            money_box = float(np.mean((r[10:65, 10:185] > 220) & (g[10:65, 10:185] > 220) & (b[10:65, 10:185] > 210)))
            list_box = float(np.mean((r[15:205, 220:460] > 225) & (g[15:205, 220:460] > 225) & (b[15:205, 220:460] > 185)))
            bot_teal = float(np.mean((r[225:315, 200:460] < 60) & (g[225:315, 200:460] > 90) & (b[225:315, 200:460] > 130)))
            return bool(money_box > 0.45 and list_box > 0.55 and bot_teal > 0.35)
        except Exception:
            return False

    def _is_start_menu(self, arr: np.ndarray) -> bool:
        """Detects the in-game Start Menu on the right with dark blue bottom help banner."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            bot_navy = float(np.mean((r[248:314, 15:465] < 40) & (b[248:314, 15:465] > 55) & (b[248:314, 15:465] > r[248:314, 15:465] + 35)))
            right_menu = float(np.mean((r[12:180, 350:468] > 50) & (np.abs(r[12:180, 350:468] - g[12:180, 350:468]) < 15) & (np.abs(g[12:180, 350:468] - b[12:180, 350:468]) < 15)))
            return bool(bot_navy > 0.55 and right_menu > 0.50)
        except Exception:
            return False

    def _is_dialogue_active(self, arr: np.ndarray) -> bool:
        """Checks if a white bottom dialogue box with dark text is on screen."""
        try:
            dialogue_area = arr[230:300, 20:460, :]
            # Dialogue box interior is mostly pure white (> 220)
            white_pixels = np.mean((dialogue_area[:, :, 0] > 220) & (dialogue_area[:, :, 1] > 220) & (dialogue_area[:, :, 2] > 220))
            # Border is dark/grey (< 145)
            border_dark = np.mean(arr[216:228, 20:460, :] < 145) > 0.30
            return bool(white_pixels > 0.40 and border_dark)
        except Exception:
            return False

    def _classify_area(
        self,
        arr: np.ndarray,
        area_hint: str,
        is_battle: bool,
        is_naming: bool,
        is_trainer_card: bool = False,
        is_title_or_menu: bool = False,
        title_desc: str = "",
        is_shop_menu: bool = False,
        is_start_menu: bool = False,
        is_dialogue: bool = False
    ) -> Tuple[str, str]:
        """Classifies the environment based on structural colors, flooring, and void layout."""
        if is_battle:
            return "BATTLE", "In a Pokémon Battle"
        if is_naming:
            return "NAMING_SCREEN", "Character Naming Screen"
        if is_trainer_card:
            return "TRAINER_CARD", "Player Trainer Card Screen"
        if is_title_or_menu:
            return "TITLE_SCREEN", title_desc or "Title / Main Menu Screen"
        if is_shop_menu:
            return "POKEMART", "Poké Mart - Item Shop"

        h, w, _ = arr.shape
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # Void borders (< 25 RGB)
        void_mask = (r_all < 25) & (g_all < 25) & (b_all < 25)
        total_void = float(np.mean(void_mask))
        south_void = float(np.mean(void_mask[h - 40:, :])) > 0.25
        north_void = float(np.mean(void_mask[:40, :])) > 0.35
        west_void = float(np.mean(void_mask[:, :40])) > 0.35
        east_void = float(np.mean(void_mask[:, w - 40:])) > 0.35

        # 1. Outdoor Overworld Check:
        # Seamless full-screen map without black room void borders
        has_no_void_borders = (total_void < 0.04 and not north_void and not west_void and not east_void and not south_void)
        if has_no_void_borders:
            return "OVERWORLD", "Pallet Town / Kanto Overworld"

        # Inspect indoor floor region (y: 60..230)
        mid_floor = arr[60:230, :]
        mr = mid_floor[:, :, 0].astype(int)
        mg = mid_floor[:, :, 1].astype(int)
        mb = mid_floor[:, :, 2].astype(int)

        tatami = float(np.mean((mr > 140) & (mg > 120) & (mb < 110) & (mr > mb + 35)))
        green_rug = float(np.mean((mr < 140) & (mg > 140) & (mb < 150) & (mg > mr + 15)))
        tatami_score = tatami + green_rug
        mart_blue = float(np.mean((mb > mr + 30) & (mb > mg + 10) & (mb > 140)))
        lab_white = float(np.mean((mr > 180) & (mg > 180) & (mb > 180) & (np.abs(mr - mg) < 20) & (np.abs(mg - mb) < 25)))
        lab_dimmed = float(np.mean((mr > 55) & (mr < 90) & (np.abs(mr - mg) < 8) & (np.abs(mg - mb) < 8)))

        # 2. Poké Mart: Distinctive blue indoor floor tiles
        if mart_blue > 0.20 and tatami < 0.10:
            return "POKEMART", "Poké Mart - Item Shop"

        # 3. Oak's Research Lab: Bright white/grey diagonally-tiled floor + room void borders
        if (lab_white > 0.30 or (lab_dimmed > 0.30 and is_start_menu)) and tatami < 0.08 and total_void > 0.08:
            return "OAKS_LAB", "Pallet Town - Professor Oak's Research Lab"

        # 4. Poké Center: White/Pink floor, central healing counter with red Poké Ball emblem
        center_floor = (r_all > 220) & (g_all > 210) & (b_all > 210)
        red_counter_emblem = (r_all[60:140, :] > 190) & (g_all[60:140, :] < 80) & (b_all[60:140, :] < 80)
        if np.mean(center_floor) > 0.25 and np.sum(red_counter_emblem) > 150 and tatami < 0.08 and lab_white < 0.35:
            return "POKECENTER", "Pokémon Center - Free Healing Station"

        # 5. House Indoor Rooms (Tatami wooden floor + green rug): Distinguish 1F Living Room vs 2F Bedroom
        if tatami >= 0.12 or (tatami_score >= 0.18 and total_void >= 0.15):
            # Find the green floor rug (5th to 95th percentile bounding box)
            rug_mask = (r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)
            ry, rx = np.where(rug_mask)
            rug_blue_cnt = 0
            rug_bot_y = 220
            if len(ry) > 100:
                y1, y2 = int(np.percentile(ry, 5)), int(np.percentile(ry, 95))
                x1, x2 = int(np.percentile(rx, 5)), int(np.percentile(rx, 95))
                rug_bot_y = y2
                rug_patch = arr[y1:y2 + 1, x1:x2 + 1]
                rr = rug_patch[:, :, 0].astype(int)
                rg = rug_patch[:, :, 1].astype(int)
                rb = rug_patch[:, :, 2].astype(int)
                # In 1F Living Room, the blue-and-white checkered dining table + 4 blue chairs sit inside the green rug (>1500 px)
                # In 2F Bedroom, only the TV's 2 tiny blue screens sit inside the green rug (<=160 px)
                rug_blue_cnt = int(np.sum((rb > rr + 35) & (rb > 175) & (rg > 130)))

            if rug_blue_cnt >= 500:
                return "LIVING_ROOM_1F", "Pallet Town - Mom's Living Room (1F)"
            return "BEDROOM_2F", "Pallet Town - Red's Bedroom (2F)"

        # 6. Weak Fallback / Area Hint tie-breaker (only when visual features are ambiguous and not a full-screen overworld)
        if total_void > 0.10:
            hint_lower = area_hint.lower()
            if "living" in hint_lower or "1f" in hint_lower or "mom" in hint_lower:
                return "LIVING_ROOM_1F", "Pallet Town - Mom's Living Room (1F)"
            if "bedroom" in hint_lower or "2f" in hint_lower:
                return "BEDROOM_2F", "Pallet Town - Red's Bedroom (2F)"
            if "lab" in hint_lower:
                return "OAKS_LAB", "Pallet Town - Professor Oak's Research Lab"
            if "center" in hint_lower:
                return "POKECENTER", "Pokémon Center - Free Healing Station"
            if "mart" in hint_lower:
                return "POKEMART", "Poké Mart - Item Shop"
            if "gym" in hint_lower:
                return "GYM", "Pokémon Gym - Badge Challenge"

        return "OVERWORLD", "Pallet Town / Kanto Overworld"

    # -------------------------------------------------------------------------
    # Player Localization & Orientation
    # -------------------------------------------------------------------------
    def _detect_player(self, arr: np.ndarray) -> PlayerState:
        """Finds Red's sprite via red cap color (both Daytime/Indoor and Nighttime/Dusk palettes) and determines facing direction."""
        h, w, _ = arr.shape
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Daytime / Indoor bright red cap mask
        cap_mask = (r_all > 175) & (g_all < 105) & (b_all < 105)

        valid_cap_x: List[int] = []
        valid_cap_y: List[int] = []

        # Search tight camera-center window [115:175, 215:265] first so red building roofs/mats never pull player off-center
        for y_min, y_max, x_min, x_max in [(115, 175, 215, 265), (80, 230, 160, 325)]:
            cy, cx = np.where(cap_mask[y_min:y_max, x_min:x_max])
            cy += y_min
            cx += x_min
            for y_val, x_val in zip(cy, cx):
                patch = arr[min(h - 1, y_val + 6):min(h, y_val + 24), max(0, x_val - 8):min(w, x_val + 8)]
                skin = (patch[:, :, 0] > 200) & (patch[:, :, 1] > 140) & (patch[:, :, 1] < 210) & (patch[:, :, 2] > 90) & (patch[:, :, 2] < 165)
                dark_hair = (patch[:, :, 0] >= 30) & (patch[:, :, 0] < 90) & (patch[:, :, 1] >= 30) & (patch[:, :, 1] < 80) & (patch[:, :, 2] >= 30) & (patch[:, :, 2] < 80)
                if np.sum(skin) >= 6 or np.sum(dark_hair) >= 12:
                    valid_cap_x.append(int(x_val))
                    valid_cap_y.append(int(y_val))
            if len(valid_cap_x) >= 12:
                break

        # 2. Nighttime / Dusk Overworld shader cap mask (dark crimson cap with pure black sprite outline below)
        if len(valid_cap_x) < 12:
            night_cap = (
                (r_all >= 75) & (r_all <= 165) &
                (g_all <= 65) & (b_all <= 70) &
                (r_all >= g_all + 30) & (r_all >= b_all + 20)
            )
            ncy, ncx = np.where(night_cap[115:180, 210:270])
            ncy += 115
            ncx += 210
            for y_val, x_val in zip(ncy, ncx):
                patch = arr[min(h - 1, y_val + 4):min(h, y_val + 26), max(0, x_val - 10):min(w, x_val + 10)]
                black_outline = (patch[:, :, 0] < 15) & (patch[:, :, 1] < 15) & (patch[:, :, 2] < 15)
                if np.sum(black_outline) >= 10:
                    valid_cap_x.append(int(x_val))
                    valid_cap_y.append(int(y_val))

        if len(valid_cap_x) >= 12:
            py = int(np.median(valid_cap_y))
            px = int(np.median(valid_cap_x))
        else:
            cy, cx = np.where(cap_mask[115:175, 215:265])
            if len(cy) >= 12:
                py = int(np.median(cy)) + 115
                px = int(np.median(cx)) + 215
            else:
                return PlayerState(detected=False)

        # Ground collision tracking by FEET:
        # In GBA Pokémon, Red's sprite is 16x32 (feet are at ground collision level ~22px below cap center).
        feet_py = py + 22
        col, row = pixel_to_tile(px, feet_py)

        # Determine facing direction by inspecting sprite patch below cap
        y1, y2 = max(0, py - 6), min(h, py + 22)
        x1, x2 = max(0, px - 12), min(w, px + 12)
        patch = arr[y1:y2, x1:x2]
        pr = patch[:, :, 0].astype(int)
        pg = patch[:, :, 1].astype(int)
        pb = patch[:, :, 2].astype(int)

        # Skin tone pixels (Daytime + Nighttime shader)
        day_skin = (pr > 210) & (pg > 140) & (pg < 220) & (pb > 90) & (pb < 170)
        night_skin = (pr >= 85) & (pr <= 135) & (pg >= 50) & (pg <= 95) & (pr >= pg + 20)
        skin_count = int(np.sum(day_skin | night_skin))

        # Dark hair / back of cap pixels
        dark_hair = (pr < 90) & (pg < 80) & (pb < 80)
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
        """Analyzes Red's 2F Bedroom fixtures: green rug anchor, stairs, railing, bed, TV, PC, void borders."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Locate the central 5x4 green floor rug to anchor the entire room regardless of camera scroll
        rug_mask = (r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)
        rug_cols: List[int] = []
        rug_rows: List[int] = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = rug_mask[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if patch.size > 0 and float(np.mean(patch)) > 0.25:
                    rug_cols.append(c)
                    rug_rows.append(r)

        if rug_cols and rug_rows:
            rug_c1, rug_c2 = min(rug_cols), max(rug_cols)
            rug_r1, rug_r2 = min(rug_rows), max(rug_rows)
        else:
            rug_c1, rug_c2, rug_r1, rug_r2 = 4, 8, 5, 8

        # Nominal orange landing mat is 2 tiles East of rug_c2 and 2 tiles North of rug_r1
        nom_mat_col = rug_c2 + 2
        nom_mat_row = rug_r1 - 2

        # Check for visible orange landing mat tile (excluding North Wall notice sign which has pure white interior)
        mat_pixels = (
            (r_all > 190) &
            (g_all > 110) & (g_all < 165) &
            (b_all > 70) & (b_all < 125)
        )
        pure_white = (r_all > 240) & (g_all > 240) & (b_all > 240)
        detected_mat_tile: Optional[Tuple[int, int]] = None
        for r in range(max(0, rug_r1 - 3), min(GRID_ROWS, rug_r1 + 1)):
            for c in range(max(0, rug_c2), GRID_COLS):
                x1, y1, x2, y2 = tile_bounds(c, r)
                mp = mat_pixels[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                wp = pure_white[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if mp.size > 0 and float(np.mean(mp)) > 0.22 and int(np.sum(wp)) < 25:
                    detected_mat_tile = (c, r)
                    break
            if detected_mat_tile:
                break

        if detected_mat_tile is not None:
            mat_col, mat_row = detected_mat_tile
        else:
            mat_col = max(1, min(GRID_COLS - 1, nom_mat_col))
            mat_row = max(0, min(GRID_ROWS - 1, nom_mat_row))

        # Room offset relative to nominal view
        dx = rug_c1 - 4
        dy = rug_r1 - 6

        # Stairs Warp: 1 tile WEST (LEFT) of the orange landing mat
        stairs_col = max(0, mat_col - 1)
        stairs_row = mat_row

        # 2. Room Void Borders & North Wall
        col_left = max(0, rug_c1 - 3)
        col_right = min(GRID_COLS - 1, rug_c2 + 3)
        if col_left > 0:
            grid[:, :col_left] = CELL_VOID
        if col_right + 1 < GRID_COLS:
            grid[:, col_right + 1:] = CELL_VOID

        north_wall_bottom = rug_r1 - 3
        for r in range(0, max(0, north_wall_bottom + 1)):
            for c in range(col_left, col_right + 1):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_WALL

        # 3. Staircase Safety Railing & Steps (relative to green rug so it is accurate even when mat is off-screen)
        rail_west_col = rug_c2
        rail_bot_row = rug_r1 - 1
        for r in range(max(0, rug_r1 - 3), max(0, rail_bot_row + 1)):
            if 0 <= rail_west_col < GRID_COLS and 0 <= r < GRID_ROWS:
                grid[r, rail_west_col] = CELL_RAILING
        if 0 <= rug_c2 + 1 < GRID_COLS and 0 <= rail_bot_row < GRID_ROWS:
            grid[rail_bot_row, rug_c2 + 1] = CELL_RAILING

        # Upper wall behind staircase
        if 0 <= nom_mat_row - 1 < GRID_ROWS:
            for c in [rug_c2 + 1, rug_c2 + 2]:
                if 0 <= c < GRID_COLS and grid[nom_mat_row - 1, c] != CELL_VOID:
                    grid[nom_mat_row - 1, c] = CELL_WALL

        # 4. Bed and TV Furniture Obstacles
        # Bed: 2 tiles wide to the West of the green rug
        for r in range(max(0, rug_r1 + 1), min(GRID_ROWS, rug_r2 + 2)):
            for c in range(max(0, rug_c1 - 3), max(0, rug_c1 - 1)):
                if 0 <= c < GRID_COLS and grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # TV Console / NES Stand: center column of green rug
        tv_col = (rug_c1 + rug_c2) // 2
        for r in range(max(0, rug_r1 - 1), min(GRID_ROWS, rug_r1 + 2)):
            if 0 <= tv_col < GRID_COLS and grid[r, tv_col] == CELL_WALKABLE:
                grid[r, tv_col] = CELL_OBSTACLE

        # Keep player's current tile walkable
        if player.detected and 0 <= player.tile_row < GRID_ROWS and 0 <= player.tile_col < GRID_COLS:
            grid[player.tile_row, player.tile_col] = CELL_WALKABLE

        # If the staircase is currently scrolled off the top/right edge of the screen, place the target on the nearest Northeast walkable floor tile
        if nom_mat_row < 0 or nom_mat_col >= GRID_COLS or grid[stairs_row, stairs_col] == CELL_VOID:
            best_ne = None
            for r in range(max(0, rug_r1 - 2), min(GRID_ROWS, rug_r1 + 2)):
                for c in range(min(GRID_COLS - 1, rug_c2 + 2), max(0, rug_c1), -1):
                    if grid[r, c] == CELL_WALKABLE:
                        best_ne = (c, r)
                        break
                if best_ne:
                    break
            if best_ne:
                mat_col, mat_row = best_ne
                stairs_col, stairs_row = best_ne

        grid[mat_row, mat_col] = CELL_WALKABLE
        grid[stairs_row, stairs_col] = CELL_WARP

        stairs_px, stairs_py = tile_to_pixel_center(stairs_col, stairs_row)
        pois.append(POI(
            name="STAIRS_DOWN",
            category="WARP",
            tile_col=stairs_col,
            tile_row=stairs_row,
            pixel_x=stairs_px,
            pixel_y=stairs_py,
            description="Staircase descending to 1F Living Room (Head onto orange mat then step WEST/LEFT into stairs)",
            interaction_tile=(stairs_col, stairs_row)
        ))

        mat_px, mat_py = tile_to_pixel_center(mat_col, mat_row)
        pois.append(POI(
            name="STAIRS_MAT",
            category="WARP",
            tile_col=mat_col,
            tile_row=mat_row,
            pixel_x=mat_px,
            pixel_y=mat_py,
            description="Orange stairs landing mat (Step here, then step WEST into STAIRS_DOWN)",
            interaction_tile=(stairs_col, stairs_row)
        ))

        sign_col = max(0, min(GRID_COLS - 1, rug_c2 + 2))
        sign_row = max(0, min(GRID_ROWS - 1, max(0, rug_r1 - 3)))
        sign_px, sign_py = tile_to_pixel_center(sign_col, sign_row)
        pois.append(POI(
            name="WALL_SIGN",
            category="OBJECT",
            tile_col=sign_col,
            tile_row=sign_row,
            pixel_x=sign_px,
            pixel_y=sign_py,
            description="Notice Sign on North Wall (Wall fixture, not an exit)",
            interaction_tile=(sign_col, min(GRID_ROWS - 1, sign_row + 1))
        ))

        pc_col = max(0, min(GRID_COLS - 1, rug_c1 - 3))
        pc_row = max(0, min(GRID_ROWS - 1, max(0, rug_r1 - 3)))
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

        notes.extend([
            "Red is upstairs in his 2F Bedroom.",
            f"PRIMARY EXIT: Staircase to 1F is located at Col {stairs_col}, Row {stairs_row} (via orange landing mat at Col {mat_col}, Row {mat_row}).",
            f"NAVIGATION RULE: Choose 'EXIT_ROOM' or target 'STAIRS_DOWN' to route around the white railing onto the orange mat and step WEST into the stairs!",
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
        available_intents: List[str],
        is_dialogue: bool = False
    ):
        """Analyzes 1F Living Room: green rug anchor, front door exit mat, dining table with Mom, stairs up to 2F."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Locate the central 6x4 green floor rug to anchor the room
        rug_mask = (r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)
        rug_cols: List[int] = []
        rug_rows: List[int] = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = rug_mask[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if patch.size > 0 and float(np.mean(patch)) > 0.25:
                    rug_cols.append(c)
                    rug_rows.append(r)

        if rug_cols and rug_rows:
            rug_c1, rug_c2 = min(rug_cols), max(rug_cols)
            rug_r1, rug_r2 = min(rug_rows), max(rug_rows)
        else:
            rug_c1, rug_c2, rug_r1, rug_r2 = 2, 7, 4, 7

        # 2. Front Door Exit Mat: Strictly below the green rug (so Mom's red dress inside the rug is never matched)
        rug_bot_y = (rug_r2 + 1) * TILE_SIZE + GRID_Y_OFFSET - 8
        max_y_search = 215 if is_dialogue else self.target_height
        red_mat = (
            (r_all > 165) &
            (r_all > g_all + 35) &
            (r_all > b_all + 35) &
            (np.arange(self.target_height)[:, None] >= max(140, rug_bot_y)) &
            (np.arange(self.target_height)[:, None] < max_y_search)
        )
        my, mx = np.where(red_mat)
        if len(my) >= 20:
            exit_col, exit_row = pixel_to_tile(int(np.median(mx)), int(np.median(my)))
        else:
            exit_col = max(0, min(GRID_COLS - 1, rug_c1))
            exit_row = max(3, min(GRID_ROWS - 1, rug_r2 + 2))

        # 3. Mark North Wall above rug, Dining Table + Chairs inside rug, and Staircase on East wall
        north_wall_row = rug_r1 - 2
        for r in range(0, max(0, north_wall_row + 1)):
            for c in range(GRID_COLS):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_WALL

        # Dining table, 4 blue chairs, and Mom occupy the interior of the green rug: (rug_c1+1 .. rug_c2-1, rug_r1+1 .. rug_r2-1)
        for r in range(max(0, rug_r1 + 1 if rug_r1 > 0 else 0), min(GRID_ROWS, rug_r2)):
            for c in range(max(0, rug_c1 + 1), min(GRID_COLS, rug_c2)):
                if (c, r) != (player.tile_col, player.tile_row) and (c, r) != (exit_col, exit_row):
                    grid[r, c] = CELL_OBSTACLE

        # Staircase up to 2F on the East side (Cols >= rug_c2 + 2, Rows rug_r1 - 2 .. rug_r1 - 1)
        for r in range(max(0, rug_r1 - 2), min(GRID_ROWS, rug_r1)):
            for c in range(min(GRID_COLS, rug_c2 + 2), GRID_COLS):
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # 4. Mark Front Door warp tiles (and if Red is already standing on the mat, target 1 tile SOUTH to step out!)
        door_interact_col, door_interact_row = exit_col, exit_row
        if player.detected and (player.tile_col, player.tile_row) == (exit_col, exit_row) and exit_row + 1 < GRID_ROWS:
            door_interact_row = exit_row + 1

        grid[exit_row, exit_col] = CELL_WARP
        grid[door_interact_row, door_interact_col] = CELL_WARP
        if exit_col + 1 < GRID_COLS and grid[exit_row, exit_col + 1] != CELL_VOID:
            grid[exit_row, exit_col + 1] = CELL_WARP

        exit_px, exit_py = tile_to_pixel_center(exit_col, exit_row)
        door_poi = POI(
            name="FRONT_DOOR",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=exit_px,
            pixel_y=exit_py,
            description="Front Door exit mat leading outside to Pallet Town",
            interaction_tile=(door_interact_col, door_interact_row)
        )

        # 5. Mom seated on the right side of the dining table inside the green rug
        mom_col = max(0, min(GRID_COLS - 1, rug_c2 - 1))
        mom_row = max(0, min(GRID_ROWS - 1, rug_r1 + 1 if rug_r1 > 0 else 0))
        mom_interact = (mom_col, max(0, mom_row - 1)) if mom_row > 0 else (min(GRID_COLS - 1, mom_col + 1), mom_row)
        mom_px, mom_py = tile_to_pixel_center(mom_col, mom_row)
        mom_poi = POI(
            name="MOM",
            category="NPC",
            tile_col=mom_col,
            tile_row=mom_row,
            pixel_x=mom_px,
            pixel_y=mom_py,
            description="Red's Mom (Restores party HP)",
            interaction_tile=mom_interact
        )

        # FRONT_DOOR is placed first so it is the primary WARP target
        pois.extend([door_poi, mom_poi])

        notes.extend([
            "Red is downstairs on 1F in Mom's Living Room.",
            f"PRIMARY EXIT: Front door to Pallet Town is at Col {exit_col}, Row {exit_row} (Red welcome mat).",
            f"NAVIGATION RULE: Choose 'EXIT_ROOM' or target 'FRONT_DOOR' to step onto the red mat at Col {door_interact_col}, Row {door_interact_row} and exit into Pallet Town!",
            f"NPC: Mom is at Col {mom_col}, Row {mom_row} (seated at the dining table).",
            "VOID WALLS: Room boundary walls and black void borders are impassable."
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
        """Analyzes Professor Oak's Research Lab dynamically across any camera scroll position."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Dynamically locate the olive-green Starter Table (3 tiles wide x 1 tile tall on right side of Lab)
        green_tbl = (g_all > r_all + 25) & (g_all > b_all + 25) & (g_all > 130) & (r_all > 80)
        tbl_cols: List[int] = []
        tbl_rows: List[int] = []
        for r in range(1, GRID_ROWS - 1):
            for c in range(5, GRID_COLS - 1):
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = green_tbl[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if patch.size > 0 and float(np.mean(patch)) > 0.20:
                    tbl_cols.append(c)
                    tbl_rows.append(r)

        if tbl_cols and tbl_rows:
            tc1, tc2 = min(tbl_cols), max(tbl_cols)
            tr1 = min(tbl_rows)
        else:
            tc1, tc2, tr1 = 8, 10, 4

        ball_col = (tc1 + tc2) // 2
        ball_row = tr1
        oak_col = max(1, tc1 - 2)
        oak_row = tr1
        exit_col = max(1, tc1 - 2)
        exit_row = min(GRID_ROWS - 1, tr1 + 6)

        # 2. Mark Lab Walls, Bookshelves, Starter Table, and Oak as Obstacles
        # North wall & upper bookshelves
        for r in range(0, max(0, tr1 - 2)):
            for c in range(GRID_COLS):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_WALL

        if 0 <= tr1 - 2 < GRID_ROWS:
            for c in range(GRID_COLS):
                if (c <= tc1 - 3 or c >= tc1) and grid[tr1 - 2, c] != CELL_VOID:
                    grid[tr1 - 2, c] = CELL_WALL

        # Left Poké Ball machine table
        for r in range(max(0, tr1 - 1), min(GRID_ROWS, tr1 + 1)):
            for c in range(max(0, tc1 - 5), max(0, tc1 - 3)):
                if 0 <= c < GRID_COLS and grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # Starter Table on the right
        for c in range(tc1, min(GRID_COLS, tc2 + 1)):
            if 0 <= tr1 < GRID_ROWS:
                grid[tr1, c] = CELL_OBSTACLE

        # Professor Oak NPC tile
        if 0 <= oak_row < GRID_ROWS and 0 <= oak_col < GRID_COLS:
            grid[oak_row, oak_col] = CELL_NPC

        # Middle row of bookshelves (with 2-tile central aisle at tc1-3 .. tc1-2)
        shelf_row = tr1 + 3
        if 0 <= shelf_row < GRID_ROWS:
            for c in range(GRID_COLS):
                if (c <= tc1 - 4 or c >= tc1 - 1) and grid[shelf_row, c] == CELL_WALKABLE:
                    grid[shelf_row, c] = CELL_OBSTACLE

        # Keep player's tile walkable
        if player.detected and 0 <= player.tile_row < GRID_ROWS and 0 <= player.tile_col < GRID_COLS:
            grid[player.tile_row, player.tile_col] = CELL_WALKABLE

        grid[exit_row, exit_col] = CELL_WARP

        # 3. Populate POIs
        starter_interact = (ball_col, min(GRID_ROWS - 1, ball_row + 1)) if ball_row + 1 < GRID_ROWS else (ball_col, max(1, ball_row - 1))
        bpx, bpy = tile_to_pixel_center(ball_col, ball_row)
        pois.append(POI(
            name="STARTER_TABLE",
            category="OBJECT",
            tile_col=ball_col,
            tile_row=ball_row,
            pixel_x=bpx,
            pixel_y=bpy,
            description=f"Starter Poké Ball on table (Approach at Col {starter_interact[0]}, Row {starter_interact[1]} and press A)",
            interaction_tile=starter_interact
        ))

        oak_interact = (oak_col, min(GRID_ROWS - 1, oak_row + 1)) if (not player.detected or player.tile_row >= oak_row) else (oak_col, max(1, oak_row - 1))
        opx, opy = tile_to_pixel_center(oak_col, oak_row)
        pois.append(POI(
            name="PROFESSOR_OAK",
            category="NPC",
            tile_col=oak_col,
            tile_row=oak_row,
            pixel_x=opx,
            pixel_y=opy,
            description="Professor Oak",
            interaction_tile=oak_interact
        ))

        epx, epy = tile_to_pixel_center(exit_col, exit_row)
        pois.append(POI(
            name="LAB_EXIT",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=epx,
            pixel_y=epy,
            description="South exit doorway back to Pallet Town",
            interaction_tile=(exit_col, exit_row)
        ))

        notes.extend([
            "Red is inside Professor Oak's Research Lab.",
            f"STARTER POKÉMON: Starter Poké Ball table is at Col {ball_col}, Row {ball_row} (Interact from Col {starter_interact[0]}, Row {starter_interact[1]}).",
            f"NPC: Professor Oak is at Col {oak_col}, Row {oak_row}.",
            f"EXIT: South doorway to Pallet Town is down the central aisle at Col {exit_col}, Row {exit_row}."
        ])

        available_intents.extend(["GO_TO_POI", "TALK_TO_NPC", "EXIT_ROOM", "WAIT"])

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
        jpx, jpy = tile_to_pixel_center(7, 2)
        pois.append(POI(
            name="NURSE_JOY",
            category="NPC",
            tile_col=7,
            tile_row=2,
            pixel_x=jpx,
            pixel_y=jpy,
            description="Nurse Joy (Free party healing)",
            interaction_tile=(7, 3)
        ))

        # PC terminal on right wall
        pcpx, pcpy = tile_to_pixel_center(12, 2)
        pois.append(POI(
            name="PC",
            category="OBJECT",
            tile_col=12,
            tile_row=2,
            pixel_x=pcpx,
            pixel_y=pcpy,
            description="Pokémon Storage System PC",
            interaction_tile=(12, 3)
        ))

        # Exit door at south center
        epx, epy = tile_to_pixel_center(7, 8)
        pois.append(POI(
            name="CENTER_EXIT",
            category="WARP",
            tile_col=7,
            tile_row=8,
            pixel_x=epx,
            pixel_y=epy,
            description="Exit door back to town",
            interaction_tile=(7, 8)
        ))

        # Counter obstacle across rows 2-3
        for c in range(4, 11):
            grid[2, c] = CELL_COUNTER
        grid[8, 7] = CELL_WARP

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
        available_intents: List[str],
        is_dialogue: bool = False,
        is_shop_menu: bool = False
    ):
        """Analyzes Poké Mart interior dynamically across camera scroll positions."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Locate red welcome mat at the south doorway (above dialogue box if active)
        max_y_search = 215 if (is_dialogue or is_shop_menu) else self.target_height
        red_mat = (
            (r_all > 165) &
            (r_all > g_all + 35) &
            (r_all > b_all + 35) &
            (np.arange(self.target_height)[:, None] >= 155) &
            (np.arange(self.target_height)[:, None] < max_y_search)
        )
        my, mx = np.where(red_mat)
        if len(my) >= 20 and not is_shop_menu and not is_dialogue:
            exit_col, exit_row = pixel_to_tile(int(np.median(mx)), int(np.median(my)))
        else:
            exit_col, exit_row = 7, 7

        # In the Poké Mart, the Cashier Clerk stands behind the left counter at (exit_col - 2, exit_row - 3)
        clerk_col = max(1, exit_col - 2)
        clerk_row = max(1, exit_row - 3)
        counter_col = max(2, exit_col - 1)
        clerk_interact = (exit_col, clerk_row)

        # Mark Counter on the left and Shelves on the right & top
        for r in range(max(0, clerk_row - 1), min(GRID_ROWS, clerk_row + 2)):
            for c in range(0, min(GRID_COLS, counter_col + 1)):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_COUNTER

        for r in range(0, max(0, clerk_row - 1)):
            for c in range(GRID_COLS):
                if grid[r, c] != CELL_VOID:
                    grid[r, c] = CELL_WALL

        for r in range(max(0, clerk_row - 1), min(GRID_ROWS, exit_row)):
            for c in range(min(GRID_COLS, exit_col + 2), GRID_COLS):
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        if player.detected and 0 <= player.tile_row < GRID_ROWS and 0 <= player.tile_col < GRID_COLS:
            grid[player.tile_row, player.tile_col] = CELL_WALKABLE

        exit_interact = (exit_col, min(GRID_ROWS - 1, exit_row + 1)) if (player.detected and (player.tile_col, player.tile_row) == (exit_col, exit_row) and exit_row + 1 < GRID_ROWS) else (exit_col, exit_row)
        grid[exit_row, exit_col] = CELL_WARP
        grid[exit_interact[1], exit_interact[0]] = CELL_WARP

        cpx, cpy = tile_to_pixel_center(clerk_col, clerk_row)
        pois.append(POI(
            name="MART_CLERK",
            category="NPC",
            tile_col=clerk_col,
            tile_row=clerk_row,
            pixel_x=cpx,
            pixel_y=cpy,
            description=f"Cashier Clerk (Stand at Col {clerk_interact[0]}, Row {clerk_interact[1]} facing LEFT and press A)",
            interaction_tile=clerk_interact
        ))

        epx, epy = tile_to_pixel_center(exit_col, exit_row)
        pois.append(POI(
            name="MART_EXIT",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=epx,
            pixel_y=epy,
            description="South exit mat leading back outside",
            interaction_tile=exit_interact
        ))

        notes.extend([
            "Red is inside the Poké Mart.",
            f"SHOPPING: Cashier Clerk is at Col {clerk_col}, Row {clerk_row} (Approach counter at Col {clerk_interact[0]}, Row {clerk_interact[1]}).",
            f"EXIT: South exit mat is at Col {exit_col}, Row {exit_row}."
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
        """Analyzes outdoor Overworld (Pallet Town, Route 1, Viridian City): maps trees, ledges, fences, tall grass, buildings, and doors."""
        h, w, _ = arr.shape
        tree_mask_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

        # 1. Build 15x10 Overworld Collision Grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if player.detected and (c, r) == (player.tile_col, player.tile_row):
                    grid[r, c] = CELL_WALKABLE
                    continue

                x1, y1, x2, y2 = tile_bounds(c, r)
                y1_c, y2_c = max(0, y1), min(h, y2)
                x1_c, x2_c = max(0, x1), min(w, x2)

                # Follower Pikachu check on the 4 cardinal tiles adjacent to Red:
                # In Pokémon Recharged Yellow, Red can walk freely through his follower Pikachu (they swap tiles)!
                if player.detected and abs(c - player.tile_col) + abs(r - player.tile_row) == 1:
                    full_patch = arr[y1_c:y2_c, x1_c:x2_c]
                    if full_patch.size > 0:
                        fpr = full_patch[:, :, 0].astype(int)
                        fpg = full_patch[:, :, 1].astype(int)
                        fpb = full_patch[:, :, 2].astype(int)
                        pika_yellow = float(np.mean(
                            ((fpr > 195) & (fpg > 165) & (fpb < 95)) |
                            ((fpr > 105) & (fpg > 90) & (fpb < 50) & (fpr > fpb + 55))
                        ))
                        if pika_yellow > 0.12:
                            grid[r, c] = CELL_WALKABLE
                            continue

                # For the tile directly NORTH of Red, inspect upper 14px so Red's cap top doesn't trigger an obstacle
                if player.detected and c == player.tile_col and r == player.tile_row - 1:
                    y2_c = min(y2_c, y1_c + 14)

                if y2_c <= y1_c or x2_c <= x1_c:
                    continue

                patch = arr[y1_c:y2_c, x1_c:x2_c]
                std_val = float(patch.std(axis=(0, 1)).mean())
                pr = patch[:, :, 0].astype(int)
                pg = patch[:, :, 1].astype(int)
                pb = patch[:, :, 2].astype(int)

                # Check if walkable Tall Wild Grass (green grass clump on mint-cyan lawn background):
                # Every pixel is green (all_green >= 0.72), top 4px show lawn background (top4_lawn >= 0.60), and 25%-55% of tile is lawn backdrop
                all_green = float(np.mean(pg > pr + 15))
                lawn_backdrop = float(np.mean((pb > pr + 12) & (pg > pr + 20) & (pb >= 75)))
                top4 = patch[:min(4, patch.shape[0]), :]
                top4_lawn = float(np.mean(
                    (top4[:, :, 2].astype(int) > top4[:, :, 0].astype(int) + 12) &
                    (top4[:, :, 1].astype(int) > top4[:, :, 0].astype(int) + 20) &
                    (top4[:, :, 2].astype(int) >= 75)
                )) if top4.size > 0 else 0.0

                green_foliage = float(np.mean((pg > pr + 22) & (pg >= pb)))
                tree_shadow = float(np.mean((pr < 30) & (pg < 45) & (pb < 25)))
                slate_struct = float(np.mean((pb > pg + 12) & (pr < 65)))
                is_tall_grass = (
                    (all_green >= 0.72 and 0.26 <= lawn_backdrop <= 0.55 and top4_lawn >= 0.60)
                    or (green_foliage > 0.50 and tree_shadow < 0.06 and slate_struct < 0.08)
                )

                # Check if Pine / Forest Tree tile (impassable canopy with low lawn_backdrop)
                tree_green = float(np.mean((pg > pr + 18) & (pg > pb + 10)))
                if not is_tall_grass and tree_green > 0.35 and std_val >= 16.5 and lawn_backdrop < 0.25:
                    tree_mask_grid[r, c] = True

                # A) Walkable flat ground / dirt road / short lawn grass / dirt-to-grass border (which has no dark shadows min >= 75)
                roof_px = float(np.mean((pr > pg + 25) & (pr > pb + 15)))
                water_px = float(np.mean((pb > pg + 25) & (pb > pr + 30)))
                min_channel = int(patch.min())
                is_flat_ground = (
                    (std_val < 16.5 or (std_val < 26.0 and min_channel >= 75 and all_green > 0.25))
                    and roof_px < 0.35
                    and water_px < 0.45
                )

                # B) Walkable flowers (lawn grass background + red petals)
                lawn_bg = float(np.mean((pg > pr + 18) & (pg > 70)))
                red_petal = float(np.mean((pr > pg + 25) & (pr > pb + 15) & (pg < 75)))
                is_flowers = (lawn_bg >= 0.35 and red_petal >= 0.08)

                if is_flat_ground or is_flowers or is_tall_grass:
                    grid[r, c] = CELL_WALKABLE
                else:
                    grid[r, c] = CELL_OBSTACLE

        # 2. Detect Building Front Walls & Doors (at least 2 tiles tall, architectural non-foliage structure, with walkable path below)
        found_target_warp = False
        just_exited_house = False

        for r in range(1, GRID_ROWS - 1):
            c = 0
            while c < GRID_COLS:
                if grid[r, c] != CELL_WALKABLE and not tree_mask_grid[r, c] and grid[r - 1, c] != CELL_WALKABLE and not tree_mask_grid[r - 1, c]:
                    c_start = c
                    while (
                        c < GRID_COLS
                        and grid[r, c] != CELL_WALKABLE
                        and not tree_mask_grid[r, c]
                        and grid[r - 1, c] != CELL_WALKABLE
                        and not tree_mask_grid[r - 1, c]
                    ):
                        c += 1
                    c_end = c - 1
                    width = c_end - c_start + 1

                    # Check if this row is the bottom front wall of a building (walkable tiles directly below it)
                    walkable_below = [
                        col for col in range(c_start, c_end + 1)
                        if grid[r + 1, col] == CELL_WALKABLE
                    ]
                    if width >= 3 and len(walkable_below) >= 2:
                        # Verify this 2-row structure is architectural (roof + wall) and not Route 1 foliage/ledges
                        bx1, by1, _, _ = tile_bounds(c_start, r - 1)
                        _, _, bx2, by2 = tile_bounds(c_end, r)
                        b_patch = arr[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                        if b_patch.size > 0:
                            br = b_patch[:, :, 0].astype(int)
                            bg = b_patch[:, :, 1].astype(int)
                            bb = b_patch[:, :, 2].astype(int)
                            foliage_ratio = float(np.mean((bg > br + 20) & (bg > bb + 10)))
                            if foliage_ratio > 0.25:
                                continue

                        # If clipped at right screen edge and narrow, door is still off-screen to the East
                        if c_end == GRID_COLS - 1 and width < 4:
                            continue

                        best_door_col = None
                        best_score = -1.0
                        # Include c_end when building is clipped at right screen edge (e.g., Oak's Lab door at Col 14)
                        search_end = (c_end + 1) if c_end == GRID_COLS - 1 else c_end
                        for col in range(c_start + 1, search_end):
                            if grid[r + 1, col] != CELL_WALKABLE:
                                continue
                            x1, y1, x2, y2 = tile_bounds(col, r)
                            dp = arr[max(0, y1 + 6):min(h, y1 + 28), max(0, x1 + 4):min(w, x1 + 28)]
                            if dp.size == 0:
                                continue
                            dr = dp[:, :, 0].astype(int)
                            dg = dp[:, :, 1].astype(int)
                            db = dp[:, :, 2].astype(int)
                            # 1) Red/Orange/Crimson wooden house door
                            red_door = float(np.mean(
                                ((dr > dg + 42) & (dr > db + 35) & (dg < 125)) |
                                ((dr > dg + 18) & (dr > db + 10) & (dr > 55) & (dg < 70))
                            ))
                            # 2) Green/Teal door (Professor Oak's Lab & Rival's House)
                            teal_door = float(np.mean((dg > dr + 18) & (dg > 95)))
                            # 3) Blue/Cyan sliding glass door (Poké Center & Poké Mart)
                            blue_door = float(np.mean((db > dr + 25) & (dg > dr + 12) & (db > 120)))
                            door_score = max(red_door, teal_door, blue_door)
                            if door_score > best_score:
                                best_score = door_score
                                best_door_col = col

                        # Check if Red is standing right in front of this building having just exited it
                        is_exited_building = (
                            player.detected
                            and r == player.tile_row - 1
                            and c_start <= player.tile_col <= c_end
                        )

                        if is_exited_building:
                            just_exited_house = True
                            door_c = player.tile_col
                            grid[r, door_c] = CELL_WALL
                            dpx, dpy = tile_to_pixel_center(door_c, r)
                            pois.append(POI(
                                name="EXITED_HOUSE_DOOR",
                                category="OBJECT",
                                tile_col=door_c,
                                tile_row=r,
                                pixel_x=dpx,
                                pixel_y=dpy,
                                description="Door of the building Red just exited (Do NOT re-enter if heading elsewhere)",
                                interaction_tile=(door_c, r + 1)
                            ))
                        elif best_door_col is not None and best_score >= 0.25:
                            grid[r, best_door_col] = CELL_WARP
                            if best_door_col - 1 >= 0:
                                grid[r, best_door_col - 1] = CELL_WALL
                            if best_door_col + 1 < GRID_COLS:
                                grid[r, best_door_col + 1] = CELL_WALL
                            dpx, dpy = tile_to_pixel_center(best_door_col, r)
                            is_lab_door = (width >= 6 or (c_end == GRID_COLS - 1 and r >= 6))
                            door_name = "OAKS_LAB" if is_lab_door else "BUILDING_DOOR"
                            door_desc = (
                                "Professor Oak's Lab Entrance Door (Walk onto this doorway to enter)"
                                if door_name == "OAKS_LAB"
                                else "Building Entrance Door (Walk onto this doorway to enter)"
                            )
                            pois.insert(0, POI(
                                name=door_name,
                                category="WARP",
                                tile_col=best_door_col,
                                tile_row=r,
                                pixel_x=dpx,
                                pixel_y=dpy,
                                description=door_desc,
                                interaction_tile=(best_door_col, r)
                            ))
                            found_target_warp = True
                else:
                    c += 1

        # 3. Check if Professor Oak's Lab is partially visible on the East/Southeast edge (or when routing Southeast in Pallet Town)
        if not found_target_warp:
            right_edge_building_rows = [
                r for r in range(4, GRID_ROWS)
                if grid[r, GRID_COLS - 1] != CELL_WALKABLE and not tree_mask_grid[r, GRID_COLS - 1]
            ]
            if len(right_edge_building_rows) >= 2 or just_exited_house:
                # Find the best walkable tile on the Southeast corridor (Cols 11..13, Rows 8..9)
                se_candidates = [
                    (12, 9), (11, 9), (13, 9),
                    (12, 8), (11, 8), (13, 8),
                    (14, 9), (14, 8)
                ]
                se_target = None
                for sc, sr in se_candidates:
                    if 0 <= sc < GRID_COLS and 0 <= sr < GRID_ROWS and grid[sr, sc] == CELL_WALKABLE:
                        se_target = (sc, sr)
                        break

                if se_target is not None:
                    sc, sr = se_target
                    spx, spy = tile_to_pixel_center(sc, sr)
                    pois.insert(0, POI(
                        name="OAKS_LAB",
                        category="WARP",
                        tile_col=sc,
                        tile_row=sr,
                        pixel_x=spx,
                        pixel_y=spy,
                        description="Southeast path around fence alongside Professor Oak's Lab leading to the Lab entrance",
                        interaction_tile=(sc, sr)
                    ))
                    notes.append(
                        f"OAK'S LAB ROUTE: Professor Oak's Lab is to the SOUTHEAST! Choose 'EXPLORE_SOUTHEAST', 'GO_TO_POI', or 'ENTER_BUILDING' with target 'OAKS_LAB' to route around the fence to Tile (Col {sc}, Row {sr})."
                    )

        notes.extend([
            "Red is outdoors in the Overworld.",
            "NAVIGATION: Impassable trees, buildings, and fences are mapped. Pathfinding will automatically route around fences and walls.",
            "Ledges can only be hopped going SOUTH. You cannot jump UP over brown ledges."
        ])

        available_intents.extend([
            "GO_TO_POI",
            "ENTER_BUILDING",
            "EXPLORE_SOUTHEAST",
            "EXPLORE_SOUTH",
            "EXPLORE_EAST",
            "EXPLORE_NORTH",
            "EXPLORE_WEST",
            "EXPLORE_NORTHEAST",
            "EXPLORE_NORTHWEST",
            "EXPLORE_SOUTHWEST",
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
