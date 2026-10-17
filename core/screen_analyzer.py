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
from collections import deque
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


# Native GBA sign templates (2x downsampled luminance, normalized for zero-mean unit-variance)
# Captures gray metal town/route signs and wooden notice signs invariant to Day/Dusk/Night lighting
_TPL_GRAY_INT = np.array([
    [82, 82, 165, 165, 82, 82, 82, 82, 82, 82, 165, 165, 82, 82],
    [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    [189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189],
    [189, 121, 121, 121, 121, 189, 189, 189, 189, 121, 121, 121, 121, 189],
    [189, 189, 121, 189, 189, 189, 189, 143, 189, 189, 189, 121, 189, 189],
    [189, 143, 121, 189, 121, 189, 121, 121, 189, 121, 189, 121, 143, 189],
    [189, 189, 143, 189, 143, 189, 189, 143, 189, 143, 189, 143, 189, 189],
    [189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189],
    [126, 126, 165, 165, 165, 126, 126, 126, 126, 126, 165, 165, 165, 126],
    [189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189],
    [82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82]
], dtype=float)

_TPL_WOOD_INT = np.array([
    [162, 71, 71, 71, 71, 71, 217, 217, 71, 71, 71, 71, 71, 162],
    [90, 217, 217, 217, 217, 159, 217, 217, 159, 217, 217, 217, 217, 71],
    [90, 217, 217, 217, 217, 217, 159, 159, 217, 217, 217, 217, 217, 71],
    [90, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 71],
    [90, 159, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 159, 71],
    [90, 159, 90, 195, 90, 195, 195, 195, 195, 90, 195, 90, 159, 71],
    [90, 159, 90, 90, 195, 90, 159, 159, 90, 195, 90, 90, 159, 71],
    [90, 159, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 159, 71],
    [90, 159, 195, 159, 159, 195, 159, 195, 159, 159, 159, 195, 159, 71],
    [90, 159, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 159, 71],
    [162, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 162]
], dtype=float)

def _norm_pattern(arr_2d: np.ndarray) -> np.ndarray:
    flat = arr_2d.flatten().astype(float)
    s = float(flat.std())
    return (flat - float(flat.mean())) / s if s > 1e-5 else np.zeros_like(flat)

TPL_GRAY_NORM = _norm_pattern(_TPL_GRAY_INT)
TPL_WOOD_NORM = _norm_pattern(_TPL_WOOD_INT)


@dataclass
class POI:
    name: str                       # e.g. "STAIRS_DOWN", "FRONT_DOOR", "MOM", "PC", "SIGN"
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
    is_bag_menu: bool = False
    bag_items_empty: bool = False
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
        if is_battle:
            # Overworld guard: If there is no orange battle narrative banner, check if Red is centered in the overworld
            r_c = arr[:, :, 0].astype(int)
            g_c = arr[:, :, 1].astype(int)
            b_c = arr[:, :, 2].astype(int)
            orange_banner = float(np.mean(
                (r_c[235:310, 20:460] > 165) & (r_c[235:310, 20:460] < 215) &
                (g_c[235:310, 20:460] > 95) & (g_c[235:310, 20:460] < 140) &
                (b_c[235:310, 20:460] > 20) & (b_c[235:310, 20:460] < 55)
            ))
            if orange_banner < 0.40:
                player_check = self._detect_player(arr)
                if player_check.detected and player_check.tile_col == 7 and player_check.tile_row == 4:
                    is_battle = False
        is_trainer_card = self._is_trainer_card(arr) and not is_battle and not is_naming
        is_title_or_menu, title_desc, is_oak_intro = self._is_title_or_main_menu(arr)
        if is_battle or is_naming or is_trainer_card:
            is_title_or_menu = False
            is_oak_intro = False

        is_shop_menu = self._is_shop_menu(arr) and not is_battle and not is_naming and not is_title_or_menu
        is_bag_menu = self._is_bag_menu(arr) and not is_battle and not is_naming and not is_title_or_menu and not is_shop_menu
        is_start_menu = self._is_start_menu(arr) and not is_battle and not is_naming and not is_title_or_menu and not is_shop_menu and not is_bag_menu
        is_dialogue = (
            (self._is_dialogue_active(arr) or is_oak_intro)
            and not is_battle
            and not is_naming
            and not is_trainer_card
            and not is_shop_menu
            and not is_bag_menu
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
            is_bag_menu=is_bag_menu,
            is_start_menu=is_start_menu,
            is_dialogue=is_dialogue
        )

        # 3. Player localization (skip on non-map screens like Battle, Naming, Title, Trainer Card, Bag)
        if area_type in ("BATTLE", "NAMING_SCREEN", "TITLE_SCREEN", "TRAINER_CARD", "BAG_MENU"):
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
        elif is_bag_menu:
            area_type = "BAG_MENU"
            area_desc = "Player Bag Inventory Screen"
            bag_empty = self._is_bag_empty(arr)
            if bag_empty:
                notes.append("Player Bag is open, but this item pocket is completely EMPTY (0 items / only 'Cancel'). You have 0 usable items or Poké Balls here! Press B ('EXIT_ROOM') to close and return.")
            else:
                notes.append("Player Bag is open. Press B ('EXIT_ROOM') to close and return.")
            available_intents = ["EXIT_ROOM", "ADVANCE_DIALOGUE", "WAIT"]
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
                self._analyze_oaks_lab(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue, is_start_menu=is_start_menu)
            elif area_type == "POKECENTER":
                self._analyze_pokecenter(arr, collision_grid, player, pois, notes, available_intents)
            elif area_type == "POKEMART":
                self._analyze_pokemart(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue, is_shop_menu=is_shop_menu)
            elif area_type == "GATEHOUSE":
                self._analyze_gatehouse(arr, collision_grid, player, pois, notes, available_intents, is_dialogue=is_dialogue)
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
            is_bag_menu=is_bag_menu,
            bag_items_empty=(is_bag_menu and self._is_bag_empty(arr)),
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
        """Detects Pokémon Battle via Enemy/Player cream HUD cards or orange/dark battle banner."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            # Battle screen never has black room void border (< 20 RGB) across bottom edge
            bot_void = float(np.mean((r[280:, :] < 20) & (g[280:, :] < 20) & (b[280:, :] < 20)))
            if bot_void > 0.40:
                return False

            # Recharged Yellow / Gen 3 Battle HUD cream card fill: RGB ~ [248, 248, 216] (Daytime)
            cream_mask = (r > 230) & (g > 230) & (b >= 190) & (b <= 232) & ((r - b) >= 12) & ((g - b) >= 12)
            ehud_cream = float(np.mean(cream_mask[28:82, 24:212]))
            phud_cream = float(np.mean(cream_mask[148:212, 268:455]))
            if ehud_cream > 0.25 and phud_cream > 0.25:
                return True

            # Night Battle HUD cards (Recharged Yellow RTC night palette: Enemy card dark gray, Player card muted tan/olive)
            ehud_night = float(np.mean(
                (r[28:82, 24:212] > 20) & (r[28:82, 24:212] < 70) &
                (g[28:82, 24:212] > 20) & (g[28:82, 24:212] < 70) &
                (b[28:82, 24:212] > 20) & (b[28:82, 24:212] < 70)
            ))
            phud_night = float(np.mean(
                (r[148:212, 268:455] > 75) & (r[148:212, 268:455] < 145) &
                (g[148:212, 268:455] > 75) & (g[148:212, 268:455] < 145) &
                (b[148:212, 268:455] > 55) & (b[148:212, 268:455] < 125)
            ))
            if ehud_night > 0.15 and phud_night > 0.35:
                return True

            # Orange/ochre battle narrative banner at bottom (y=235..310, RGB ~ [189, 115, 33])
            orange_banner = float(np.mean(
                (r[235:310, 20:460] > 165) & (r[235:310, 20:460] < 215) &
                (g[235:310, 20:460] > 95) & (g[235:310, 20:460] < 140) &
                (b[235:310, 20:460] > 20) & (b[235:310, 20:460] < 55)
            ))
            mint_sky = float(np.mean((r[:100, :] > 200) & (g[:100, :] > 220) & (b[:100, :] > 205)))
            if orange_banner > 0.45 and (ehud_cream > 0.20 or phud_cream > 0.20 or mint_sky > 0.45):
                return True

            # Fallback for dark battle banner + green HP bar or confirmed HUD cards (requires player HUD card or HP bar so overworld forest isn't misdetected)
            bottom_box = arr[220:, :, :]
            is_dark_banner = np.mean(bottom_box[:, :, 0] < 60) > 0.6 and np.mean(bottom_box[:, :, 2] > 60) > 0.3
            hp_green = (g > 200) & (r < 150) & (b < 150)
            has_hp = np.sum(hp_green) > 50
            return bool(is_dark_banner and (has_hp or ehud_cream > 0.20 or (ehud_night > 0.15 and phud_night > 0.20)))
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

    def _is_bag_menu(self, arr: np.ndarray) -> bool:
        """Detects in-game Bag screen (backpack on left, item list on right, blue bottom banner)."""
        try:
            r = arr[:, :, 0].astype(int)
            g = arr[:, :, 1].astype(int)
            b = arr[:, :, 2].astype(int)
            item_list_white = float(np.mean((r[15:215, 170:470] > 220) & (g[15:215, 170:470] > 220) & (b[15:215, 170:470] > 160)))
            bot_blue = float(np.mean((r[225:315, 10:470] < 60) & (g[225:315, 10:470] > 80) & (b[225:315, 10:470] > 140)))
            backpack_bg_teal = float(np.mean((r[60:215, 10:160] < 120) & (g[60:215, 10:160] > 150) & (b[60:215, 10:160] > 140)))
            return bool(item_list_white > 0.60 and bot_blue > 0.50 and backpack_bg_teal > 0.35)
        except Exception:
            return False

    def _is_bag_empty(self, arr: np.ndarray) -> bool:
        """
        Detects whether the currently opened bag pocket is completely empty
        (Slot 1 is 'Cancel' or no item names exist in any slot).
        """
        try:
            # First slot text area: x in [195..280], y in [20..50]
            # Slots 2-4 text area: x in [195..400], y in [50..170]
            text_slots_other = np.sum((arr[50:170, 195:400, 0] <= 110) & (arr[50:170, 195:400, 1] <= 110) & (arr[50:170, 195:400, 2] <= 110))
            # If slots 2-4 have 0 text pixels, slot 1 has text ('Cancel'), this pocket has 0 usable items!
            return bool(text_slots_other < 25)
        except Exception:
            return False

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
        """Checks if a white bottom dialogue box with dark, blue, or red text is on screen."""
        try:
            box = arr[238:300, 35:445, :]
            r = box[:, :, 0]
            g = box[:, :, 1]
            b = box[:, :, 2]
            pure_white = (r >= 245) & (g >= 245) & (b >= 245)
            pw_left = float(np.mean(pure_white[:, :200]))
            pw_right = float(np.mean(pure_white[:, 200:]))
            text_px = float(np.mean((r < 140) | (g < 140) | (b < 140)))
            return bool(pw_left > 0.60 and pw_right > 0.60 and text_px >= 0.008)
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
        is_bag_menu: bool = False,
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
        if is_bag_menu:
            return "BAG_MENU", "Player Bag Inventory Screen"
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

        non_void_mid = (mr >= 25) | (mg >= 25) | (mb >= 25)
        tatami = float(np.mean(
            ((mr > 140) & (mg > 120) & (mb < 110) & (mr > mb + 35)) |
            ((mr > 80) & (mg > 70) & (mb < 65) & (mr > mb + 45) & (mg > mb + 40))
        ))
        green_rug = float(np.mean(
            ((mr < 140) & (mg > 140) & (mb < 150) & (mg > mr + 15)) |
            ((mr >= 45) & (mr <= 90) & (mg >= 100) & (mg <= 150) & (mb >= 45) & (mb <= 95) & (mg > mr + 30) & (mg > mb + 30))
        ))
        tatami_score = tatami + green_rug
        mart_blue = float(np.mean((mb > mr + 30) & (mb > mg + 10) & (mb > 140)))
        lab_white_mask = (mr > 180) & (mg > 180) & (mb > 180) & (np.abs(mr - mg) < 20) & (np.abs(mg - mb) < 25)
        lab_white = float(np.mean(lab_white_mask))
        lab_white_nv = float(np.sum(lab_white_mask) / max(1, np.sum(non_void_mid)))
        lab_dimmed = float(np.mean((mr > 55) & (mr < 90) & (np.abs(mr - mg) < 8) & (np.abs(mg - mb) < 8)))

        # 2. Poké Mart: Distinctive blue indoor floor tiles
        if mart_blue > 0.20 and tatami < 0.10:
            return "POKEMART", "Poké Mart - Item Shop"

        # 3. Oak's Research Lab: Bright white/grey diagonally-tiled floor + room void borders
        # Using lab_white_nv (fraction of non-void indoor pixels) ensures the South entrance of Oak's Lab
        # (where 45% of mid_floor is black void + bookshelves) is always recognized as OAKS_LAB!
        if (lab_white > 0.22 or lab_white_nv > 0.24 or (lab_dimmed > 0.20 and is_start_menu)) and tatami < 0.08 and total_void > 0.08:
            return "OAKS_LAB", "Pallet Town - Professor Oak's Research Lab"

        # 4. Poké Center: Healing counter at north with red panel (counter_red > 400), yellow checkered or pink floor, and left escalator
        counter_red = np.sum((r_all[:70, :] > 180) & (g_all[:70, :] < 100) & (b_all[:70, :] < 100))
        center_floor = (r_all > 220) & (g_all > 210) & (b_all > 210)
        center_yellow_grid = (r_all[60:240, :] > 200) & (g_all[60:240, :] > 190) & (b_all[60:240, :] > 130) & (b_all[60:240, :] < 190)
        red_counter_emblem = (r_all[60:140, :] > 190) & (g_all[60:140, :] < 80) & (b_all[60:140, :] < 80)
        if (counter_red > 400 or (np.mean(center_floor) > 0.25 and np.sum(red_counter_emblem) > 150)) and tatami < 0.08 and lab_white < 0.35 and (total_void > 0.05 or is_dialogue or north_void):
            return "POKECENTER", "Pokémon Center - Free Healing Station"

        # 4b. Gatehouse / Border Passage: Distinctive blue runner carpet (width ~3-4 tiles, height >= 6 tiles) across yellow checkerboard floor
        gatehouse_runner_cnt = int(np.sum((b_all > 130) & (b_all > r_all + 25) & (g_all > 105) & (g_all < 185)))
        floor_yellow_cnt = int(np.sum((r_all > 150) & (g_all > 140) & (r_all > b_all + 25)))
        if gatehouse_runner_cnt >= 5000 and floor_yellow_cnt >= 20000 and total_void > 0.05:
            return "GATEHOUSE", "Route 2 / Viridian Forest Gatehouse"

        # 5. House Indoor Rooms (Tatami wooden floor + green rug, Daytime + Nighttime): Distinguish 1F Living Room vs 2F Bedroom
        if tatami >= 0.12 or (tatami_score >= 0.18 and total_void >= 0.15):
            # Find the green floor rug (5th to 95th percentile bounding box, Daytime + Nighttime)
            rug_mask = (
                ((r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)) |
                ((r_all >= 45) & (r_all <= 90) & (g_all >= 100) & (g_all <= 150) & (b_all >= 45) & (b_all <= 95) & (g_all > r_all + 30) & (g_all > b_all + 30))
            )
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
                rug_blue_cnt = int(np.sum(
                    ((rb > rr + 35) & (rb > 175) & (rg > 130)) |
                    ((rb > rr + 30) & (rb > 125) & (rg > 105))
                ))

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
            if total_void > 0.15:
                if mart_blue > 0.12:
                    return "POKEMART", "Poké Mart - Item Shop"
                if lab_white_nv > 0.15:
                    return "OAKS_LAB", "Pallet Town - Professor Oak's Research Lab"
                if tatami_score > 0.08:
                    return "LIVING_ROOM_1F", "Pallet Town - Mom's Living Room (1F)"

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

        # 2. Nighttime / Dusk shader cap mask (search tight head window [128:154, 224:256] first so nighttime red doormat below y=155 never pulls cap center down)
        if len(valid_cap_x) < 12:
            night_cap = (
                (r_all >= 75) & (r_all <= 165) &
                (g_all <= 65) & (b_all <= 70) &
                (r_all >= g_all + 30) & (r_all >= b_all + 20)
            )
            for y_min, y_max, x_min, x_max in [(128, 154, 224, 256), (115, 180, 210, 270)]:
                ncy, ncx = np.where(night_cap[y_min:y_max, x_min:x_max])
                ncy += y_min
                ncx += x_min
                for y_val, x_val in zip(ncy, ncx):
                    patch = arr[min(h - 1, y_val + 4):min(h, y_val + 26), max(0, x_val - 10):min(w, x_val + 10)]
                    black_outline = (patch[:, :, 0] < 15) & (patch[:, :, 1] < 15) & (patch[:, :, 2] < 15)
                    if np.sum(black_outline) >= 10:
                        valid_cap_x.append(int(x_val))
                        valid_cap_y.append(int(y_val))
                if len(valid_cap_x) >= 12:
                    break

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

        # Determine facing direction by inspecting the cap front (y=144..151) and left/right head profile (y=144..158).
        # Previous skin-tone heuristic matched Red's brown backpack at y=162..164 when facing UP!
        if 232 <= px <= 248 and 136 <= py <= 150:
            patch_cap = arr[144:151, 234:246]
            patch_L = arr[144:158, 230:238]
            patch_R = arr[144:158, 242:250]
        else:
            patch_cap = arr[max(0, py + 2):min(h, py + 9), max(0, px - 5):min(w, px + 7)]
            patch_L = arr[max(0, py + 2):min(h, py + 16), max(0, px - 9):min(w, px - 1)]
            patch_R = arr[max(0, py + 2):min(h, py + 16), max(0, px + 3):min(w, px + 11)]

        cap_r = float(patch_cap[:, :, 0].mean()) if patch_cap.size > 0 else 0.0
        cap_g = float(patch_cap[:, :, 1].mean()) if patch_cap.size > 0 else 0.0
        lr_diff = (float(patch_L.mean()) - float(patch_R.mean())) if (patch_L.size > 0 and patch_R.size > 0) else 0.0

        if lr_diff > 12.0:
            facing = "LEFT"
        elif lr_diff < -12.0:
            facing = "RIGHT"
        elif cap_g > 75.0 and (cap_r - cap_g) < 75.0:
            facing = "DOWN"
        else:
            facing = "UP"

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

        # 1. Locate the central 5x4 green floor rug to anchor the entire room regardless of camera scroll (Day + Night)
        rug_mask = (
            ((r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)) |
            ((r_all >= 45) & (r_all <= 90) & (g_all >= 100) & (g_all <= 150) & (b_all >= 45) & (b_all <= 95) & (g_all > r_all + 30) & (g_all > b_all + 30))
        )
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
            ((r_all > 190) & (g_all > 110) & (g_all < 165) & (b_all > 70) & (b_all < 125)) |
            ((r_all > 115) & (g_all > 65) & (g_all < 110) & (b_all > 40) & (b_all < 85) & (r_all > g_all + 35))
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

        # 1. Locate the central 6x4 green floor rug to anchor the room (Daytime + Nighttime/Dusk)
        rug_mask = (
            ((r_all >= 80) & (r_all <= 130) & (g_all >= 175) & (g_all <= 220) & (b_all >= 100) & (b_all <= 145)) |
            ((r_all >= 45) & (r_all <= 90) & (g_all >= 100) & (g_all <= 150) & (b_all >= 45) & (b_all <= 95) & (g_all > r_all + 30) & (g_all > b_all + 30))
        )
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

        # 2. Front Door Exit Mat: Strictly below the green rug (Daytime + Nighttime/Dusk)
        rug_bot_y = (rug_r2 + 1) * TILE_SIZE + GRID_Y_OFFSET - 8
        max_y_search = 215 if is_dialogue else self.target_height
        red_mat = (
            (
                ((r_all > 165) & (r_all > g_all + 35) & (r_all > b_all + 35)) |
                ((r_all > 105) & (g_all < 65) & (b_all < 65) & (r_all > g_all + 50))
            ) &
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
        available_intents: List[str],
        is_dialogue: bool = False,
        is_start_menu: bool = False
    ):
        """Analyzes Professor Oak's Research Lab dynamically across any camera scroll position."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # Check if Follower Pikachu is standing on a cardinal tile adjacent to Red
        pikachu_following = False
        if player.detected:
            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nc, nr = player.tile_col + dc, player.tile_row + dr
                if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                    px1, py1, px2, py2 = tile_bounds(nc, nr)
                    pp = arr[max(0, py1):min(self.target_height, py2), max(0, px1):min(self.target_width, px2)]
                    if pp.size > 0:
                        ppr = pp[:, :, 0].astype(int)
                        ppg = pp[:, :, 1].astype(int)
                        ppb = pp[:, :, 2].astype(int)
                        pika_yellow = float(np.mean(
                            ((ppr > 195) & (ppg > 165) & (ppb < 95)) |
                            ((ppr > 105) & (ppg > 90) & (ppb < 50) & (ppr > ppb + 55))
                        ))
                        if pika_yellow > 0.14:
                            pikachu_following = True
                            grid[nr, nc] = CELL_WALKABLE

        # 1. Dynamically locate the olive-green Starter Table (3 tiles wide x 1 tile tall in Cols 7..11 of Lab)
        # Exclude Col 1 and Col 13 where green potted plants sit near the south entrance!
        green_tbl = (g_all > r_all + 25) & (g_all > b_all + 25) & (g_all > 130) & (r_all > 80)
        tbl_cols: List[int] = []
        tbl_rows: List[int] = []
        for r in range(0, GRID_ROWS - 1):
            for c in range(6, 12):
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = green_tbl[max(0, y1):min(self.target_height, y2), max(0, x1):min(self.target_width, x2)]
                if patch.size > 0 and float(np.mean(patch)) > 0.20:
                    tbl_cols.append(c)
                    tbl_rows.append(r)

        table_visible = (len(set(tbl_cols)) >= 2 and bool(tbl_rows))
        has_pokeball = True
        has_oak = True

        if table_visible:
            tc1, tc2 = min(tbl_cols), max(tbl_cols)
            tr1 = min(tbl_rows)

            ball_col = (tc1 + tc2) // 2
            ball_row = tr1
            oak_col = max(1, tc1 - 2)
            oak_row = tr1
            exit_col = max(1, tc1 - 2)
            exit_row = min(GRID_ROWS - 1, tr1 + 6)

            # Inspect whether the red/white Starter Poké Ball is on the table and whether Professor Oak is present
            if not is_dialogue and not is_start_menu:
                bx1, by1, bx2, by2 = tile_bounds(ball_col, ball_row)
                b_patch = arr[max(0, by1 + 2):min(self.target_height, by2 - 10), max(0, bx1 + 6):min(self.target_width, bx2 - 6)]
                red_ball_px = 0
                if b_patch.size > 0:
                    bpr = b_patch[:, :, 0].astype(int)
                    bpg = b_patch[:, :, 1].astype(int)
                    bpb = b_patch[:, :, 2].astype(int)
                    red_ball_px = int(np.sum((bpr > 170) & (bpr > bpg + 40) & (bpr > bpb + 40)))
                has_pokeball = (red_ball_px >= 15)

                ox1, oy1, ox2, oy2 = tile_bounds(oak_col, oak_row)
                o_patch = arr[max(0, oy1):min(self.target_height, oy2), max(0, ox1):min(self.target_width, ox2)]
                oak_std = float(o_patch.std()) if o_patch.size > 0 else 0.0
                has_oak = (oak_std >= 30.0) and not (player.detected and (player.tile_col, player.tile_row) == (oak_col, oak_row))

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

            # Professor Oak NPC tile (only mark as obstacle if Oak is present)
            if has_oak and 0 <= oak_row < GRID_ROWS and 0 <= oak_col < GRID_COLS:
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

            starter_interact = (ball_col, min(GRID_ROWS - 1, ball_row + 1)) if ball_row + 1 < GRID_ROWS else (ball_col, max(1, ball_row - 1))
            oak_interact = (oak_col, min(GRID_ROWS - 1, oak_row + 1)) if (not player.detected or player.tile_row >= oak_row) else (oak_col, max(1, oak_row - 1))
        else:
            # South entrance of Professor Oak's Lab (Starter Table & Oak are still off-screen to the NORTH up the central aisle!):
            # Keep the central aisle (Cols 6..8, Rows 0..4) completely walkable so Red walks straight UP!
            for r in (0, 1):
                for c in range(GRID_COLS):
                    if (c <= 5 or c >= 9) and grid[r, c] == CELL_WALKABLE:
                        grid[r, c] = CELL_OBSTACLE
            for r in range(0, 5):
                for c in (6, 7, 8):
                    if grid[r, c] != CELL_VOID:
                        grid[r, c] = CELL_WALKABLE

            if player.detected and 0 <= player.tile_row < GRID_ROWS and 0 <= player.tile_col < GRID_COLS:
                grid[player.tile_row, player.tile_col] = CELL_WALKABLE

            ball_col, ball_row = 7, 0
            starter_interact = (7, 0)
            oak_col, oak_row = 7, 0
            oak_interact = (7, 0)
            last_floor_row = player.tile_row if player.detected else 4
            for r in range(4, GRID_ROWS):
                if grid[r, 7] != CELL_VOID:
                    last_floor_row = r
                    grid[r, 7] = CELL_WALKABLE
            exit_col = 7
            exit_row = min(GRID_ROWS - 1, last_floor_row + 1)
            grid[exit_row, exit_col] = CELL_WARP

        # 3. Populate POIs
        bpx, bpy = tile_to_pixel_center(ball_col, ball_row)
        starter_poi = POI(
            name="STARTER_TABLE",
            category="OBJECT",
            tile_col=ball_col,
            tile_row=ball_row,
            pixel_x=bpx,
            pixel_y=bpy,
            description=(
                f"Starter Poké Ball on table (Approach at Col {starter_interact[0]}, Row {starter_interact[1]} and press A)"
                if has_pokeball else "Empty Starter Table (Starter Poké Ball is not on the table)"
            ),
            interaction_tile=starter_interact
        )

        opx, opy = tile_to_pixel_center(oak_col, oak_row)
        oak_poi = POI(
            name="PROFESSOR_OAK",
            category="NPC",
            tile_col=oak_col,
            tile_row=oak_row,
            pixel_x=opx,
            pixel_y=opy,
            description="Professor Oak",
            interaction_tile=oak_interact
        )

        epx, epy = tile_to_pixel_center(exit_col, exit_row)
        exit_poi = POI(
            name="LAB_EXIT",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=epx,
            pixel_y=epy,
            description=(
                "South exit doorway back to Pallet Town (Choose 'EXIT_ROOM' with target 'LAB_EXIT' to leave the Lab)"
                if (not has_pokeball or pikachu_following)
                else "South exit doorway back to Pallet Town (Do NOT exit before receiving starter Pokémon!)"
            ),
            interaction_tile=(exit_col, exit_row)
        )

        if (table_visible and not has_pokeball) or pikachu_following:
            pois.extend([exit_poi, oak_poi, starter_poi])
        else:
            pois.extend([starter_poi, oak_poi, exit_poi])

        notes.append("Red is inside Professor Oak's Research Lab.")
        if table_visible and not has_pokeball and not has_oak:
            notes.append(
                f"LAB_TABLE_EMPTY (OAK NOT HERE): Professor Oak and the Starter Poké Ball are NOT in the Lab yet! Choose 'EXIT_ROOM' with target 'LAB_EXIT' at Col {exit_col}, Row {exit_row} to leave the Lab and walk NORTH to the tall grass exit of Pallet Town to trigger Professor Oak!"
            )
        elif (table_visible and not has_pokeball and has_oak) or pikachu_following:
            notes.append(
                f"STARTER_ALREADY_CLAIMED (LAB_TABLE_EMPTY): Red has already received his Starter Pikachu! Choose 'EXIT_ROOM' with target 'LAB_EXIT' at Col {exit_col}, Row {exit_row} to walk SOUTH and exit Professor Oak's Lab into Pallet Town!"
            )
        else:
            notes.extend([
                f"STARTER POKÉMON: Starter Poké Ball table is NORTH at Col {ball_col}, Row {ball_row} (Target 'STARTER_TABLE' at Col {starter_interact[0]}, Row {starter_interact[1]}).",
                f"NPC: Professor Oak is NORTH at Col {oak_col}, Row {oak_row}.",
                f"EXIT: South doorway to Pallet Town is at Col {exit_col}, Row {exit_row} (Do NOT step DOWN onto exit unless leaving!)."
            ])

        available_intents.extend(["GO_TO_POI", "TALK_TO_NPC", "EXPLORE_NORTH", "EXIT_ROOM", "WAIT"])

    def _analyze_pokecenter(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str]
    ):
        """Analyzes Pokémon Center interior dynamically across camera scroll positions."""
        r_all = arr[:, :, 0].astype(int)
        g_all = arr[:, :, 1].astype(int)
        b_all = arr[:, :, 2].astype(int)

        # 1. Detect red counter position at north
        counter_rows = [
            r for r in range(4)
            if np.sum((r_all[r*32:(r+1)*32, :] > 180) & (g_all[r*32:(r+1)*32, :] < 100) & (b_all[r*32:(r+1)*32, :] < 100)) > 200
        ]
        counter_r = min(counter_rows) if counter_rows else 0
        nurse_joy_row = counter_r
        nurse_joy_interact_row = counter_r + 1

        # 2. Detect south exit mat (red welcome mat before room void)
        exit_col, exit_row = 7, 5
        best_mat_score = 0.0
        for r in range(5, GRID_ROWS):
            for c in range(5, 10):
                if player.detected and (c, r) == (player.tile_col, player.tile_row):
                    continue
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = arr[y1:y2, x1:x2]
                pr = patch[:, :, 0].astype(int)
                pg = patch[:, :, 1].astype(int)
                pb = patch[:, :, 2].astype(int)
                is_red = float(np.mean((pr > 150) & (pr > pg + 40) & (pr > pb + 40)))
                if is_red > best_mat_score and is_red > 0.12:
                    best_mat_score = is_red
                    exit_col, exit_row = c, r

        # 3. Mark grid: counter obstacles, void below exit, left escalator, right PC table
        for c in range(4, 11):
            if 0 <= nurse_joy_row < GRID_ROWS:
                grid[nurse_joy_row, c] = CELL_COUNTER

        # Escalator on the west wall (Cols 0..1, Rows counter_r+1..counter_r+3)
        for r in range(nurse_joy_row + 1, min(GRID_ROWS, nurse_joy_row + 4)):
            for c in (0, 1):
                if grid[r, c] == CELL_WALKABLE:
                    grid[r, c] = CELL_OBSTACLE

        # PC terminal on east wall at (12, nurse_joy_row + 2)
        pc_col = 12
        pc_row = min(GRID_ROWS - 1, nurse_joy_row + 2)
        grid[pc_row, pc_col] = CELL_OBSTACLE
        grid[pc_row, pc_col - 1] = CELL_OBSTACLE

        # Void below exit
        for r in range(exit_row + 1, GRID_ROWS):
            for c in range(GRID_COLS):
                grid[r, c] = CELL_VOID

        # Keep player's tile walkable
        if player.detected and 0 <= player.tile_row < GRID_ROWS and 0 <= player.tile_col < GRID_COLS:
            grid[player.tile_row, player.tile_col] = CELL_WALKABLE

        # Mark exit warp
        grid[exit_row, exit_col] = CELL_WARP

        # Nurse Joy at counter top-center
        jpx, jpy = tile_to_pixel_center(7, nurse_joy_row)
        pois.append(POI(
            name="NURSE_JOY",
            category="NPC",
            tile_col=7,
            tile_row=nurse_joy_row,
            pixel_x=jpx,
            pixel_y=jpy,
            description="Nurse Joy (Free party healing at counter)",
            interaction_tile=(7, nurse_joy_interact_row)
        ))

        # PC terminal on right wall
        pcpx, pcpy = tile_to_pixel_center(pc_col, pc_row)
        pois.append(POI(
            name="PC",
            category="OBJECT",
            tile_col=pc_col,
            tile_row=pc_row,
            pixel_x=pcpx,
            pixel_y=pcpy,
            description="Pokémon Storage System PC",
            interaction_tile=(pc_col, pc_row + 1 if pc_row + 1 < exit_row else pc_row - 1)
        ))

        # Exit door at south center
        epx, epy = tile_to_pixel_center(exit_col, exit_row)
        exit_interact = (exit_col, min(GRID_ROWS - 1, exit_row + 1)) if (player.detected and (player.tile_col, player.tile_row) == (exit_col, exit_row) and exit_row + 1 < GRID_ROWS) else (exit_col, exit_row)
        pois.append(POI(
            name="CENTER_EXIT",
            category="WARP",
            tile_col=exit_col,
            tile_row=exit_row,
            pixel_x=epx,
            pixel_y=epy,
            description="Exit door back to town",
            interaction_tile=exit_interact
        ))

        notes.extend([
            "Red is inside the Pokémon Center.",
            f"HEALING: Walk to Nurse Joy counter at Col 7, Row {nurse_joy_interact_row} to heal your party.",
            f"EXIT: South door at Col {exit_col}, Row {exit_row}."
        ])

        available_intents.extend(["TALK_TO_NPC", "EXIT_ROOM", "GO_TO_POI", "WAIT"])

    def _analyze_gatehouse(
        self,
        arr: np.ndarray,
        grid: np.ndarray,
        player: PlayerState,
        pois: List[POI],
        notes: List[str],
        available_intents: List[str],
        is_dialogue: bool = False
    ):
        """Analyzes Route / Forest Gatehouse interior: north doorway, south doorway, guard NPC, tables/chairs."""
        h, w, _ = arr.shape
        self._populate_void_cells(arr, grid)

        # Mark all void cells and border walls
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if grid[r, c] == CELL_VOID:
                    continue
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = arr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if patch.size > 0:
                    pr, pg, pb = patch[:, :, 0].astype(int), patch[:, :, 1].astype(int), patch[:, :, 2].astype(int)
                    # Tables & chairs in gatehouse center: white/metallic desk surfaces
                    is_table = float(np.mean((pr > 210) & (pg > 210) & (pb > 210))) > 0.20
                    is_chair = float(np.mean((pb > 170) & (pb > pr + 40) & (pr < 130))) > 0.15
                    if is_table or is_chair:
                        if not (player.detected and (c, r) == (player.tile_col, player.tile_row)):
                            grid[r, c] = CELL_OBSTACLE
                    else:
                        grid[r, c] = CELL_WALKABLE

        # Guard / Border Patrol Officer NPC (e.g., Col 4, Row 3)
        guard_col, guard_row = 4, 3
        if player.detected and (guard_col, guard_row) != (player.tile_col, player.tile_row):
            grid[guard_row, guard_col] = CELL_OBSTACLE
        gpx, gpy = tile_to_pixel_center(guard_col, guard_row)
        pois.append(POI(
            name="GUARD",
            category="NPC",
            tile_col=guard_col,
            tile_row=guard_row,
            pixel_x=gpx,
            pixel_y=gpy,
            description="Gatehouse Guard",
            interaction_tile=(guard_col, guard_row + 1)
        ))

        # 1. Locate North Doorway opening dynamically (distinctive dark portal adjacent to walkable floor)
        north_door_col = 7
        north_door_row = 3
        doorway_found = False
        for r in range(GRID_ROWS - 1):
            for c in range(2, 13):
                if grid[r, c] == CELL_VOID:
                    continue
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = arr[y1:y2, x1:x2]
                if patch.size > 0:
                    dark_ratio = float(np.mean((patch[:, :, 0] < 125) & (patch[:, :, 1] < 125) & (patch[:, :, 2] < 135)))
                    # Walkable floor or player directly below this tile indicates the doorway opening
                    is_above_floor = (grid[r + 1, c] == CELL_WALKABLE) or (player.detected and (c, r + 1) == (player.tile_col, player.tile_row))
                    if dark_ratio > 0.45 and is_above_floor:
                        north_door_col = c
                        north_door_row = r
                        doorway_found = True
                        break
            if doorway_found:
                break

        grid[north_door_row, north_door_col] = CELL_WARP
        if north_door_col + 1 < GRID_COLS and grid[north_door_row, north_door_col + 1] != CELL_VOID:
            grid[north_door_row, north_door_col + 1] = CELL_WARP
        if north_door_col - 1 >= 0 and grid[north_door_row, north_door_col - 1] != CELL_VOID:
            grid[north_door_row, north_door_col - 1] = CELL_WARP

        # Wall row directly above floor / adjacent to doorway
        wall_row = max(0, north_door_row - 1)
        for c in range(GRID_COLS):
            if grid[wall_row, c] != CELL_VOID and grid[wall_row, c] != CELL_WARP:
                grid[wall_row, c] = CELL_WALL

        npx, npy = tile_to_pixel_center(north_door_col, north_door_row)
        pois.insert(0, POI(
            name="NORTH_GATE",
            category="WARP",
            tile_col=north_door_col,
            tile_row=north_door_row,
            pixel_x=npx,
            pixel_y=npy,
            description="North Exit doorway leading into Viridian Forest (Pewter City direction)",
            interaction_tile=(north_door_col, north_door_row)
        ))

        # 2. South Exit doorway / mat
        south_door_row = 8
        south_door_col = 1
        red_mat_found = False
        for r in range(5, GRID_ROWS):
            for c in range(GRID_COLS):
                x1, y1, x2, y2 = tile_bounds(c, r)
                patch = arr[y1:y2, x1:x2]
                if patch.size > 0:
                    is_red_mat = float(np.mean((patch[:, :, 0] > 160) & (patch[:, :, 1] < 110) & (patch[:, :, 2] < 110))) > 0.25
                    if is_red_mat:
                        south_door_col = c
                        south_door_row = r
                        red_mat_found = True
                        break
            if red_mat_found:
                break

        if red_mat_found:
            grid[south_door_row, south_door_col] = CELL_WARP
            spx, spy = tile_to_pixel_center(south_door_col, south_door_row)
            pois.append(POI(
                name="SOUTH_GATE",
                category="WARP",
                tile_col=south_door_col,
                tile_row=south_door_row,
                pixel_x=spx,
                pixel_y=spy,
                description="South Exit doorway leading back to Route 2 / Viridian City",
                interaction_tile=(south_door_col, south_door_row)
            ))

        notes.extend([
            "Red is inside the Route 2 / Viridian Forest Gatehouse.",
            f"To travel NORTH (toward Pewter City): Step NORTH through NORTH_GATE doorway at Col {north_door_col}, Row {north_door_row}.",
            f"To travel SOUTH (toward Viridian City): Step SOUTH through SOUTH_GATE doorway at Col {south_door_col}, Row {south_door_row}."
        ])

        available_intents.extend(["EXIT_ROOM", "GO_TO_POI", "EXPLORE_NORTH", "EXPLORE_SOUTH", "TALK_TO_NPC", "WAIT"])

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
        available_intents: List[str],
        is_dialogue: bool = False
    ):
        """Analyzes outdoor Overworld (Pallet Town, Route 1, Viridian City): maps trees, ledges, fences, tall grass, buildings, and doors."""
        h, w, _ = arr.shape
        tree_mask_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        pikachu_following = False

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
                            ((fpr > 95) & (fpg > 80) & (fpb < 60) & (fpr > fpb + 40) & (fpg > fpb + 25))
                        ))
                        ground_bg = float(np.mean(fpb >= fpr - 10))
                        if pika_yellow > 0.08 and ground_bg >= 0.20 and int(full_patch.min()) <= 24:
                            pikachu_following = True
                            grid[r, c] = CELL_WALKABLE
                            continue

                # For the tile directly NORTH of Red, inspect upper 24px (excluding bottom 8px where Red's cap pokes up)
                # so brown ledges or obstacles on Row 3 are still detected!
                if player.detected and c == player.tile_col and r == player.tile_row - 1:
                    y2_c = min(y2_c, y1_c + 24)

                if y2_c <= y1_c or x2_c <= x1_c:
                    continue

                patch = arr[y1_c:y2_c, x1_c:x2_c]
                std_val = float(patch.std(axis=(0, 1)).mean())
                pr = patch[:, :, 0].astype(int)
                pg = patch[:, :, 1].astype(int)
                pb = patch[:, :, 2].astype(int)

                # Check if walkable Tall Wild Grass (green grass clump on mint-cyan lawn background, Daytime + Nighttime):
                # Every pixel is green (all_green >= 0.72), top 4px show lawn background (top4_lawn >= 0.60), and 25%-55% of tile is lawn backdrop
                all_green = float(np.mean(pg > pr + 15))
                lawn_backdrop = float(np.mean(
                    ((pb > pr + 12) & (pg > pr + 20) & (pb >= 75)) |
                    ((pb > pr + 10) & (pg > pr + 15) & (pb >= 30) & (pg < 100))
                ))
                top4 = patch[:min(4, patch.shape[0]), :]
                top4_lawn = float(np.mean(
                    ((top4[:, :, 2].astype(int) > top4[:, :, 0].astype(int) + 12) &
                     (top4[:, :, 1].astype(int) > top4[:, :, 0].astype(int) + 20) &
                     (top4[:, :, 2].astype(int) >= 75)) |
                    ((top4[:, :, 2].astype(int) > top4[:, :, 0].astype(int) + 10) &
                     (top4[:, :, 1].astype(int) > top4[:, :, 0].astype(int) + 15) &
                     (top4[:, :, 2].astype(int) >= 30) & (top4[:, :, 1].astype(int) < 100))
                )) if top4.size > 0 else 0.0

                is_tall_grass = (
                    all_green >= 0.72 and 0.26 <= lawn_backdrop <= 0.55 and top4_lawn >= 0.60
                )

                # Check if Pine / Forest Tree tile (impassable canopy with low lawn_backdrop)
                tree_green = float(np.mean((pg > pr + 18) & (pg > pb + 10)))
                if not is_tall_grass and tree_green > 0.35 and std_val >= 16.5 and lawn_backdrop < 0.25:
                    tree_mask_grid[r, c] = True

                # A) Walkable flat ground / dirt road / short lawn grass / cobblestone road border (min_channel >= 60, low color saturation)
                roof_px = float(np.mean((pr > pg + 25) & (pr > pb + 15)))
                water_px = float(np.mean((pb > pr + 40) & (pb > pg + 20) & (pb >= 135)))
                fence_px = float(np.mean((pr > pg + 15) & (pr > pb + 15) & (pr > 75) & (pr < 165)))
                min_channel = int(patch.min())
                gray_road = (
                    min_channel >= 60
                    and std_val < 38.0
                    and float(np.mean((np.abs(pr - pg) < 32) & (np.abs(pg - pb) < 28))) > 0.65
                )
                is_flat_ground = (
                    (std_val < 16.5 or (std_val < 26.0 and min_channel >= 50 and all_green > 0.25) or gray_road)
                    and roof_px < 0.35
                    and water_px < 0.25
                    and fence_px < 0.20
                )

                # B) Walkable flowers (lawn grass background + bright red flower petals, never brown ledges)
                lawn_bg = float(np.mean((pg > pr + 18) & (pg > 70)))
                red_petal = float(np.mean((pr > pg + 50) & (pr > pb + 40) & ((pr > 175) | (pg < 55))))
                is_flowers = (lawn_bg >= 0.35 and red_petal >= 0.08)

                if is_flat_ground or is_flowers or is_tall_grass:
                    grid[r, c] = CELL_WALKABLE
                else:
                    grid[r, c] = CELL_OBSTACLE

        # 1b. Top-left white area-name popup banner (y=6..38, x=12..240, e.g. 'ROUTE 1' or 'VIRIDIAN CITY'):
        # Inherit walkability from Row 1 for Cols 0..7 so the popup sign doesn't block northern pathfinding!
        banner_patch = arr[6:38, 12:240]
        if banner_patch.size > 0:
            banner_white = float(np.mean(
                (banner_patch[:, :, 0] > 235) & (banner_patch[:, :, 1] > 235) & (banner_patch[:, :, 2] > 235)
            ))
            if banner_white > 0.50:
                for c in range(0, 8):
                    grid[0, c] = grid[1, c]
                    tree_mask_grid[0, c] = tree_mask_grid[1, c]
                notes.append("AREA TRANSITION BANNER visible at top-left of screen.")

        if pikachu_following:
            notes.append("PIKACHU_FOLLOWING: Starter Pikachu is following Red!")

        # 1c. Pitched Roof Apex Passability:
        # In GBA Pokémon games, the 1-tile triangular peak at the top center of a house roof
        # (flanked by walkable road on both left and right, with the 3-tile building roof directly below)
        # is an overhead-layer sprite that Red can walk behind!
        for r in range(0, GRID_ROWS - 1):
            for c in range(1, GRID_COLS - 1):
                if (
                    grid[r, c] == CELL_OBSTACLE
                    and not tree_mask_grid[r, c]
                    and grid[r, c - 1] == CELL_WALKABLE
                    and grid[r, c + 1] == CELL_WALKABLE
                    and grid[r + 1, c - 1] == CELL_OBSTACLE
                    and grid[r + 1, c] == CELL_OBSTACLE
                    and grid[r + 1, c + 1] == CELL_OBSTACLE
                ):
                    grid[r, c] = CELL_WALKABLE

        # 2. Detect Building Front Walls & Doors (at least 2 tiles tall, architectural non-foliage structure, with walkable path below)
        found_target_warp = False
        just_exited_house = False
        max_wall_row = 6 if is_dialogue else (GRID_ROWS - 1)

        for r in range(1, max_wall_row):
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

                    # Check if this row is the bottom front wall of a building (walkable tiles directly below it, or row 5 above dialogue box)
                    walkable_below = [
                        col for col in range(c_start, c_end + 1)
                        if grid[r + 1, col] == CELL_WALKABLE
                    ]
                    if width >= 3 and (len(walkable_below) >= 1 or (is_dialogue and r == 5)):
                        # Verify this 2-row structure is architectural (roof + wall) and not Route 1 foliage/ledges or water pond
                        bx1, by1, _, _ = tile_bounds(c_start, r - 1)
                        _, _, bx2, by2 = tile_bounds(c_end, r)
                        b_patch = arr[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                        if b_patch.size > 0:
                            br = b_patch[:, :, 0].astype(int)
                            bg = b_patch[:, :, 1].astype(int)
                            bb = b_patch[:, :, 2].astype(int)
                            foliage_ratio = float(np.mean((bg > br + 20) & (bg > bb + 10)))
                            water_pond_ratio = float(np.mean((bb == 222) & (bg == 165) & (br == 99)))
                            if foliage_ratio > 0.25 or water_pond_ratio > 0.40:
                                continue

                        # If clipped at right screen edge and narrow, door is still off-screen to the East
                        if c_end == GRID_COLS - 1 and width < 4:
                            continue

                        best_door_col = None
                        best_score = -1.0
                        best_door_kind = "RED"
                        # Include c_end when building is clipped at right screen edge (e.g., Oak's Lab door at Col 14)
                        search_end = (c_end + 1) if c_end == GRID_COLS - 1 else c_end
                        for col in range(c_start + 1, search_end):
                            if grid[r + 1, col] != CELL_WALKABLE and not (is_dialogue and r == 5):
                                continue
                            x1, y1, x2, y2 = tile_bounds(col, r)
                            dp = arr[max(0, y1 + 6):min(h, y1 + 28), max(0, x1 + 4):min(w, x1 + 28)]
                            if dp.size == 0 or float(dp.std()) < 22.0 or float(dp.std(axis=(0, 1)).max()) < 5.0:
                                continue
                            dr = dp[:, :, 0].astype(int)
                            dg = dp[:, :, 1].astype(int)
                            db = dp[:, :, 2].astype(int)
                            # 1) Red/Orange/Crimson wooden house door (Daytime + Nighttime)
                            red_door = float(np.mean(
                                ((dr > dg + 42) & (dr > db + 35) & (dg < 125)) |
                                ((dr > dg + 18) & (dr > db + 10) & (dr > 55) & (dg < 70))
                            ))
                            # 2) Green/Teal door (Professor Oak's Lab & Rival's House, Daytime + Nighttime)
                            teal_door = float(np.mean(
                                ((dg > dr + 18) & (dg > 95)) |
                                ((dg > dr + 18) & (db > dr + 18) & (dg > 45) & (dr < 48))
                            ))
                            # 3) Blue/Cyan sliding glass door (Poké Center & Poké Mart, Daytime + Nighttime)
                            blue_door = float(np.mean(
                                ((db > dr + 25) & (dg > dr + 12) & (db > 120)) |
                                ((db > dr + 45) & (db > dg + 25) & (db > 95) & (dr < 55))
                            ))
                            door_score = max(red_door, teal_door, blue_door)
                            if door_score > best_score:
                                best_score = door_score
                                best_door_col = col
                                if blue_door >= red_door and blue_door >= teal_door:
                                    best_door_kind = "BLUE"
                                elif teal_door >= red_door:
                                    best_door_kind = "TEAL"
                                else:
                                    best_door_kind = "RED"

                        # Also check for an open doorway mid-animation (e.g., Oak's Lab sliding door open into dark gray doorway)
                        if best_score < 0.45 and width >= 5:
                            mid_col = (c_start + c_end) // 2
                            if grid[r + 1, mid_col] == CELL_WALKABLE:
                                x1, y1, x2, y2 = tile_bounds(mid_col, r)
                                dp = arr[max(0, y1 + 6):min(h, y1 + 28), max(0, x1 + 4):min(w, x1 + 28)]
                                wx1, wy1, wx2, wy2 = tile_bounds(max(0, mid_col - 1), r)
                                wp = arr[max(0, wy1):min(h, wy2), max(0, wx1):min(w, wx2)]
                                rx1, ry1, rx2, ry2 = tile_bounds(mid_col, max(0, r - 1))
                                rp = arr[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)]
                                if dp.size > 0 and wp.size > 0 and rp.size > 0:
                                    dp_mean = float(dp.mean())
                                    wp_mean = float(wp.mean())
                                    roof_r = float(rp[:, :, 0].mean())
                                    roof_g = float(rp[:, :, 1].mean())
                                    if dp_mean < 125.0 and wp_mean > dp_mean + 35.0 and roof_r < roof_g + 25.0:
                                        best_door_col = mid_col
                                        best_score = 0.50
                                        best_door_kind = "TEAL"

                        # Check if this building is Professor Oak's Lab (teal/green or open lab door without Poké Mart blue roof)
                        is_oaks_lab_building = False
                        if best_door_col is not None and best_score >= 0.25 and best_door_kind == "TEAL":
                            rx1, ry1, rx2, ry2 = tile_bounds(best_door_col, max(0, r - 1))
                            roof_patch = arr[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)]
                            roof_r = float(roof_patch[:, :, 0].mean()) if roof_patch.size > 0 else 0.0
                            roof_g = float(roof_patch[:, :, 1].mean()) if roof_patch.size > 0 else 0.0
                            roof_b = float(roof_patch[:, :, 2].mean()) if roof_patch.size > 0 else 0.0
                            if roof_g > roof_r + 2.0 and not (roof_b > roof_r + 12.0):
                                # Horizontal fence over green lawn grass (not a building roof)
                                best_door_col = None
                                best_score = -1.0
                            elif not (roof_b > roof_r + 12) and not (is_dialogue and r == 2):
                                is_oaks_lab_building = True

                        # Check if Red is standing right in front of this building having just exited it
                        # Never suppress Professor Oak's Lab as EXITED_HOUSE_DOOR so Red can always enter Oak's Lab!
                        is_exited_building = (
                            player.detected
                            and r == player.tile_row - 1
                            and c_start <= player.tile_col <= c_end
                            and not is_oaks_lab_building
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

                            # Determine specific building door type from door, roof, and wall color
                            rx1, ry1, rx2, ry2 = tile_bounds(best_door_col, max(0, r - 1))
                            roof_patch = arr[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)]
                            roof_r = float(roof_patch[:, :, 0].mean()) if roof_patch.size > 0 else 0.0
                            roof_b = float(roof_patch[:, :, 2].mean()) if roof_patch.size > 0 else 0.0

                            if best_door_kind in ("BLUE", "TEAL") and roof_b > roof_r + 12:
                                door_name = "POKEMART_DOOR"
                                door_desc = "Poké Mart Entrance Door (Walk onto this doorway to enter)"
                            elif best_door_kind == "BLUE" and roof_r > roof_b + 15:
                                door_name = "POKECENTER_DOOR"
                                door_desc = "Pokémon Center Entrance Door (Walk onto this doorway to enter)"
                            elif is_oaks_lab_building:
                                door_name = "OAKS_LAB"
                                door_desc = "Professor Oak's Lab Entrance Door (Walk onto this doorway to enter)"
                            else:
                                door_name = "BUILDING_DOOR"
                                door_desc = "Building Entrance Door (Walk onto this doorway to enter)"

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

        # If both West and East house doors in Pallet Town are visible, label West as Mom's House and East as Rival's House
        house_doors = [p for p in pois if p.name == "BUILDING_DOOR"]
        if len(house_doors) >= 2:
            house_doors_sorted = sorted(house_doors, key=lambda p: p.tile_col)
            west_door = house_doors_sorted[0]
            east_door = house_doors_sorted[-1]
            if west_door.tile_col <= 6 and east_door.tile_col >= 9:
                west_door.description = "Red's / Mom's House Entrance Door (West house in Pallet Town — Walk onto this doorway to enter Mom's House)"
                east_door.description = "Rival's / Daisy's House Entrance Door (East house in Pallet Town)"
                # Ensure West door (Mom's House) is listed before East door
                pois.remove(west_door)
                pois.insert(0, west_door)

        # 2b. Check top 16px strip (y = 0..16, above Row 0) for Professor Oak's Lab front wall & door
        # when Red has walked 1 row South of Oak's Lab so the doorway is at the very top edge of the screen.
        if not found_target_warp and not is_dialogue:
            top_wall_cols = [
                col for col in range(GRID_COLS)
                if float(arr[0:16, col * TILE_SIZE:(col + 1) * TILE_SIZE].std()) >= 24.0
                and grid[0, col] == CELL_WALKABLE
            ]
            tc = 0
            while tc < GRID_COLS and not found_target_warp:
                if tc in top_wall_cols:
                    cs = tc
                    while tc < GRID_COLS and tc in top_wall_cols:
                        tc += 1
                    ce = tc - 1
                    if ce - cs + 1 >= 4:
                        for col in range(cs + 1, ce):
                            dp = arr[0:16, col * TILE_SIZE + 4:(col + 1) * TILE_SIZE - 4]
                            if dp.size == 0 or float(dp.std()) < 22.0:
                                continue
                            dr = dp[:, :, 0].astype(int)
                            dg = dp[:, :, 1].astype(int)
                            db = dp[:, :, 2].astype(int)
                            teal_top = float(np.mean(
                                ((dg > dr + 18) & (dg > 95)) |
                                ((dg > dr + 18) & (db > dr + 18) & (dg > 45) & (dr < 48))
                            ))
                            lp = arr[0:16, (col - 1) * TILE_SIZE:col * TILE_SIZE]
                            rp = arr[0:16, (col + 1) * TILE_SIZE:(col + 2) * TILE_SIZE]
                            lr_c, lg_c, lb_c = lp[:, :, 0].astype(int), lp[:, :, 1].astype(int), lp[:, :, 2].astype(int)
                            rr_c, rg_c, rb_c = rp[:, :, 0].astype(int), rp[:, :, 1].astype(int), rp[:, :, 2].astype(int)
                            teal_l = float(np.mean(
                                ((lg_c > lr_c + 18) & (lg_c > 95)) |
                                ((lg_c > lr_c + 18) & (lb_c > lr_c + 18) & (lg_c > 45) & (lr_c < 48))
                            ))
                            teal_r = float(np.mean(
                                ((rg_c > rr_c + 18) & (rg_c > 95)) |
                                ((rg_c > rr_c + 18) & (rb_c > rr_c + 18) & (rg_c > 45) & (rr_c < 48))
                            ))
                            if (
                                teal_top >= 0.30
                                and teal_l < 0.05
                                and teal_r < 0.05
                                and float(lp.std()) >= 28.0
                                and float(rp.std()) >= 28.0
                            ):
                                grid[0, col] = CELL_WARP
                                dpx, dpy = tile_to_pixel_center(col, 0)
                                pois.insert(0, POI(
                                    name="OAKS_LAB",
                                    category="WARP",
                                    tile_col=col,
                                    tile_row=0,
                                    pixel_x=dpx,
                                    pixel_y=dpy,
                                    description=f"Professor Oak's Lab Entrance Door at top edge (Walk to ({col}, 0) and step UP to enter)",
                                    interaction_tile=(col, 0)
                                ))
                                found_target_warp = True
                                notes.append(
                                    f"Professor Oak's Lab entrance door is visible at the top edge ({col}, 0). Choose 'ENTER_BUILDING' with target 'OAKS_LAB'!"
                                )
                                break
                else:
                    tc += 1

        # 2c. South Pallet Town Pond & Fence Landmark:
        # If Red overshot South of Professor Oak's Lab (standing beside the Southwest Pallet Town Pond at Cols 1..6, Rows 2..9
        # with the Lab fence across Cols 8..13 on Rows 1..3 and open path at Row 0, Cols 8..13), place OAKS_LAB at (12, 0)
        # so Red routes NORTH around the fence directly back to Professor Oak's Lab!
        if not found_target_warp and not is_dialogue:
            water_count = 0
            for wr in range(2, GRID_ROWS):
                for wc in range(1, 7):
                    wx1, wy1, wx2, wy2 = tile_bounds(wc, wr)
                    wp = arr[max(0, wy1 + 4):min(h, wy2 - 4), max(0, wx1 + 4):min(w, wx2 - 4)]
                    if wp.size > 0:
                        wr_c = wp[:, :, 0].astype(int)
                        wg_c = wp[:, :, 1].astype(int)
                        wb_c = wp[:, :, 2].astype(int)
                        w_px = float(np.mean(
                            ((wb_c > wr_c + 35) & (wb_c > wg_c + 10) & (wr_c < 95)) |
                            ((wb_c > wr_c + 55) & (wb_c > wg_c + 40) & (wb_c > 95) & (wr_c < 55))
                        ))
                        if w_px > 0.45:
                            water_count += 1
            fence_east = any(
                sum(1 for fc in range(8, 14) if grid[fr, fc] != CELL_WALKABLE) >= 4
                for fr in range(1, 4)
            )
            walkable_north_east = (
                sum(1 for fc in range(8, 14) if grid[0, fc] == CELL_WALKABLE) >= 4
                and grid[0, 12] == CELL_WALKABLE
            )
            if water_count >= 8 and fence_east and walkable_north_east:
                grid[0, 12] = CELL_WARP
                dpx, dpy = tile_to_pixel_center(12, 0)
                pois.insert(0, POI(
                    name="OAKS_LAB",
                    category="WARP",
                    tile_col=12,
                    tile_row=0,
                    pixel_x=dpx,
                    pixel_y=dpy,
                    description="Professor Oak's Lab Entrance (North/Northeast behind the fence at (12, 0))",
                    interaction_tile=(12, 0)
                ))
                found_target_warp = True
                notes.append(
                    "OVERSHOT OAK'S LAB: Red walked too far South past Professor Oak's Lab to the Pallet Town Pond! Oak's Lab is NORTH/NORTHEAST behind the fence at (12, 0). Choose 'ENTER_BUILDING' with target 'OAKS_LAB' or 'EXPLORE_NORTH'!"
                )

        # 2c-2. Route 2 / Viridian Forest Gatehouse Entrance:
        # Detect gray stone steps and doorway leading North into the Gatehouse (e.g., Cols 1..4 at Row 0)
        if not found_target_warp and not is_dialogue:
            step_cols = []
            for c in range(GRID_COLS):
                sp = arr[0:32, c * TILE_SIZE:(c + 1) * TILE_SIZE]
                if sp.size > 0:
                    chan_std = float(np.std(sp, axis=2).mean())
                    mean_val = float(sp.mean())
                    if chan_std < 20.0 and mean_val > 150.0:
                        step_cols.append(c)
            if len(step_cols) >= 3 and 0 in step_cols or 1 in step_cols:
                # Doorway is at the center of the steps (e.g., col 2 or 3)
                door_c = (step_cols[0] + step_cols[-1]) // 2
                grid[0, door_c] = CELL_WARP
                # Mark neighboring step tiles as walkable or warp so pathfinder can easily step onto the stairs
                for sc in step_cols:
                    if grid[0, sc] == CELL_OBSTACLE:
                        grid[0, sc] = CELL_WALKABLE
                grid[0, door_c] = CELL_WARP
                dpx, dpy = tile_to_pixel_center(door_c, 0)
                pois.insert(0, POI(
                    name="GATEHOUSE_DOOR",
                    category="WARP",
                    tile_col=door_c,
                    tile_row=0,
                    pixel_x=dpx,
                    pixel_y=dpy,
                    description=f"Route 2 Gatehouse / Viridian Forest Entrance Door (Walk onto ({door_c}, 0) and step UP to enter)",
                    interaction_tile=(door_c, 0)
                ))
                found_target_warp = True
                notes.append(
                    f"ROUTE 2 GATEHOUSE ENTRANCE: The Gatehouse entrance leading into Viridian Forest is visible at ({door_c}, 0). Choose 'ENTER_BUILDING' with target 'GATEHOUSE_DOOR' or 'EXPLORE_NORTH' to proceed into Viridian Forest towards Pewter City!"
                )

        if just_exited_house and not found_target_warp:
            notes.append(
                "JUST EXITED BUILDING: If heading to Professor Oak's Lab in Southeast Pallet Town, choose 'EXPLORE_SOUTHEAST' or 'EXPLORE_SOUTH' to walk around the house and fence."
            )

        # 2c-ii. Rear Roof Eave Corridor:
        # The top roof row of a building is an overhead layer that Red can walk BEHIND (e.g. the strip
        # between a fence and the back of the Poké Mart). Such a tile shows ground in its top few pixels,
        # roof below that, and the roof continues on the tile underneath it.
        # Applied AFTER door detection so building/door structure detection is unaffected.
        def _ground_ratio(p):
            pr_, pg_, pb_ = p[:, :, 0], p[:, :, 1], p[:, :, 2]
            lawn = (pg_ > pr_ + 20) & (pg_ >= pb_)
            dirt = (pr_ >= 130) & (pg_ >= 105) & (pb_ < pr_ - 25) & (pr_ - pg_ < 45)
            gray = (np.abs(pr_ - pg_) < 20) & (np.abs(pg_ - pb_) < 20) & (p.min(axis=2) >= 60)
            return float(np.mean(lawn | dirt | gray))

        eave_tiles = np.zeros_like(tree_mask_grid, dtype=bool)
        for r in range(0, GRID_ROWS - 1):
            for c in range(GRID_COLS):
                if grid[r, c] != CELL_OBSTACLE or tree_mask_grid[r, c] or grid[r + 1, c] == CELL_WALKABLE:
                    continue
                x1, y1, x2, y2 = tile_bounds(c, r)
                tile_p = arr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].astype(int)
                if tile_p.shape[0] < 16 or tile_p.shape[1] < 16:
                    continue
                if _ground_ratio(tile_p[:4]) >= 0.70 and _ground_ratio(tile_p[8:]) < 0.30:
                    eave_tiles[r, c] = True

        for r in range(0, GRID_ROWS - 1):
            for c in range(GRID_COLS):
                if eave_tiles[r, c]:
                    grid[r, c] = CELL_WALKABLE
            # Fill single-tile gaps along the eave row (tiles hidden under Red / follower Pikachu sprites)
            for c in range(1, GRID_COLS - 1):
                if (
                    grid[r, c] == CELL_OBSTACLE
                    and not tree_mask_grid[r, c]
                    and grid[r + 1, c] != CELL_WALKABLE
                    and grid[r, c - 1] == CELL_WALKABLE
                    and grid[r, c + 1] == CELL_WALKABLE
                    and (eave_tiles[r, c - 1] or eave_tiles[r, c + 1])
                ):
                    grid[r, c] = CELL_WALKABLE

        # 2d. Detect Overworld Signs (Gray metal town/route signs and wooden notice signs)
        detected_signs = []
        if not is_dialogue:
            arr_lum = arr.mean(axis=2)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    if player.detected and (c, r) == (player.tile_col, player.tile_row):
                        continue
                    x1, y1, x2, y2 = tile_bounds(c, r)
                    matched_sign = None
                    # Full tile check (rows 0..8) or bottom edge slice (row 9)
                    if r < 9:
                        for dy in (0, -10, -14, 2):
                            yy1, yy2 = y1 + dy, y2 + dy
                            if yy1 < 0 or yy2 > h:
                                continue
                            sg = arr_lum[yy1+2:yy2-6:2, x1+2:x2-2:2]
                            if sg.shape == (12, 14) and sg.std() > 15:
                                if float(np.dot(_norm_pattern(sg), TPL_GRAY_NORM) / TPL_GRAY_NORM.size) > 0.95:
                                    matched_sign = "GRAY_SIGN"
                                    break
                            sw = arr_lum[yy1+2:yy2-8:2, x1+2:x2-2:2]
                            if sw.shape == (11, 14) and sw.std() > 15:
                                if float(np.dot(_norm_pattern(sw), TPL_WOOD_NORM) / TPL_WOOD_NORM.size) > 0.95:
                                    matched_sign = "WOOD_SIGN"
                                    break
                    else:
                        sub_top = arr_lum[y1+2:h:2, x1+2:x2-2:2]
                        if sub_top.shape == (7, 14) and sub_top.std() > 15:
                            sub_n = _norm_pattern(sub_top)
                            if float(np.dot(sub_n, _norm_pattern(_TPL_WOOD_INT[:7])) / sub_n.size) > 0.95:
                                matched_sign = "WOOD_SIGN"
                            elif float(np.dot(sub_n, _norm_pattern(_TPL_GRAY_INT[:7])) / sub_n.size) > 0.95:
                                matched_sign = "GRAY_SIGN"

                    if matched_sign:
                        # Find walkable interaction tile, prioritizing 1 tile South (facing UP at sign)
                        interact_tile = None
                        for dc, dr in [(0, 1), (-1, 0), (1, 0), (0, -1)]:
                            nc, nr = c + dc, r + dr
                            if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS and grid[nr, nc] == CELL_WALKABLE:
                                interact_tile = (nc, nr)
                                break
                        if interact_tile:
                            grid[r, c] = CELL_OBSTACLE
                            detected_signs.append((c, r, interact_tile, matched_sign))

        # Sort signs by Manhattan distance to player and add closest as primary "SIGN" POI
        if detected_signs:
            p_col = player.tile_col if player.detected else 7
            p_row = player.tile_row if player.detected else 4
            detected_signs.sort(key=lambda s: abs(s[2][0] - p_col) + abs(s[2][1] - p_row))
            for i, (sc, sr, sint, skind) in enumerate(detected_signs):
                poi_name = "SIGN" if i == 0 else f"SIGN_{i+1}"
                spx, spy = tile_to_pixel_center(sc, sr)
                pois.append(POI(
                    name=poi_name,
                    category="OBJECT",
                    tile_col=sc,
                    tile_row=sr,
                    pixel_x=spx,
                    pixel_y=spy,
                    description=f"Readable Signpost (Approach at Col {sint[0]}, Row {sint[1]} facing sign and press A to read location / landmark info)",
                    interaction_tile=sint
                ))
            prim_c, prim_r, prim_int, _ = detected_signs[0]
            notes.append(
                f"READABLE SIGNPOST: A readable sign ('SIGN') is at Col {prim_c}, Row {prim_r} (interaction tile at Col {prim_int[0]}, Row {prim_int[1]}). If you are lost, disoriented, or unsure of your current location, choose 'GO_TO_POI' with target_poi 'SIGN' to read it and identify your location!"
            )

        # 3. Compute BFS Directional Reachability from Red's current tile so the LLM knows which directions are OPEN vs BLOCKED (dead ends)
        sc_col = player.tile_col if player.detected else 7
        sr_row = player.tile_row if player.detected else 4
        reachable_tiles = {(sc_col, sr_row)}
        bfs_q = deque([(sc_col, sr_row)])
        while bfs_q:
            cc, cr = bfs_q.popleft()
            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nc, nr = cc + dc, cr + dr
                if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS and (nc, nr) not in reachable_tiles:
                    if grid[nr, nc] in (CELL_WALKABLE, CELL_WARP):
                        reachable_tiles.add((nc, nr))
                        bfs_q.append((nc, nr))

        dir_reachability = {
            "NORTH": any(r <= sr_row - 2 for _, r in reachable_tiles),
            "SOUTH": any(r >= sr_row + 2 for _, r in reachable_tiles),
            "EAST": any(c >= sc_col + 2 for c, _ in reachable_tiles),
            "WEST": any(c <= sc_col - 2 for c, _ in reachable_tiles),
            "NORTHEAST": any(c >= sc_col + 1 and r <= sr_row - 1 and (c - sc_col) + (sr_row - r) >= 3 for c, r in reachable_tiles),
            "NORTHWEST": any(c <= sc_col - 1 and r <= sr_row - 1 and (sc_col - c) + (sr_row - r) >= 3 for c, r in reachable_tiles),
            "SOUTHEAST": any(c >= sc_col + 1 and r >= sr_row + 1 and (c - sc_col) + (r - sr_row) >= 3 for c, r in reachable_tiles),
            "SOUTHWEST": any(c <= sc_col - 1 and r >= sr_row + 1 and (sc_col - c) + (r - sr_row) >= 3 for c, r in reachable_tiles),
        }

        open_dirs = [d for d, is_open in dir_reachability.items() if is_open]
        blocked_dirs = [d for d, is_open in dir_reachability.items() if not is_open]

        if not is_dialogue:
            if blocked_dirs:
                notes.append(
                    f"BLOCKED DIRECTIONS (STUCK / DEAD END): Cannot move {', '.join(blocked_dirs)} from here — completely blocked by impassable trees/walls/water! Do NOT choose {', '.join('EXPLORE_' + d for d in blocked_dirs)}!"
                )
            if open_dirs:
                notes.append(
                    f"OPEN EXPLORATION DIRECTIONS: {', '.join('EXPLORE_' + d for d in open_dirs)}."
                )

        notes.extend([
            "Red is outdoors in the Overworld.",
            "NAVIGATION: Impassable trees, buildings, and fences are mapped. Pathfinding will automatically route around fences and walls.",
            "Ledges can only be hopped going SOUTH. You cannot jump UP over brown ledges."
        ])

        available_intents.extend(["GO_TO_POI", "ENTER_BUILDING"])
        if is_dialogue or not open_dirs:
            available_intents.extend(["EXPLORE_SOUTH", "EXPLORE_NORTH", "EXPLORE_WEST", "EXPLORE_EAST"])
        else:
            for d in open_dirs:
                available_intents.append(f"EXPLORE_{d}")
        available_intents.extend(["TALK_TO_NPC", "WAIT"])

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
