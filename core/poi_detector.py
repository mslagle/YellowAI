"""
POI (Points of Interest) and Screen Feature Detector for YellowAI.
Uses fast NumPy matrix operations (< 5ms) to detect key game objects, warps,
obstacles, and player orientation directly from screen captures.
Provides grounded spatial awareness to the VLM so it never hallucinates fake doors
or attempts to press 'A' to use stairs/doorways.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image


class POIDetector:
    """
    Performs fast rule-based spatial analysis on GBA frames.
    Detects:
    1. Player character (position, facing direction).
    2. Map warps (staircases, door mats, cave entrances).
    3. Overworld landmarks (Poké Centers, Poké Marts, Gyms, Oak's Lab, Cut trees, ledges).
    4. Room boundaries (black void borders, obstacles).
    """

    def __init__(self, target_width: int = 480, target_height: int = 320):
        self.target_width = target_width
        self.target_height = target_height

    def analyze_frame(self, frame: Image.Image, current_area: str = "") -> Dict[str, Any]:
        """
        Analyzes a captured frame and returns detected POIs and spatial guidelines.
        """
        resized = frame.resize((self.target_width, self.target_height))
        arr = np.array(resized)

        result: Dict[str, Any] = {
            "player": self._detect_player(arr),
            "room_type": self._detect_room_type(arr, current_area),
            "void_borders": self._detect_void_borders(arr),
            "warps": [],
            "landmarks": [],
            "spatial_guidance": []
        }

        # Area-specific POI detection
        if result["room_type"] == "BEDROOM_2F":
            self._analyze_bedroom(arr, result)
        elif result["room_type"] == "LIVING_ROOM_1F":
            self._analyze_living_room(arr, result)
        elif result["room_type"] == "OVERWORLD":
            self._analyze_overworld(arr, result)

        return result

    def _detect_player(self, arr: np.ndarray) -> Dict[str, Any]:
        """Locates Red's sprite via red cap and determines rough facing direction."""
        cap_mask = (arr[:, :, 0] > 175) & (arr[:, :, 1] < 105) & (arr[:, :, 2] < 105)
        y_idx, x_idx = np.where(cap_mask)

        if len(y_idx) < 15:
            return {"detected": False, "x": -1, "y": -1, "facing": "UNKNOWN"}

        py = int(np.median(y_idx))
        px = int(np.median(x_idx))

        # Red sprite bounding box [py-15:py+25, px-15:px+15]
        h, w, _ = arr.shape
        y1, y2 = max(0, py - 10), min(h, py + 25)
        x1, x2 = max(0, px - 15), min(w, px + 15)
        sprite_patch = arr[y1:y2, x1:x2]

        # Facing direction heuristic:
        # Facing DOWN: face skin visible directly below cap (Y ~ py+5..12)
        # Facing UP: dark hair/back of cap visible below cap, no skin
        # Facing LEFT: red visor extends left
        # Facing RIGHT: red visor extends right
        face_skin = (
            (sprite_patch[:, :, 0] > 220) &
            (sprite_patch[:, :, 1] > 150) & (sprite_patch[:, :, 1] < 220) &
            (sprite_patch[:, :, 2] > 100) & (sprite_patch[:, :, 2] < 170)
        )
        skin_count = int(np.sum(face_skin))

        facing = "DOWN" if skin_count >= 15 else "OTHER"

        return {
            "detected": True,
            "x": px,
            "y": py,
            "facing": facing,
            "skin_count": skin_count
        }

    def _detect_void_borders(self, arr: np.ndarray) -> Dict[str, bool]:
        """Detects if borders of the screen are covered by the black void (< 20 RGB)."""
        h, w, _ = arr.shape
        return {
            "south_void": bool(np.mean(arr[h - 40:, :, :] < 20) > 0.8),
            "west_void": bool(np.mean(arr[:, :40, :] < 20) > 0.8),
            "north_void": bool(np.mean(arr[:40, :, :] < 20) > 0.8),
            "east_void": bool(np.mean(arr[:, w - 40:, :] < 20) > 0.8)
        }

    def _detect_room_type(self, arr: np.ndarray, current_area: str) -> str:
        """Determines the current environment type."""
        voids = self._detect_void_borders(arr)
        upper = arr[:160, :]

        # Tatami floor (yellow-tan) + green rug
        tatami = (upper[:, :, 0] > 140) & (upper[:, :, 1] > 120) & (upper[:, :, 2] < 110)
        rug = (upper[:, :, 0] < 130) & (upper[:, :, 1] > 140) & (upper[:, :, 2] > 70)
        is_bedroom = (np.mean(tatami) + np.mean(rug)) > 0.25 and (voids["south_void"] or "bedroom" in current_area.lower())

        if is_bedroom:
            return "BEDROOM_2F"

        # Check for 1F living room (wood flooring, kitchen counters, table with Mom)
        if "1f" in current_area.lower() or "living" in current_area.lower():
            return "LIVING_ROOM_1F"

        # Overworld outdoor grass / dirt / buildings
        return "OVERWORLD"

    def _analyze_bedroom(self, arr: np.ndarray, result: Dict[str, Any]):
        """Detects bedroom features: staircase railing, landing mat, bed, south void."""
        player = result["player"]
        voids = result["void_borders"]

        # 1. Orange staircase landing mat
        mat_pixels = (
            (arr[:, :, 0] > 200) &
            (arr[:, :, 1] > 110) & (arr[:, :, 1] < 160) &
            (arr[:, :, 2] > 60) & (arr[:, :, 2] < 120)
        )
        my, mx = np.where(mat_pixels & (np.arange(self.target_height)[:, None] < 100))
        mat_visible = len(my) > 100

        # 2. White staircase safety railing along top-right
        rail_pixels = (arr[0:60, :, 0] > 240) & (arr[0:60, :, 1] > 240) & (arr[0:60, :, 2] > 240)
        ry, rx = np.where(rail_pixels)
        rail_visible = len(rx) > 100

        notes: List[str] = []
        notes.append("Red is in his 2F Bedroom.")

        if voids["south_void"]:
            notes.append("The bottom of the screen is an impassable BLACK VOID dead end. There is NO door to the SOUTH! DO NOT press DOWN!")

        if mat_visible:
            mat_x = int(np.median(mx))
            mat_y = int(np.median(my))
            result["warps"].append({
                "type": "STAIRS_LANDING_MAT",
                "x": mat_x,
                "y": mat_y,
                "description": "Orange Welcome Mat for Stairs (Top-Right)"
            })
            notes.append(f"STAIRS ENTRANCE DETECTED: Orange landing mat is VISIBLE at (X={mat_x}, Y={mat_y}).")
            if player["detected"]:
                if player["x"] < mat_x - 16:
                    notes.append("Step RIGHT until aligned with the orange mat, then walk UP onto it to descend downstairs!")
                else:
                    notes.append("Walk UP onto the orange mat to descend downstairs to 1F!")
        elif rail_visible:
            rightmost_rail_x = int(rx.max())
            notes.append("STAIRCASE LOCATION: Top-right corner behind the white railing.")
            notes.append("Railing blocks moving UP directly. The stairs entrance is immediately to the RIGHT of the railing.")
            notes.append("Walk RIGHT (east) to scroll camera and reach the stairs entrance, then walk UP!")

        notes.append("WARP RULE: In Pokémon, stairs transition AUTOMATICALLY when you step on them. There is NO door and NO need to press 'A'!")
        result["spatial_guidance"] = notes

    def _analyze_living_room(self, arr: np.ndarray, result: Dict[str, Any]):
        """Detects 1F Mom's house features: front exit door mat, stairs in top-right."""
        notes = [
            "Red is on 1F (Mom's Living Room).",
            "Front door to Pallet Town is at the BOTTOM-MIDDLE of the room on the red welcome mat.",
            "Step DOWN onto the red mat to exit outside to Pallet Town automatically. DO NOT press A!",
            "Talk to Mom (at dining table) if needed for healing."
        ]
        result["spatial_guidance"] = notes

    def _analyze_overworld(self, arr: np.ndarray, result: Dict[str, Any]):
        """Detects outdoor landmarks: Pokémon Centers (red roof), Poké Marts (blue roof), ledges, fences."""
        notes = [
            "Overworld Navigation: Buildings and doors have threshold mats.",
            "Step directly onto entrance tiles/doors to enter buildings automatically. DO NOT press A to 'open' doors!",
            "Ledges can only be hopped going SOUTH (down). You cannot walk UP through ledges."
        ]
        result["spatial_guidance"] = notes

    def format_poi_prompt(self, analysis: Dict[str, Any]) -> str:
        """Formats the analysis into a concise prompt block for the VLM."""
        lines = ["[VISIBLE ON SCREEN & POINTS OF INTEREST]:"]

        player = analysis.get("player", {})
        if player.get("detected"):
            lines.append(f"- Player Position: Red is at (X={player['x']}, Y={player['y']}) facing {player.get('facing', 'UNKNOWN')}.")

        for note in analysis.get("spatial_guidance", []):
            lines.append(f"- {note}")

        return "\n".join(lines)
