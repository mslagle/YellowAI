"""
POI (Points of Interest) and Screen Feature Detector for YellowAI.
Delegates spatial perception to ScreenAnalyzer so both interfaces share the same
high-accuracy NumPy scene classifier, player localization, and POI detector.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from core.screen_analyzer import ScreenAnalyzer


class POIDetector:
    """
    Performs fast rule-based spatial analysis on GBA frames via ScreenAnalyzer.
    """

    def __init__(
        self,
        target_width: int = 480,
        target_height: int = 320,
        debug_save_path: Optional[str] = "logs/last_poi_annotated.png"
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.debug_save_path = debug_save_path
        self.analyzer = ScreenAnalyzer(
            target_width=target_width,
            target_height=target_height,
            debug_poi_path=debug_save_path or "logs/last_poi_annotated.png"
        )

    def analyze_frame(self, frame: Image.Image, current_area: str = "") -> Dict[str, Any]:
        """
        Analyzes a captured frame using ScreenAnalyzer and returns a dictionary of detected
        player coordinates, room_type, void_borders, warps, landmarks, and spatial_guidance.
        """
        scene = self.analyzer.analyze(frame, current_area_hint=current_area)
        resized = frame.resize((self.target_width, self.target_height)).convert("RGB")
        arr = np.array(resized)

        warps: List[Dict[str, Any]] = []
        landmarks: List[Dict[str, Any]] = []
        for poi in scene.pois:
            entry = {
                "name": poi.name,
                "type": poi.name,
                "category": poi.category,
                "col": poi.tile_col,
                "row": poi.tile_row,
                "interaction_tile": poi.interaction_tile,
                "x": poi.pixel_x,
                "y": poi.pixel_y,
                "description": poi.description,
            }
            if poi.category == "WARP":
                warps.append(entry)
            else:
                landmarks.append(entry)

        return {
            "player": {
                "detected": scene.player.detected,
                "x": scene.player.pixel_x,
                "y": scene.player.pixel_y,
                "col": scene.player.tile_col,
                "row": scene.player.tile_row,
                "facing": scene.player.facing,
            },
            "room_type": scene.area_type,
            "void_borders": self._detect_void_borders(arr),
            "warps": warps,
            "landmarks": landmarks,
            "spatial_guidance": list(scene.spatial_notes),
        }

    def _detect_void_borders(self, arr: np.ndarray) -> Dict[str, bool]:
        """Detects if borders of the screen are covered by the black void (< 25 RGB)."""
        h, w, _ = arr.shape
        return {
            "south_void": bool(np.mean(arr[h - 40:, :, :] < 25) > 0.75),
            "west_void": bool(np.mean(arr[:, :40, :] < 25) > 0.75),
            "north_void": bool(np.mean(arr[:40, :, :] < 25) > 0.75),
            "east_void": bool(np.mean(arr[:, w - 40:, :] < 25) > 0.75),
        }

    def format_poi_prompt(self, analysis: Dict[str, Any]) -> str:
        """Formats the analysis into a concise prompt block for the VLM."""
        lines = ["[VISIBLE ON SCREEN & POINTS OF INTEREST]:"]

        player = analysis.get("player", {})
        if player.get("detected"):
            lines.append(
                f"- Player Position: Red is at Tile (Col={player.get('col')}, Row={player.get('row')}) "
                f"facing {player.get('facing', 'UNKNOWN')}."
            )

        for note in analysis.get("spatial_guidance", []):
            lines.append(f"- {note}")

        return "\n".join(lines)
