"""
Deterministic Movement Controller and Pathfinding Engine for YellowAI.
Executes high-level navigation intents (EXIT_ROOM, GO_TO_POI, TALK_TO_NPC, EXPLORE_NORTH, etc.)
by:
1. Finding optimal collision-free routes on the 15x10 tile grid via A* / BFS pathfinding.
2. Translating waypoints into precise GBA button sequences, accounting for GBA character facing and turning mechanics.
3. Generating and saving `logs/last_route_planned.png` showing Start, End, collision grid, and route line.
4. Executing steps with reactive interruption (halts immediately if scene transitions, dialogue opens, or battle triggers).
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from collections import deque
import heapq
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.screen_analyzer import (
    SceneAnalysis,
    POI,
    GRID_COLS,
    GRID_ROWS,
    TILE_SIZE,
    CELL_WALKABLE,
    CELL_VOID,
    CELL_WALL,
    CELL_OBSTACLE,
    CELL_RAILING,
    CELL_COUNTER,
    CELL_NPC,
    CELL_WARP
)
from emulator.input_controller import InputController
from emulator.screen_capture import ScreenCapture


class MovementController:
    """
    Motor cortex of YellowAI.
    Translates high-level strategic intents into grounded, obstacle-free GBA button sequences.
    """

    def __init__(
        self,
        input_controller: Optional[InputController] = None,
        screen_capture: Optional[ScreenCapture] = None,
        step_settle_delay: float = 0.18,
        route_debug_path: str = "logs/last_route_planned.png"
    ):
        self.input_controller = input_controller
        self.screen_capture = screen_capture
        self.step_settle_delay = step_settle_delay
        self.route_debug_path = route_debug_path

    # -------------------------------------------------------------------------
    # Pathfinding (A* on 15x10 Collision Grid)
    # -------------------------------------------------------------------------
    def find_path(
        self,
        collision_grid: np.ndarray,
        start_tile: Tuple[int, int],
        goal_tile: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Computes the shortest walkable path from start_tile (col, row) to goal_tile (col, row).
        Returns a list of tile coordinates [(sc, sr), ..., (gc, gr)] or empty list if no path exists.
        """
        start_c, start_r = start_tile
        goal_c, goal_r = goal_tile

        if not (0 <= start_c < GRID_COLS and 0 <= start_r < GRID_ROWS):
            return []
        if not (0 <= goal_c < GRID_COLS and 0 <= goal_r < GRID_ROWS):
            return []
        if start_tile == goal_tile:
            return [start_tile]

        # Priority queue: (f_score, cost, (col, row), path)
        open_set = []
        start_h = abs(start_c - goal_c) + abs(start_r - goal_r)
        heapq.heappush(open_set, (start_h, 0, start_tile, [start_tile]))

        visited = set()

        while open_set:
            f, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            curr_c, curr_r = current
            if current == goal_tile:
                return path

            # 4-directional cardinal movement (Up, Down, Left, Right)
            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nc = curr_c + dc
                nr = curr_r + dr

                if not (0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS):
                    continue
                if (nc, nr) in visited:
                    continue

                cell_val = collision_grid[nr, nc]
                # Walkable condition: cell must be WALKABLE or WARP, or the destination goal tile itself
                is_goal = (nc, nr) == goal_tile
                is_passable = (cell_val == CELL_WALKABLE or cell_val == CELL_WARP or is_goal)

                if not is_passable:
                    continue

                new_cost = cost + 1
                new_h = abs(nc - goal_c) + abs(nr - goal_r)
                new_f = new_cost + new_h
                heapq.heappush(open_set, (new_f, new_cost, (nc, nr), path + [(nc, nr)]))

        return []

    # -------------------------------------------------------------------------
    # Intent Translation & Target Destination Selection
    # -------------------------------------------------------------------------
    def resolve_target_tile(
        self,
        intent: str,
        target_poi_name: Optional[str],
        scene: SceneAnalysis
    ) -> Optional[Tuple[int, int, str]]:
        """
        Resolves the target tile (col, row) and label based on the high-level intent and scene state.
        Returns (target_col, target_row, target_label) or None.
        """
        intent = (intent or "WAIT").upper().strip()
        p = scene.player
        if not p.detected:
            return None

        # 1. EXIT_ROOM: Target the primary warp exit in the room
        if intent == "EXIT_ROOM":
            for poi in scene.pois:
                if poi.category == "WARP":
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            # Fallbacks by area
            if scene.area_type == "BEDROOM_2F":
                return 12, 2, "STAIRS_DOWN"
            elif scene.area_type == "LIVING_ROOM_1F":
                return 4, 7, "FRONT_DOOR"
            elif scene.area_type == "OAKS_LAB":
                return 5, 8, "LAB_EXIT"
            elif scene.area_type == "POKECENTER":
                return 7, 8, "CENTER_EXIT"
            elif scene.area_type == "POKEMART":
                return 4, 8, "MART_EXIT"

        # 2. GO_TO_POI / TALK_TO_NPC: Look up specific POI
        if intent in ("GO_TO_POI", "TALK_TO_NPC") and target_poi_name:
            t_upper = target_poi_name.upper().strip()
            for poi in scene.pois:
                if t_upper in poi.name.upper() or poi.name.upper() in t_upper:
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name

            # If TALK_TO_NPC and no specific name matched, pick nearest NPC
            if intent == "TALK_TO_NPC":
                for poi in scene.pois:
                    if poi.category == "NPC":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name

        # 3. Directional exploration (EXPLORE_NORTH, SOUTH, EAST, WEST)
        if intent.startswith("EXPLORE_"):
            direction = intent.split("_")[-1]
            return self._find_furthest_walkable_tile(direction, (p.tile_col, p.tile_row), scene.collision_grid)

        # 4. ENTER_BUILDING: Look for building entrance / warp door
        if intent == "ENTER_BUILDING":
            for poi in scene.pois:
                if poi.category == "WARP":
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            # In overworld, walk NORTH towards building door
            return self._find_furthest_walkable_tile("NORTH", (p.tile_col, p.tile_row), scene.collision_grid)

        return None

    def _find_furthest_walkable_tile(
        self,
        direction: str,
        start_tile: Tuple[int, int],
        grid: np.ndarray
    ) -> Tuple[int, int, str]:
        """Scans in the cardinal direction to find the furthest walkable tile up to 4 tiles away."""
        sc, sr = start_tile
        vectors = {
            "NORTH": (0, -1),
            "SOUTH": (0, 1),
            "WEST": (-1, 0),
            "EAST": (1, 0)
        }
        dc, dr = vectors.get(direction.upper(), (0, 0))
        target_c, target_r = sc, sr

        for step in range(1, 5):
            nc = sc + dc * step
            nr = sr + dr * step
            if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                if grid[nr, nc] == CELL_WALKABLE:
                    target_c, target_r = nc, nr
                else:
                    break
            else:
                break

        return target_c, target_r, f"EXPLORE_{direction}"

    # -------------------------------------------------------------------------
    # Waypoints to GBA Buttons Converter
    # -------------------------------------------------------------------------
    def convert_path_to_buttons(
        self,
        path: List[Tuple[int, int]],
        initial_facing: str = "UNKNOWN"
    ) -> List[str]:
        """
        Translates a list of tile waypoints into exact GBA button presses.
        Accounts for GBA turning mechanics: when changing direction,
        an extra tap of that direction is prepended so Red turns in place then steps forward.
        """
        if len(path) <= 1:
            return []

        buttons: List[str] = []
        current_facing = initial_facing.upper().strip()

        for i in range(len(path) - 1):
            c1, r1 = path[i]
            c2, r2 = path[i + 1]

            if c2 > c1:
                step_btn = "RIGHT"
            elif c2 < c1:
                step_btn = "LEFT"
            elif r2 > r1:
                step_btn = "DOWN"
            elif r2 < r1:
                step_btn = "UP"
            else:
                continue

            # GBA Facing Mechanics:
            # If changing direction from what Red was facing, the first press only turns in place!
            if current_facing != "UNKNOWN" and current_facing != step_btn:
                buttons.append(step_btn)  # Turn press
                current_facing = step_btn

            buttons.append(step_btn)      # Move step press
            current_facing = step_btn

        return buttons

    # -------------------------------------------------------------------------
    # Route Image Visualizer (Dual Debug #2)
    # -------------------------------------------------------------------------
    def save_route_visualization(
        self,
        frame: Image.Image,
        scene: SceneAnalysis,
        start_tile: Tuple[int, int],
        goal_tile: Tuple[int, int],
        target_name: str,
        path: List[Tuple[int, int]],
        buttons: List[str]
    ):
        """
        Generates and saves `logs/last_route_planned.png` showing:
        - The underlying captured frame
        - Semi-transparent collision grid overlay (red blocked tiles)
        - Green START circle + text at Red's tile
        - Gold/Red END box + text at destination tile
        - Bold polyline route connecting Start -> End
        - Title banner with Intent, Step count, and Buttons
        """
        try:
            vis = frame.resize((480, 320)).copy()
            draw = ImageDraw.Draw(vis)

            # 1. Draw collision grid overlay
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    val = scene.collision_grid[r, c]
                    x1 = c * TILE_SIZE
                    y1 = r * TILE_SIZE
                    x2 = (c + 1) * TILE_SIZE
                    y2 = (r + 1) * TILE_SIZE
                    if val != CELL_WALKABLE:
                        draw.rectangle([x1, y1, x2, y2], outline=(180, 40, 40), width=1)

            # 2. Draw planned path line
            if len(path) > 1:
                pixel_points = [
                    (col * TILE_SIZE + 16, row * TILE_SIZE + 16)
                    for col, row in path
                ]
                # Draw thick route line
                for i in range(len(pixel_points) - 1):
                    p1 = pixel_points[i]
                    p2 = pixel_points[i + 1]
                    draw.line([p1, p2], fill=(0, 255, 255), width=4)
                    # Draw waypoint dot
                    draw.ellipse([p2[0] - 3, p2[1] - 3, p2[0] + 3, p2[1] + 3], fill=(255, 255, 0))

            # 3. Draw START marker (Green circle)
            sc, sr = start_tile
            sx = sc * TILE_SIZE + 16
            sy = sr * TILE_SIZE + 16
            draw.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 255, 0), width=3)
            draw.text((max(0, sx - 20), max(0, sy - 25)), "START", fill=(0, 255, 0))

            # 4. Draw END marker (Gold / Red target box)
            gc, gr = goal_tile
            gx = gc * TILE_SIZE + 16
            gy = gr * TILE_SIZE + 16
            draw.rectangle([gx - 12, gy - 12, gx + 12, gy + 12], outline=(255, 50, 50), width=3)
            draw.text((max(0, gx - 32), max(0, gy - 25)), f"END: {target_name}", fill=(255, 50, 50))

            # 5. Header banner
            draw.rectangle([0, 0, 480, 22], fill=(10, 10, 30, 220))
            btn_str = " -> ".join(buttons[:6]) + ("..." if len(buttons) > 6 else "")
            draw.text((8, 4), f"Route Plan: {target_name} | {len(path)} tiles | Buttons: [{btn_str}]", fill=(0, 255, 255))

            Path(self.route_debug_path).parent.mkdir(parents=True, exist_ok=True)
            vis.save(self.route_debug_path)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Full Intent Execution Pipeline
    # -------------------------------------------------------------------------
    def plan_and_execute(
        self,
        intent: str,
        target_poi_name: Optional[str],
        scene: SceneAnalysis,
        frame: Image.Image
    ) -> Dict[str, Any]:
        """
        Full motor execution pipeline:
        1. Resolves target destination tile.
        2. Computes A* collision-free path.
        3. Converts to GBA button sequence.
        4. Saves visual route map to `logs/last_route_planned.png`.
        5. Executes buttons sequentially with reactive interrupts.
        """
        p = scene.player
        if not p.detected:
            # If player sprite isn't visible (e.g. dialogue, title, menu), fallback to A or B
            if intent == "ADVANCE_DIALOGUE":
                if self.input_controller:
                    self.input_controller.press_button("A")
                return {"success": True, "buttons": ["A"], "status": "ADVANCED_DIALOGUE"}
            return {"success": False, "buttons": [], "status": "PLAYER_NOT_DETECTED"}

        start_tile = (p.tile_col, p.tile_row)
        resolved = self.resolve_target_tile(intent, target_poi_name, scene)

        if not resolved:
            return {"success": False, "buttons": [], "status": f"COULD_NOT_RESOLVE_{intent}"}

        goal_c, goal_r, target_name = resolved
        goal_tile = (goal_c, goal_r)

        # Compute collision-free path via A*
        path = self.find_path(scene.collision_grid, start_tile, goal_tile)
        if not path:
            # If direct path blocked, attempt adjacent tile
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj = (goal_c + dc, goal_r + dr)
                path = self.find_path(scene.collision_grid, start_tile, adj)
                if path:
                    goal_tile = adj
                    break

        # Convert waypoints to button sequence
        buttons = self.convert_path_to_buttons(path, initial_facing=p.facing)

        # Save route debug visualization
        self.save_route_visualization(
            frame=frame,
            scene=scene,
            start_tile=start_tile,
            goal_tile=goal_tile,
            target_name=target_name,
            path=path,
            buttons=buttons
        )

        # Execute buttons if input_controller is connected
        executed_buttons: List[str] = []
        interrupted: bool = False
        interrupt_reason: Optional[str] = None

        if self.input_controller and buttons:
            prev_thumb = None
            for idx, btn in enumerate(buttons):
                if self.screen_capture and not self.screen_capture.is_window_alive():
                    interrupted = True
                    interrupt_reason = "EMULATOR_CLOSED"
                    break

                self.input_controller.press_button(btn)
                executed_buttons.append(btn)
                time.sleep(self.step_settle_delay)

                # Reactive visual verification
                if self.screen_capture:
                    step_frame, curr_thumb = self.screen_capture.capture_frame()
                    if prev_thumb is not None and curr_thumb is not None:
                        diff = float(np.mean(np.abs(curr_thumb.astype(float) - prev_thumb.astype(float))))
                        # Scene warp (doorway / stairs / battle flash)
                        if diff >= 50.0:
                            interrupted = True
                            interrupt_reason = "SCENE_WARP_OR_BATTLE"
                            break
                    prev_thumb = curr_thumb

        return {
            "success": True,
            "target": target_name,
            "path_length": len(path),
            "planned_buttons": buttons,
            "executed_buttons": executed_buttons,
            "interrupted": interrupted,
            "interrupt_reason": interrupt_reason
        }
