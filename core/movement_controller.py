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
    GRID_Y_OFFSET,
    tile_bounds,
    tile_to_pixel_center,
    pixel_to_tile,
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
        step_settle_delay: float = 0.25,
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
                # Walkable condition: intermediate tiles must be CELL_WALKABLE; the destination goal_tile itself is always passable
                is_goal = (nc, nr) == goal_tile
                is_passable = (cell_val == CELL_WALKABLE or is_goal)

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
            if scene.area_type == "LIVING_ROOM_1F":
                for poi in scene.pois:
                    if poi.name == "FRONT_DOOR":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
                return 2, 9, "FRONT_DOOR"
            for poi in scene.pois:
                if poi.category == "WARP":
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            # Fallbacks by area
            if scene.area_type == "BEDROOM_2F":
                return 9, 4, "STAIRS_DOWN"
            elif scene.area_type == "LIVING_ROOM_1F":
                return 2, 9, "FRONT_DOOR"
            elif scene.area_type == "OAKS_LAB":
                return 5, 8, "LAB_EXIT"
            elif scene.area_type == "POKECENTER":
                return 7, 8, "CENTER_EXIT"
            elif scene.area_type == "POKEMART":
                return 4, 8, "MART_EXIT"

        # 2. Explicit target_poi_name lookup (works across GO_TO_POI, TALK_TO_NPC, ENTER_BUILDING, and EXPLORE_*)
        if target_poi_name:
            t_upper = target_poi_name.upper().strip()
            if scene.area_type == "LIVING_ROOM_1F" and any(k in t_upper for k in ["DOOR", "EXIT", "OUTSIDE"]):
                for poi in scene.pois:
                    if poi.name == "FRONT_DOOR":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
                return 2, 9, "FRONT_DOOR"
            if scene.area_type == "OAKS_LAB" and "EXIT" not in t_upper:
                if any(k in t_upper for k in ["STARTER", "TABLE", "BALL", "LAB", "POKEMON"]):
                    for poi in scene.pois:
                        if poi.name == "STARTER_TABLE":
                            return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
                elif "OAK" in t_upper:
                    for poi in scene.pois:
                        if poi.name == "PROFESSOR_OAK":
                            return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            for poi in scene.pois:
                if t_upper in poi.name.upper() or poi.name.upper() in t_upper:
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            if scene.area_type == "OVERWORLD" and any(k in t_upper for k in ["LAB", "OAK"]):
                for poi in scene.pois:
                    if poi.name == "OAKS_LAB":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
                return self._find_furthest_walkable_tile("SOUTHEAST", (p.tile_col, p.tile_row), scene.collision_grid)
            if scene.area_type == "OVERWORLD" and any(k in t_upper for k in ["MOM", "RED", "HOME"]):
                house_doors = [poi for poi in scene.pois if poi.name == "BUILDING_DOOR"]
                if house_doors:
                    west_door = min(house_doors, key=lambda d: d.tile_col)
                    return west_door.interaction_tile[0], west_door.interaction_tile[1], west_door.name
            if scene.area_type == "OVERWORLD" and any(k in t_upper for k in ["DAISY", "RIVAL", "SISTER", "MAP"]):
                house_doors = [poi for poi in scene.pois if poi.name == "BUILDING_DOOR"]
                if house_doors:
                    east_door = max(house_doors, key=lambda d: d.tile_col)
                    return east_door.interaction_tile[0], east_door.interaction_tile[1], east_door.name
            if scene.area_type == "OVERWORLD" and any(k in t_upper for k in ["DOOR", "BUILDING", "CENTER", "MART", "HOUSE"]):
                for poi in scene.pois:
                    if poi.category == "WARP":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name

        if intent == "TALK_TO_NPC":
            for poi in scene.pois:
                if poi.category == "NPC":
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name

        # 3. Directional exploration (EXPLORE_NORTH, SOUTH, EAST, WEST, SOUTHEAST, etc.)
        if intent.startswith("EXPLORE_"):
            direction = intent.split("_")[-1]
            # If Professor Oak's Lab is actively detected on screen in Pallet Town (before Pikachu is following Red)
            # and the LLM orders EXPLORE_SOUTH / EXPLORE_SOUTHEAST / EXPLORE_EAST, snap directly to OAKS_LAB so Red never walks past it!
            has_pikachu_following = any("PIKACHU_FOLLOWING" in n for n in scene.spatial_notes)
            if scene.area_type == "OVERWORLD" and direction in ("SOUTH", "SOUTHEAST", "EAST") and not has_pikachu_following:
                for poi in scene.pois:
                    if poi.name == "OAKS_LAB":
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            return self._find_furthest_walkable_tile(direction, (p.tile_col, p.tile_row), scene.collision_grid)

        # 4. ENTER_BUILDING / GO_TO_POI fallback:
        # Inside OAKS_LAB / POKECENTER / POKEMART, never default a generic GO_TO_POI / ENTER_BUILDING to the room's exit door!
        if intent in ("ENTER_BUILDING", "GO_TO_POI"):
            if scene.area_type in ("OAKS_LAB", "POKECENTER", "POKEMART"):
                for poi in scene.pois:
                    if poi.category in ("OBJECT", "NPC") and "EXIT" not in poi.name:
                        return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            for poi in scene.pois:
                if poi.category == "WARP":
                    return poi.interaction_tile[0], poi.interaction_tile[1], poi.name
            # Fallback to directional exploration
            return self._find_furthest_walkable_tile("SOUTH", (p.tile_col, p.tile_row), scene.collision_grid)

        return None

    def _find_furthest_walkable_tile(
        self,
        direction: str,
        start_tile: Tuple[int, int],
        grid: np.ndarray
    ) -> Tuple[int, int, str]:
        """
        Finds the best reachable walkable tile in the requested cardinal or intercardinal (diagonal) direction.
        Supports: NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST.
        If a fence, wall, or tree blocks the straight line within 3 tiles, automatically selects
        a reachable tile further in that direction around the obstacle so A* routes cleanly around it!
        """
        sc, sr = start_tile
        dir_upper = direction.upper().strip()
        is_diagonal = dir_upper in ("NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST")

        if not is_diagonal:
            vectors = {
                "NORTH": (0, -1),
                "SOUTH": (0, 1),
                "WEST": (-1, 0),
                "EAST": (1, 0)
            }
            dc, dr = vectors.get(dir_upper, (0, 0))
            target_c, target_r = sc, sr
            straight_steps = 0

            for step in range(1, 5):
                nc = sc + dc * step
                nr = sr + dr * step
                if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                    if grid[nr, nc] == CELL_WALKABLE:
                        target_c, target_r = nc, nr
                        straight_steps = step
                    else:
                        break
                else:
                    break

            next_c = sc + dc * (straight_steps + 1)
            next_r = sr + dr * (straight_steps + 1)
            blocked_ahead = (
                straight_steps < 4
                and 0 <= next_c < GRID_COLS
                and 0 <= next_r < GRID_ROWS
                and grid[next_r, next_c] != CELL_WALKABLE
            )

            if not blocked_ahead and straight_steps >= 3:
                return target_c, target_r, f"EXPLORE_{dir_upper}"
        else:
            target_c, target_r = sc, sr

        # Find the best reachable tile in the target cardinal or diagonal quadrant via A*
        best_candidate = None
        best_score = -9999.0

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if (c, r) == start_tile or grid[r, c] != CELL_WALKABLE:
                    continue

                if dir_upper == "SOUTH":
                    advance = r - sr
                    ortho = abs(c - sc)
                    corridor_open = (r == 0 or grid[r - 1, c] == CELL_WALKABLE) and (r <= 1 or grid[r - 2, c] == CELL_WALKABLE)
                    east_bias = 4.0 if c > sc else 0.0
                    max_adv = 5
                elif dir_upper == "NORTH":
                    advance = sr - r
                    ortho = abs(c - sc)
                    corridor_open = (r == GRID_ROWS - 1 or grid[r + 1, c] == CELL_WALKABLE)
                    east_bias = 0.0
                    max_adv = 5
                elif dir_upper == "EAST":
                    advance = c - sc
                    ortho = abs(r - sr)
                    corridor_open = (c == 0 or grid[r, c - 1] == CELL_WALKABLE)
                    east_bias = 0.0
                    max_adv = 5
                elif dir_upper == "WEST":
                    advance = sc - c
                    ortho = abs(r - sr)
                    corridor_open = (c == GRID_COLS - 1 or grid[r, c + 1] == CELL_WALKABLE)
                    east_bias = 0.0
                    max_adv = 5
                elif dir_upper in ("SOUTHEAST", "SOUTHWEST", "NORTHEAST", "NORTHWEST"):
                    dx = (c - sc) if "EAST" in dir_upper else (sc - c)
                    dy = (r - sr) if "SOUTH" in dir_upper else (sr - r)
                    if dx <= 0 or dy <= 0:
                        continue
                    advance = dx + dy
                    ortho = abs(dx - dy)
                    corridor_open = True
                    east_bias = min(dx, dy) * 3.0
                    max_adv = 10
                else:
                    continue

                if advance <= 0 or advance > max_adv:
                    continue
                if not corridor_open:
                    continue

                path = self.find_path(grid, start_tile, (c, r))
                if not path or len(path) <= 1 or len(path) > 13:
                    continue

                # Score: maximize advance in desired direction, prefer balanced diagonal / open corridors
                score = (advance * 10.0) - (ortho * 1.2) - (len(path) * 0.3) + east_bias
                if score > best_score:
                    best_score = score
                    best_candidate = (c, r)

        if best_candidate is not None:
            return best_candidate[0], best_candidate[1], f"EXPLORE_{dir_upper}"

        if (target_c, target_r) != start_tile:
            return target_c, target_r, f"EXPLORE_{dir_upper}"

        # Dead-end corner fallback: If the requested direction is 100% blocked (zero reachable progress),
        # try component or opposite open directions so Red routes out of the corner instead of standing still!
        fallback_order = {
            "SOUTHEAST": ["SOUTH", "EAST", "NORTH", "WEST", "NORTHWEST"],
            "SOUTHWEST": ["SOUTH", "WEST", "NORTH", "EAST", "NORTHEAST"],
            "NORTHEAST": ["NORTH", "EAST", "WEST", "SOUTH", "SOUTHWEST"],
            "NORTHWEST": ["NORTH", "WEST", "EAST", "SOUTH", "SOUTHEAST"],
            "SOUTH": ["SOUTHEAST", "SOUTHWEST", "EAST", "WEST", "NORTH"],
            "EAST": ["NORTHEAST", "SOUTHEAST", "NORTH", "SOUTH", "WEST"],
            "NORTH": ["NORTHEAST", "NORTHWEST", "EAST", "WEST", "SOUTH"],
            "WEST": ["NORTHWEST", "SOUTHWEST", "NORTH", "SOUTH", "EAST"],
        }.get(dir_upper, ["NORTH", "WEST", "SOUTH", "EAST"])

        for alt_dir in fallback_order:
            best_alt = None
            best_alt_adv = 0
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    if (c, r) == start_tile or grid[r, c] != CELL_WALKABLE:
                        continue
                    adv = 0
                    if alt_dir == "NORTH" and r <= sr - 2:
                        adv = (sr - r) * 10 - abs(c - sc)
                    elif alt_dir == "WEST" and c <= sc - 2:
                        adv = (sc - c) * 10 - abs(r - sr)
                    elif alt_dir == "SOUTH" and r >= sr + 2:
                        adv = (r - sr) * 10 - abs(c - sc)
                    elif alt_dir == "EAST" and c >= sc + 2:
                        adv = (c - sc) * 10 - abs(r - sr)
                    if adv > best_alt_adv:
                        p = self.find_path(grid, start_tile, (c, r))
                        if p and 2 <= len(p) <= 9:
                            best_alt_adv = adv
                            best_alt = (c, r)
            if best_alt is not None:
                return best_alt[0], best_alt[1], f"EXPLORE_{alt_dir}"

        return target_c, target_r, f"EXPLORE_{dir_upper}"

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
                    x1, y1, x2, y2 = tile_bounds(c, r)
                    if val != CELL_WALKABLE:
                        draw.rectangle([x1, y1, x2, y2], outline=(100, 30, 130), width=1)

            # 2. Draw planned path line
            if len(path) > 1:
                pixel_points = [
                    tile_to_pixel_center(col, row)
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
            sx, sy = tile_to_pixel_center(sc, sr)
            draw.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 255, 0), width=3)
            draw.text((max(0, sx - 20), max(0, sy - 25)), "START", fill=(0, 255, 0))

            # 4. Draw END marker (Gold target box)
            gc, gr = goal_tile
            gx, gy = tile_to_pixel_center(gc, gr)
            draw.rectangle([gx - 12, gy - 12, gx + 12, gy + 12], outline=(255, 200, 0), width=3)
            draw.text((max(0, gx - 32), max(0, gy - 25)), f"END: {target_name}", fill=(255, 200, 0))

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
        # Immediate UI screen / dialogue advance handling (regardless of player detection)
        if scene.area_type == "NAMING_SCREEN":
            naming_btns = ["B", "START", "A"]
            if self.input_controller:
                for b in naming_btns:
                    self.input_controller.press_button(b)
                    time.sleep(0.25)
            return {
                "success": True,
                "target": "CONFIRM_NAMING_SCREEN",
                "path_length": 0,
                "planned_buttons": naming_btns,
                "executed_buttons": naming_btns if self.input_controller else [],
                "interrupted": False,
                "interrupt_reason": None,
                "status": "CONFIRMED_DEFAULT_NAME"
            }

        if scene.area_type == "TRAINER_CARD":
            if self.input_controller:
                self.input_controller.press_button("B")
            return {
                "success": True,
                "target": "CLOSE_TRAINER_CARD",
                "path_length": 0,
                "planned_buttons": ["B"],
                "executed_buttons": ["B"] if self.input_controller else [],
                "interrupted": False,
                "interrupt_reason": None,
                "status": "CLOSED_TRAINER_CARD"
            }

        if intent == "ADVANCE_DIALOGUE" or scene.area_type == "TITLE_SCREEN":
            if self.input_controller:
                self.input_controller.press_button("A")
            return {
                "success": True,
                "target": "DIALOGUE_ADVANCE",
                "path_length": 0,
                "planned_buttons": ["A"],
                "executed_buttons": ["A"] if self.input_controller else [],
                "interrupted": False,
                "interrupt_reason": None,
                "status": "ADVANCED_DIALOGUE"
            }

        p = scene.player
        if not p.detected:
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

        # If targeting an NPC or interactive OBJECT (e.g., STARTER_TABLE, PROFESSOR_OAK, MOM, MART_CLERK, PC),
        # ensure Red turns toward the POI tile at the end of the path and presses 'A' to interact!
        matched_poi = next((poi for poi in scene.pois if poi.name == target_name), None)
        if (
            matched_poi is not None
            and matched_poi.category in ("NPC", "OBJECT")
            and matched_poi.name != "EXITED_HOUSE_DOOR"
            and (matched_poi.tile_col, matched_poi.tile_row) != matched_poi.interaction_tile
            and path
        ):
            end_c, end_r = path[-1]
            pc, pr = matched_poi.tile_col, matched_poi.tile_row
            req_facing = None
            if pc < end_c:
                req_facing = "LEFT"
            elif pc > end_c:
                req_facing = "RIGHT"
            elif pr < end_r:
                req_facing = "UP"
            elif pr > end_r:
                req_facing = "DOWN"

            final_facing = buttons[-1] if buttons else p.facing
            if req_facing and final_facing != req_facing:
                buttons.append(req_facing)
            buttons.append("A")
        elif (
            matched_poi is not None
            and matched_poi.category == "WARP"
            and scene.area_type == "OVERWORLD"
            and matched_poi.tile_row == 0
            and path
        ):
            # Top-edge Overworld doorway (e.g., Oak's Lab door in y=0..16 directly North of Row 0):
            # After reaching (col, 0), turn UP and step UP into the doorway!
            final_facing = buttons[-1] if buttons else p.facing
            if final_facing != "UP":
                buttons.append("UP")
            buttons.append("UP")
        elif (
            matched_poi is not None
            and matched_poi.name in ("FRONT_DOOR", "MART_EXIT", "CENTER_EXIT")
            and start_tile != (matched_poi.tile_col, matched_poi.tile_row)
            and path
        ):
            # Indoor south welcome mat: after walking onto the mat, step DOWN once more to trigger the exit warp!
            final_facing = buttons[-1] if buttons else p.facing
            if final_facing != "DOWN":
                buttons.append("DOWN")
            buttons.append("DOWN")

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
        is_warp_target = (
            (matched_poi is not None and matched_poi.category == "WARP")
            or intent in ("ENTER_BUILDING", "EXIT_ROOM")
            or (0 <= goal_r < GRID_ROWS and 0 <= goal_c < GRID_COLS and scene.collision_grid[goal_r, goal_c] == CELL_WARP)
        )

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

            # If we stepped into a doorway/warp or triggered a scene warp, wait for the GBA door-sliding
            # animation + fade-to-black + interior room load (~1.2s) before capturing the next turn's frame!
            if (interrupted and interrupt_reason == "SCENE_WARP_OR_BATTLE") or is_warp_target:
                time.sleep(1.2)
                if self.screen_capture:
                    _, post_warp_thumb = self.screen_capture.capture_frame()
                    if post_warp_thumb is not None and float(post_warp_thumb.mean()) < 25.0:
                        time.sleep(0.6)

        return {
            "success": True,
            "target": target_name,
            "path_length": len(path),
            "planned_buttons": buttons,
            "executed_buttons": executed_buttons,
            "interrupted": interrupted,
            "interrupt_reason": interrupt_reason
        }
