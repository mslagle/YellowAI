"""
Deterministic Battle Controller for YellowAI.
Executes high-level battle strategies (FIGHT move_slot, BAG item, SWITCH_POKEMON, RUN)
by translating strategic choices into exact GBA battle cursor sequences without LLM hallucination.
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
from emulator.input_controller import InputController


class BattleController:
    """
    Translates high-level battle commands from the strategic LLM into deterministic GBA button sequences.
    """

    def __init__(self, input_controller: Optional[InputController] = None, button_delay: float = 0.18):
        self.input_controller = input_controller
        self.button_delay = button_delay

    def plan_battle_action(
        self,
        battle_action: Optional[str] = "FIGHT",
        move_slot: Optional[int] = 1,
        target_item: Optional[str] = None,
        switch_target_slot: Optional[int] = None,
        frame: Optional[Image.Image] = None
    ) -> List[str]:
        """
        Translates strategic intent into GBA button sequence.
        GBA Battle Menu:
        - Default cursor is at FIGHT (Top-Left).
        - Top-Right: BAG
        - Bottom-Left: POKÉMON
        - Bottom-Right: RUN
        """
        action = (battle_action or "FIGHT").upper().strip()

        # Check battle UI state if frame is provided:
        # In day or night mode:
        # Command box is at x=250..470, y=235..310 (Day fill: white/cream RGB > 200; Night fill: dark teal/gray RGB 70..130)
        # Move select sub-menu expands move options across left box (Day fill: white RGB > 200; Night fill: dark teal/gray)
        in_move_select = False
        has_command_box = False
        if frame is not None:
            try:
                arr = np.array(frame.resize((480, 320)).convert("RGB"))
                right_menu = arr[235:310, 250:470]
                left_box = arr[235:310, 16:300]
                
                # Command box fill detection (Day or Night)
                right_fill_day = float(np.mean((right_menu[:, :, 0] > 200) & (right_menu[:, :, 1] > 200) & (right_menu[:, :, 2] > 200)))
                right_fill_night = float(np.mean(
                    (right_menu[:, :, 0] > 65) & (right_menu[:, :, 0] < 135) &
                    (right_menu[:, :, 1] > 65) & (right_menu[:, :, 1] < 135) &
                    (np.abs(right_menu[:, :, 0].astype(int) - right_menu[:, :, 1].astype(int)) < 18)
                ))
                has_command_box = (right_fill_day >= 0.35 or right_fill_night >= 0.45)

                left_fill_day = float(np.mean((left_box[:, :, 0] > 200) & (left_box[:, :, 1] > 200) & (left_box[:, :, 2] > 200)))
                left_fill_night = float(np.mean(
                    (left_box[:, :, 0] > 65) & (left_box[:, :, 0] < 135) &
                    (left_box[:, :, 1] > 65) & (left_box[:, :, 1] < 135) &
                    (np.abs(left_box[:, :, 0].astype(int) - left_box[:, :, 1].astype(int)) < 18)
                ))
                if (left_fill_day >= 0.45 or left_fill_night >= 0.55) and not has_command_box:
                    in_move_select = True

                # If neither the 2x2 command box nor move selection is present, a full-screen battle dialogue banner is showing
                if not has_command_box and not in_move_select:
                    return ["A"]
            except Exception:
                pass

        # Detect current cursor position in 2x2 command box if not in move selection
        current_cursor = "FIGHT"
        if frame is not None and not in_move_select:
            try:
                arr = np.array(frame.resize((480, 320)).convert("RGB"))
                slots = {
                    "FIGHT": (252, 246, 268, 270),
                    "BAG": (368, 246, 382, 270),
                    "POKEMON": (252, 280, 268, 304),
                    "RUN": (368, 280, 382, 304),
                }
                scores = {}
                for name, (x1, y1, x2, y2) in slots.items():
                    crop = arr[y1:y2, x1:x2]
                    dark = int(np.sum((crop[:, :, 0] < 110) & (crop[:, :, 1] < 110) & (crop[:, :, 2] < 110)))
                    scores[name] = dark
                best_slot = max(scores, key=scores.get)
                if scores[best_slot] >= 30:
                    current_cursor = best_slot
            except Exception:
                pass

        # Helper to compute navigation from current cursor to target command
        # Layout:
        # FIGHT (0,0)   | BAG (1,0)
        # POKEMON (0,1) | RUN (1,1)
        coords = {
            "FIGHT": (0, 0),
            "BAG": (1, 0),
            "POKEMON": (0, 1),
            "RUN": (1, 1),
        }

        def nav_to_command(target_cmd: str) -> List[str]:
            cur_x, cur_y = coords.get(current_cursor, (0, 0))
            tgt_x, tgt_y = coords[target_cmd]
            seq = []
            if tgt_x > cur_x:
                seq.append("RIGHT")
            elif tgt_x < cur_x:
                seq.append("LEFT")
            if tgt_y > cur_y:
                seq.append("DOWN")
            elif tgt_y < cur_y:
                seq.append("UP")
            return seq

        # 1. RUN from battle
        if action == "RUN":
            if in_move_select:
                return ["B", "DOWN", "RIGHT", "A"]
            return nav_to_command("RUN") + ["A"]

        # 2. BAG / Use Item (Poké Ball or Potion)
        if action == "BAG":
            if in_move_select:
                return ["B", "RIGHT", "A", "A"]
            return nav_to_command("BAG") + ["A", "A"]

        # 3. SWITCH POKÉMON
        if action in ("SWITCH", "SWITCH_POKEMON", "POKEMON"):
            slot = switch_target_slot or 2
            seq = ["B"] if in_move_select else []
            seq.extend(nav_to_command("POKEMON"))
            seq.append("A")
            steps_down = max(0, min(5, slot - 1))
            seq.extend(["DOWN"] * steps_down)
            seq.extend(["A", "A"])
            return seq

        # 4. FIGHT / Attack Move Selection
        slot = move_slot or 1
        prefix = [] if in_move_select else (nav_to_command("FIGHT") + ["A"])
        if slot == 1:
            return prefix + ["A"]
        elif slot == 2:
            return prefix + ["RIGHT", "A"]
        elif slot == 3:
            return prefix + ["DOWN", "A"]
        elif slot == 4:
            return prefix + ["DOWN", "RIGHT", "A"]

        return prefix + ["A"]

    def execute_battle_strategy(
        self,
        battle_action: Optional[str] = "FIGHT",
        move_slot: Optional[int] = 1,
        target_item: Optional[str] = None,
        switch_target_slot: Optional[int] = None,
        frame: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """Plans and executes the battle button sequence."""
        buttons = self.plan_battle_action(
            battle_action=battle_action,
            move_slot=move_slot,
            target_item=target_item,
            switch_target_slot=switch_target_slot,
            frame=frame
        )

        executed: List[str] = []
        if self.input_controller:
            for btn in buttons:
                self.input_controller.press_button(btn)
                executed.append(btn)
                time.sleep(self.button_delay)

        return {
            "success": True,
            "battle_action": battle_action,
            "buttons": buttons,
            "executed": executed
        }
