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
        # 1) Battle dialogue / text box (orange/teal full-width box, neither 2x2 command box nor move list is white) -> press ["A"]
        # 2) Move selection sub-menu already open (left box x=16..300, y=235..310 is bright white)
        in_move_select = False
        if frame is not None:
            try:
                arr = np.array(frame.resize((480, 320)).convert("RGB"))
                right_menu = arr[235:310, 250:470]
                left_box = arr[235:310, 16:300]
                right_white = float(np.mean((right_menu[:, :, 0] > 230) & (right_menu[:, :, 1] > 230) & (right_menu[:, :, 2] > 230)))
                left_white = float(np.mean((left_box[:, :, 0] > 230) & (left_box[:, :, 1] > 230) & (left_box[:, :, 2] > 230)))
                if right_white < 0.25 and left_white < 0.25:
                    # Battle text / narrative banner is active (e.g., "Wild PIDGEY appeared!", "Prof. Oak used Poké Ball!")
                    return ["A"]
                if left_white >= 0.45:
                    in_move_select = True
            except Exception:
                pass

        # 1. RUN from battle
        if action == "RUN":
            if in_move_select:
                return ["B", "DOWN", "RIGHT", "A"]
            # Cursor from FIGHT -> DOWN to POKEMON -> RIGHT to RUN -> A to confirm
            return ["DOWN", "RIGHT", "A"]

        # 2. BAG / Use Item (Poké Ball or Potion)
        if action == "BAG":
            if in_move_select:
                return ["B", "RIGHT", "A", "A"]
            # Cursor from FIGHT -> RIGHT to BAG -> A to open -> A to use selected item
            return ["RIGHT", "A", "A"]

        # 3. SWITCH POKÉMON
        if action in ("SWITCH", "SWITCH_POKEMON", "POKEMON"):
            slot = switch_target_slot or 2
            seq = ["B", "DOWN", "A"] if in_move_select else ["DOWN", "A"]
            steps_down = max(0, min(5, slot - 1))
            seq.extend(["DOWN"] * steps_down)
            seq.extend(["A", "A"])
            return seq

        # 4. FIGHT / Attack Move Selection
        slot = move_slot or 1
        prefix = [] if in_move_select else ["A"]
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
