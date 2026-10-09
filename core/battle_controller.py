"""
Deterministic Battle Controller for YellowAI.
Executes high-level battle strategies (FIGHT move_slot, BAG item, SWITCH_POKEMON, RUN)
by translating strategic choices into exact GBA battle cursor sequences without LLM hallucination.
"""

from typing import List, Optional, Dict, Any
from emulator.input_controller import InputController


class BattleController:
    """
    Translates high-level battle commands from the strategic LLM into deterministic GBA button sequences.
    """

    def __init__(self, input_controller: Optional[InputController] = None):
        self.input_controller = input_controller

    def plan_battle_action(
        self,
        battle_action: Optional[str] = "FIGHT",
        move_slot: Optional[int] = 1,
        target_item: Optional[str] = None,
        switch_target_slot: Optional[int] = None
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

        # 1. RUN from battle
        if action == "RUN":
            # Cursor from FIGHT -> DOWN to POKEMON -> RIGHT to RUN -> A to confirm
            return ["DOWN", "RIGHT", "A"]

        # 2. BAG / Use Item (Poké Ball or Potion)
        if action == "BAG":
            # Cursor from FIGHT -> RIGHT to BAG -> A to open -> A to use selected item
            return ["RIGHT", "A", "A"]

        # 3. SWITCH POKÉMON
        if action in ("SWITCH", "SWITCH_POKEMON", "POKEMON"):
            slot = switch_target_slot or 2
            # Cursor from FIGHT -> DOWN to POKEMON -> A to open party
            seq = ["DOWN", "A"]
            # Move cursor to target slot (Slot 2 is 1 down, Slot 3 is 2 down, etc.)
            steps_down = max(0, min(5, slot - 1))
            seq.extend(["DOWN"] * steps_down)
            seq.extend(["A", "A"])  # Select and confirm switch
            return seq

        # 4. FIGHT / Attack Move Selection
        slot = move_slot or 1
        if slot == 1:
            # Slot 1: Top-Left move
            return ["A", "A"]
        elif slot == 2:
            # Slot 2: Top-Right move
            return ["A", "RIGHT", "A"]
        elif slot == 3:
            # Slot 3: Bottom-Left move
            return ["A", "DOWN", "A"]
        elif slot == 4:
            # Slot 4: Bottom-Right move
            return ["A", "DOWN", "RIGHT", "A"]

        return ["A", "A"]

    def execute_battle_strategy(
        self,
        battle_action: Optional[str] = "FIGHT",
        move_slot: Optional[int] = 1,
        target_item: Optional[str] = None,
        switch_target_slot: Optional[int] = None
    ) -> Dict[str, Any]:
        """Plans and executes the battle button sequence."""
        buttons = self.plan_battle_action(
            battle_action=battle_action,
            move_slot=move_slot,
            target_item=target_item,
            switch_target_slot=switch_target_slot
        )

        executed: List[str] = []
        if self.input_controller:
            for btn in buttons:
                self.input_controller.press_button(btn)
                executed.append(btn)

        return {
            "success": True,
            "battle_action": battle_action,
            "buttons": buttons,
            "executed": executed
        }
