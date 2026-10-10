"""
Anti-stuck watchdog and movement tracking engine for YellowAI.
Tracks recent actions and detects when the game screen remains motionless across turns
or when the character gets trapped in an alternating ping-pong loop (e.g. UP/DOWN doorway oscillation).
Provides Gemini with recent action history, collision alerts, and oscillation warnings so the AI
can dynamically adapt its direction rather than relying on brittle hardcoded overrides.
"""

from typing import Optional, List, Tuple
import numpy as np

OPPOSITE_DIRECTIONS = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT"
}


class ActionRecord:
    def __init__(self, turn_id: int, button: str, reasoning: str, outcome: str = "PENDING"):
        self.turn_id = turn_id
        self.button = button
        self.reasoning = reasoning
        self.outcome: str = outcome  # "MOVED", "BLOCKED", or "PENDING"

    def __repr__(self):
        return f"<ActionRecord Turn {self.turn_id}: [{self.button}] -> {self.outcome}>"


class AntiStuckWatchdog:
    def __init__(
        self,
        threshold_consecutive: int = 2,
        emergency_threshold: int = 8,
        diff_threshold: float = 3.0,
        max_history: int = 20,
        warp_diff_threshold: float = 60.0
    ):
        self.threshold_consecutive = threshold_consecutive
        self.emergency_threshold = emergency_threshold
        self.diff_threshold = diff_threshold
        self.max_history = max_history
        self.warp_diff_threshold = warp_diff_threshold

        self.consecutive_stuck_turns = 0
        self.oscillation_detected = False
        self.oscillation_cycles = 0
        self.doorway_warp_turns = 0
        self.last_warp_direction: Optional[str] = None
        self.last_thumb: Optional[np.ndarray] = None
        self.history: List[ActionRecord] = []
        self.rescue_cooldown = 0

    def record_action(self, turn_id: int, button: str, reasoning: str, outcome: Optional[str] = None):
        """Records an action taken at the end of a turn, awaiting outcome evaluation on next frame."""
        record = ActionRecord(turn_id=turn_id, button=button, reasoning=reasoning, outcome=outcome or "PENDING")
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if outcome == "BLOCKED":
            self.consecutive_stuck_turns += 1
        elif outcome == "MOVED":
            self.consecutive_stuck_turns = 0

    def check_step_interrupt(
        self,
        button: str,
        prev_thumb: Optional[np.ndarray],
        curr_thumb: Optional[np.ndarray],
        is_menu_or_naming: bool = False,
        current_area: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates an immediate step within a multi-action batch.
        Returns (True, reason) if an interrupt condition is detected that should halt remaining actions,
        or (False, None) if movement is proceeding normally.
        """
        if prev_thumb is None or curr_thumb is None:
            return False, None

        btn = button.upper().strip()
        diff = float(np.mean(np.abs(curr_thumb.astype(float) - prev_thumb.astype(float))))
        mean_val = float(np.mean(curr_thumb))

        # 1. Doorway warp or battle encounter flash/fade
        # Upstairs rooms (bedroom, 2F, 3F, roof) do not have south doorway exits!
        is_upstairs = any(x in current_area.lower() for x in ["bedroom", "2f", "3f", "roof", "naming"])
        if not is_menu_or_naming and (diff >= self.warp_diff_threshold or mean_val < 12.0):
            if btn in OPPOSITE_DIRECTIONS and not is_upstairs:
                self.doorway_warp_turns = 2
                self.last_warp_direction = btn
            return True, f"SCENE_WARP_OR_BATTLE (diff={diff:.1f})"

        # 2. Collision / Wall hit on directional movement
        # On menus and keyboard naming screens, moving a cursor produces subtle changes (diff ~0.3 - 1.5).
        # A true wall bump or edge collision means zero visual difference (< 0.20).
        step_threshold = 0.20 if is_menu_or_naming else 0.35
        if btn in OPPOSITE_DIRECTIONS and diff < step_threshold:
            return True, f"COLLISION_BLOCKED (button={btn}, diff={diff:.2f})"

        return False, None

    def evaluate_frame(
        self,
        current_thumb: Optional[np.ndarray],
        is_menu_or_naming: bool = False,
        current_area: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates the newly captured frame against the previous frame.
        Updates the outcome of the most recent action record (MOVED vs BLOCKED).
        Detects doorway warp transitions, dialogue locks, and obstacle oscillations.
        """
        if current_thumb is None:
            return False, None

        if self.rescue_cooldown > 0:
            self.rescue_cooldown -= 1

        # 1. Evaluate pending action record from previous turn (if outcome was not already set)
        if self.history and self.history[-1].outcome == "PENDING":
            last_record = self.history[-1]
            if self.last_thumb is not None:
                diff = float(np.mean(np.abs(current_thumb.astype(float) - self.last_thumb.astype(float))))
                turn_thresh = 0.20 if is_menu_or_naming else self.diff_threshold
                if last_record.button.upper() == "WAIT":
                    last_record.outcome = "WAIT"
                elif diff < turn_thresh:
                    last_record.outcome = "BLOCKED"
                    self.consecutive_stuck_turns += 1
                else:
                    last_record.outcome = "MOVED"
                    self.consecutive_stuck_turns = 0

                    # Strategy 5: Detect doorway warp transition (major scene change or fade)
                    # Never trigger in upstairs rooms (bedroom, 2F) where no south doorway exits exist!
                    is_upstairs = any(x in current_area.lower() for x in ["bedroom", "2f", "3f", "roof", "naming"])
                    btn = last_record.button.upper()
                    if not is_upstairs and btn in OPPOSITE_DIRECTIONS and (diff >= self.warp_diff_threshold or np.mean(current_thumb) < 15.0):
                        self.doorway_warp_turns = 2
                        self.last_warp_direction = btn
            else:
                last_record.outcome = "FIRST FRAME"
                self.consecutive_stuck_turns = 0

        self.last_thumb = current_thumb

        # 2. Strategy 5: Doorway Warp Transition Alert (immediately after stepping through a door)
        if self.doorway_warp_turns > 0:
            self.doorway_warp_turns -= 1
            warp_dir = self.last_warp_direction or "DOWN"
            if warp_dir == "DOWN":
                alert = (
                    "[DOORWAY TRANSITION ALERT]: You JUST emerged outside through a doorway to the SOUTH!\n"
                    "- The entrance door back inside is directly behind you (NORTH)!\n"
                    "- DO NOT select 'EXPLORE_NORTH'! Walking north re-enters the doorway immediately!\n"
                    "- Choose 'EXPLORE_SOUTH' or 'EXPLORE_EAST' to clear the entrance threshold into open paths!"
                )
            else:  # UP
                alert = (
                    "[DOORWAY TRANSITION ALERT]: You JUST stepped inside through a doorway to the NORTH!\n"
                    "- The exit door back outside is directly behind you (SOUTH)!\n"
                    "- DO NOT select 'EXPLORE_SOUTH'! Walking south steps right back outside!\n"
                    "- Choose 'EXIT_ROOM', 'GO_TO_POI', or navigate toward your indoor objective!"
                )
            return True, alert

        # 3. Check for dialogue / sign stuck condition (repeated A or B without movement)
        is_dialog_stuck, dialog_alert = self.check_dialog_stuck(threshold=2)
        if is_dialog_stuck:
            return True, dialog_alert

        # 4. Check for motionless stuck condition (bumping walls or obstacles)
        if self.consecutive_stuck_turns >= self.threshold_consecutive:
            recent_attempts = [
                f"[{r.button}]" for r in self.history[-self.consecutive_stuck_turns:]
                if r.button.upper() != "WAIT"
            ]
            last_button = self.history[-1].button.upper() if self.history else "UP"

            alert = (
                f"[COLLISION ALERT]: Motionless for {self.consecutive_stuck_turns} turns bumping an obstacle/barrier ({', '.join(recent_attempts)}).\n"
                f"- If stuck on a sign, mailbox, or dialog, do NOT press A/B; try buttons outside of A and B (press UP to step away or press START)!\n"
                f"- If trapped in a 1-tile dead end (walls or trees on your sides), BACK OUT in the opposite direction (step back the way you entered) to escape the narrow corridor!\n"
                f"- DO NOT press [{last_button}] again (blocked)."
            )
            return True, alert

        # 4. Check for repeated A button / dialogue loop (e.g. talking to same NPC repeatedly)
        is_a_loop, a_loop_alert = self.check_repeated_action("A", threshold=3)
        if is_a_loop:
            return True, a_loop_alert

        # 5. Check for doorway / path oscillation loops (e.g. DOWN -> UP -> DOWN -> UP)
        is_osc, osc_alert = self.check_oscillation()
        if is_osc:
            self.oscillation_detected = True
            self.oscillation_cycles += 1
            return True, osc_alert
        else:
            self.oscillation_detected = False
            self.oscillation_cycles = 0

        # 6. Check for alternating A / B menu oscillation loops (entering and backing out of menu)
        is_menu_loop, menu_loop_alert = self.check_menu_ab_oscillation(threshold=4)
        if is_menu_loop:
            return True, menu_loop_alert

        return False, None

    def check_dialog_stuck(self, threshold: int = 2) -> Tuple[bool, Optional[str]]:
        """
        Detects when player is repeatedly pressing A or B without any screen movement.
        In Pokémon Recharged Yellow, certain inspection dialogues (like mailboxes and signs)
        cannot be dismissed by pressing A or B alone, but require moving away or pressing START.
        """
        eval_records = [r for r in self.history if r.outcome != "PENDING"]
        if len(eval_records) < threshold:
            return False, None

        recent = eval_records[-threshold:]
        if all(r.button.upper() in ["A", "B"] for r in recent) and all("BLOCKED" in r.outcome for r in recent):
            alert = (
                f"[DIALOGUE / SIGN STUCK ALERT]: Pressed A/B for {threshold} turns with NO screen movement!\n"
                f"- In this game, some dialog boxes and signs (like mailboxes) CANNOT be exited by pressing A or B!\n"
                f"- STOP pressing A and B! Try buttons outside of A and B: press a directional button (like UP to step away from the mailbox) or press START to break out!"
            )
            return True, alert

        return False, None

    def check_repeated_action(self, button_name: str, threshold: int = 3) -> Tuple[bool, Optional[str]]:
        """Detects repeated button presses (e.g. talking to the same NPC continuously)."""
        eval_records = [r for r in self.history if r.outcome != "PENDING"]
        if len(eval_records) < threshold:
            return False, None

        recent_btns = [r.button.upper() for r in eval_records[-threshold:]]
        if all(b == button_name.upper() for b in recent_btns):
            if button_name.upper() == "A":
                alert = (
                    f"[ACTION / DIALOGUE LOOP ALERT]: Pressed [A] {threshold}+ times without progress.\n"
                    f"- IN BATTLE: If 'There\'s no PP left for this move!' or you cannot attack, STOP pressing A!\n"
                    f"  Press A/B once to clear the textbox, then navigate with D-Pad (RIGHT or DOWN) to a DIFFERENT move slot with remaining PP, or switch Pokémon / RUN!\n"
                    f"- IN OVERWORLD: If NPC repeated dialogue or sign won't close, STOP pressing A! Step away with directional keys (e.g. UP) or press START."
                )
                return True, alert
        return False, None

    def check_menu_ab_oscillation(self, threshold: int = 4) -> Tuple[bool, Optional[str]]:
        """
        Detects alternating A and B button presses (e.g. entering a sub-menu with A and canceling with B repeatedly).
        Commonly occurs when trying to exit a shop or NPC menu without selecting 'Quit' or 'Cancel'.
        """
        eval_records = [r for r in self.history if r.outcome != "PENDING"]
        if len(eval_records) < threshold:
            return False, None

        recent = [r.button.upper() for r in eval_records[-threshold:]]
        if all(b in ["A", "B"] for b in recent):
            is_alternating = all(recent[i] != recent[i - 1] for i in range(1, len(recent)))
            if is_alternating:
                alert = (
                    f"[MENU OSCILLATION ALERT]: Alternating between [A] and [B] ({' -> '.join(recent)})!\n"
                    f"- You are opening a sub-menu with [A] and backing out with [B] repeatedly!\n"
                    f"- If on Poké Mart clerk menu (Buy / Sell / Quit): DO NOT press A on 'Buy'! DO NOT press A to 'advance dialogue'!\n"
                    f"- Move cursor DOWN twice to 'Quit' and press A to exit the conversation!"
                )
                return True, alert
        return False, None

    def check_oscillation(self) -> Tuple[bool, Optional[str]]:
        """Detects alternating ping-pong movements (both 1-step and 2-step doorway oscillations)."""
        dir_records = [
            r for r in self.history
            if r.button.upper() in OPPOSITE_DIRECTIONS and r.outcome != "PENDING"
        ]

        # 1. Check for multi-step / 2-step doorway oscillation (e.g. DOWN, DOWN <-> UP, UP)
        if len(dir_records) >= 4:
            recent4 = [r.button.upper() for r in dir_records[-4:]]
            if set(recent4) == {"UP", "DOWN"} and recent4.count("UP") >= 2 and recent4.count("DOWN") >= 2:
                alert = (
                    "[DOORWAY OSCILLATION ALERT]: Repeatedly entering and exiting a doorway (DOWN, DOWN <-> UP, UP)!\n"
                    "- DO NOT press UP! You are outside the door; pressing UP enters the building!\n"
                    "- If at Oak's Lab (southeast building), THIS IS NOT DAISY'S HOUSE! Daisy does not live here!\n"
                    "- To reach Daisy's house (Town Map): walk LEFT around the fence, walk UP through central Pallet Town, and enter the NORTHEAST house (top-right)!"
                )
                return True, alert
            elif set(recent4) == {"LEFT", "RIGHT"} and recent4.count("LEFT") >= 2 and recent4.count("RIGHT") >= 2:
                alert = (
                    "[HORIZONTAL OSCILLATION ALERT]: Pacing back and forth [LEFT] <-> [RIGHT] below an obstacle/ledge!\n"
                    "- STOP pacing left and right! Do NOT run into boundary trees or continuous ledges.\n"
                    "- If at 1st ledge (near signpost): gap is immediately LEFT of signpost -> walk UP.\n"
                    "- If at 2nd ledge with pine trees on your right: you are in a grass pocket! Step DOWN (south) back to the dirt road, walk EAST (RIGHT) past the 3 pine trees, then walk UP!"
                )
                return True, alert

        # Check broader 6-step horizontal sweep (e.g. 3x LEFT then 3x RIGHT past an opening)
        if len(dir_records) >= 6:
            recent6 = [r.button.upper() for r in dir_records[-6:]]
            if set(recent6) == {"LEFT", "RIGHT"} and recent6.count("LEFT") >= 2 and recent6.count("RIGHT") >= 2:
                alert = (
                    "[HORIZONTAL SWEEP ALERT]: Sweeping back and forth below an obstacle or ledge!\n"
                    "- STOP oscillating east and west! You are caught in a cul-de-sac below the trees/ledge!\n"
                    "- Choose 'EXPLORE_SOUTH' to clear the obstacles and return to the main path, or target a specific POI!"
                )
                return True, alert

        # 2. Check for 1-step oscillation (e.g. DOWN <-> UP <-> DOWN)
        if len(dir_records) >= 3:
            r3 = dir_records[-3].button.upper()
            r2 = dir_records[-2].button.upper()
            r1 = dir_records[-1].button.upper()

            if r2 == OPPOSITE_DIRECTIONS.get(r3) and r1 == r3:
                doorway_hint = ""
                if {r2, r1} == {"UP", "DOWN"}:
                    doorway_hint = (
                        "\n- DOORWAY LOOP: You are ping-ponging inside and outside the threshold!\n"
                        "- Do NOT alternate between entering and exiting! Step into the room or follow the road!"
                    )
                    ortho_hint = "Choose an orthogonal direction ('EXPLORE_EAST' or 'EXPLORE_WEST') or 'ENTER_BUILDING' to break the loop."
                else:  # {"LEFT", "RIGHT"}
                    doorway_hint = (
                        "\n- If blocked above by a ledge or trees: you are trapped in a dead end!\n"
                        "- Choose 'EXPLORE_SOUTH' to return to the open path/road!"
                    )
                    ortho_hint = "Choose an orthogonal direction ('EXPLORE_NORTH' or 'EXPLORE_SOUTH') to break the loop."

                alert = (
                    f"[OSCILLATION ALERT]: Reversal loop detected between recent movements.\n"
                    f"- Break out of this back-and-forth pattern.{doorway_hint}\n"
                    f"- {ortho_hint}"
                )
                return True, alert

        return False, None

    def is_oscillating(self) -> bool:
        """Returns True if the system is currently in a doorway or path oscillation loop."""
        return self.oscillation_detected

    def get_last_successful_direction(self) -> Optional[str]:
        """Returns the most recent directional action that successfully resulted in movement."""
        for r in reversed(self.history):
            b = r.button.upper()
            if b in OPPOSITE_DIRECTIONS and (r.outcome == "MOVED" or "[SUCCESS" in r.outcome):
                return b
        return None

    def is_emergency_stuck(self) -> bool:
        """Triggers emergency recovery (e.g. spamming B) only if stuck for an extreme duration."""
        return self.consecutive_stuck_turns >= self.emergency_threshold

    def get_action_history_prompt(self) -> str:
        """Formats recent action attempts and their outcomes concisely."""
        evaluated_records = [r for r in self.history if r.outcome != "PENDING"]
        if not evaluated_records:
            return ""

        lines = ["[Recent Actions]:"]
        for r in evaluated_records[-4:]:
            lines.append(f"- Turn {r.turn_id}: Intent=[{r.button}] -> {r.outcome} | \"{r.reasoning}\"")

        return "\n".join(lines)

    def is_rescue_needed(
        self,
        rescue_threshold: int = 5,
        oscillation_threshold: int = 3
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects whether a last-ditch effort / tactical rescue consultation is needed.
        Triggered when:
        1. Unable to move: consecutive_stuck_turns >= rescue_threshold (e.g. 5+ turns blocked).
        2. Lost / caught in stubborn loops: persistent oscillation_cycles >= oscillation_threshold.
        3. Dialogue/sign lock: repeated A/B presses for 4+ turns with zero movement.
        """
        if self.rescue_cooldown > 0:
            return False, None

        if self.consecutive_stuck_turns >= rescue_threshold:
            return True, f"Unable to move! Character has been blocked against an obstacle for {self.consecutive_stuck_turns} consecutive turns."

        if self.oscillation_cycles >= oscillation_threshold:
            return True, f"Lost in stubborn oscillation loop for {self.oscillation_cycles} cycles."

        # Check persistent dialogue or inspection screen lock
        eval_records = [r for r in self.history if r.outcome != "PENDING"]
        if len(eval_records) >= 4:
            recent4 = eval_records[-4:]
            if all(r.button.upper() in ["A", "B"] for r in recent4) and all("BLOCKED" in r.outcome for r in recent4):
                return True, "Trapped on unresponsive dialogue or inspection sign for 4+ turns without movement."

        # Check persistent alternating A/B menu oscillation (entering and canceling sub-menu)
        if len(eval_records) >= 6:
            recent6 = [r.button.upper() for r in eval_records[-6:]]
            if all(b in ["A", "B"] for b in recent6) and all(recent6[i] != recent6[i - 1] for i in range(1, 6)):
                return True, "Trapped in an alternating [A/B] menu oscillation loop without selecting Quit or Exit."

        return False, None

    def notify_rescue_executed(self, cooldown_turns: int = 6):
        """Called after a tactical rescue consultation has taken place to reset stuck state and set cooldown."""
        self.rescue_cooldown = cooldown_turns
        self.consecutive_stuck_turns = 0
        self.oscillation_cycles = 0
        self.oscillation_detected = False

    def reset(self):
        """Resets the watchdog state."""
        self.consecutive_stuck_turns = 0
        self.oscillation_detected = False
        self.doorway_warp_turns = 0
        self.last_warp_direction = None
        self.last_thumb = None
        self.history.clear()
