"""
Game loop orchestrator for YellowAI.
Coordinates screen capture, Gemini 3.5 Flash decisions, hierarchical memory, input injection, and TTS commentary.
STRICT INVARIANT: Gemini API is NEVER called unless mGBA emulator is actively running and verified visible.
"""

import time
import json
import sys
from pathlib import Path
from typing import Optional, List

from config import (
    BUTTON_HOLD_DURATION,
    STEP_SETTLE_DELAY,
    TURN_SETTLE_DELAY,
    ANTI_STUCK_CONSECUTIVE_THRESHOLD,
    OVERLAY_DIR,
    OVERLAY_PORT,
    ENABLE_OVERLAY_SERVER,
    OVERLAY_STATE_FILE,
    MEMORY_FILE,
    KEY_MAPPINGS,
    CAPTURE_TARGET_WIDTH,
    CAPTURE_TARGET_HEIGHT,
    EMULATOR_WINDOW_TITLE_KEYWORDS,
    GEMINI_THINKING_BUDGET,
    LAST_CAPTURE_FILE,
    PREVIOUS_CAPTURE_FILE,
    ENABLE_RESCUE_THINKING,
    RESCUE_THINKING_BUDGET,
    RESCUE_ENABLE_WEB_SEARCH,
    RESCUE_STUCK_THRESHOLD,
    RESCUE_COOLDOWN_TURNS,
    ENABLE_BATTLE_THINKING,
    BATTLE_THINKING_BUDGET,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_ID,
    BIONIC_BASE_URL,
    BIONIC_MODEL_ID
)
from emulator.screen_capture import ScreenCapture
from emulator.input_controller import InputController
from agent.gemini_player import GeminiPlayer
from agent.prompts import BATTLE_TACTICAL_GUIDANCE
from core.memory_manager import MemoryManager
from core.anti_stuck import AntiStuckWatchdog, OPPOSITE_DIRECTIONS
from core.poi_detector import POIDetector
from tts.audio_queue import AudioQueueManager
from overlay.server import OverlayServer

class GameLoop:
    def __init__(
        self,
        api_key: str = "",
        model_id: str = "gemini-3.5-flash",
        backend: str = LLM_BACKEND,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model_id: str = OLLAMA_MODEL_ID,
        bionic_base_url: str = BIONIC_BASE_URL,
        bionic_model_id: str = BIONIC_MODEL_ID,
        thinking_budget: int = GEMINI_THINKING_BUDGET,
        turn_settle_delay: float = TURN_SETTLE_DELAY,
        step_settle_delay: float = STEP_SETTLE_DELAY,
        enable_tts: bool = True
    ):
        self.turn_id = 0
        self.is_running = False
        self.backend = (backend or LLM_BACKEND).lower().strip()
        self.turn_settle_delay = turn_settle_delay
        self.step_settle_delay = step_settle_delay

        # Initialize subsystems
        self.screen_capture = ScreenCapture(
            target_width=CAPTURE_TARGET_WIDTH,
            target_height=CAPTURE_TARGET_HEIGHT,
            window_keywords=EMULATOR_WINDOW_TITLE_KEYWORDS
        )
        self.input_controller = InputController(
            key_mappings=KEY_MAPPINGS,
            hold_duration=BUTTON_HOLD_DURATION
        )
        self.memory_manager = MemoryManager(storage_path=MEMORY_FILE)
        
        # Select active AI player backend (Local Bionic vs Local Ollama vs Cloud Gemini)
        if self.backend in ["bionic", "lmstudio"]:
            from agent.bionic_player import BionicPlayer
            self.gemini_player = BionicPlayer(
                base_url=bionic_base_url,
                model_id=bionic_model_id
            )
        elif self.backend == "ollama":
            from agent.ollama_player import OllamaPlayer
            self.gemini_player = OllamaPlayer(
                base_url=ollama_base_url,
                model_id=ollama_model_id
            )
        else:
            self.gemini_player = GeminiPlayer(
                api_key=api_key,
                model_id=model_id,
                thinking_budget=thinking_budget
            )
        self.anti_stuck = AntiStuckWatchdog(threshold_consecutive=ANTI_STUCK_CONSECUTIVE_THRESHOLD)
        self.poi_detector = POIDetector(target_width=CAPTURE_TARGET_WIDTH, target_height=CAPTURE_TARGET_HEIGHT)
        
        self.enable_tts = enable_tts
        self.audio_queue = AudioQueueManager() if enable_tts else None
        
        self.overlay_server = OverlayServer(OVERLAY_DIR, OVERLAY_PORT) if ENABLE_OVERLAY_SERVER else None

        # Battle state tracking for tactical thinking on encounter/switch
        self.in_battle: bool = False
        self.current_opponent: Optional[str] = None
        self.battle_advisor_given_for_opponent: Optional[str] = None
        self.previous_frame: Optional[Image.Image] = None
        self.last_facing_direction: Optional[str] = None

    def _is_naming_screen(self, frame: Image.Image) -> bool:
        """Visual check for the GBA Gen 3 / Recharged Yellow naming keyboard screen in all modes (Upper/Lower/Symbols)."""
        try:
            import numpy as np
            resized = frame.resize((480, 320))
            arr = np.array(resized)
            # 1. Top banner: blue header band
            top_blue = np.mean((arr[5:20, 10:200, 0] < 50) & (arr[5:20, 10:200, 1] > 70) & (arr[5:20, 10:200, 2] > 180))
            # 2. White name card in upper half
            white_card = np.mean((arr[60:120, 120:380, 0] > 230) & (arr[60:120, 120:380, 1] > 230) & (arr[60:120, 120:380, 2] > 230))
            # 3. Yellow OK button at bottom-right
            ok_yellow = np.mean((arr[250:290, 390:460, 0] > 180) & (arr[250:290, 390:460, 1] > 170) & (arr[250:290, 390:460, 2] < 140))
            return bool(top_blue > 0.35 and white_card > 0.45 and ok_yellow > 0.05)
        except Exception:
            return False

    def _is_bedroom(self, frame: Image.Image) -> bool:
        """Detects if the player is in Red's 2F bedroom upstairs surrounded by black void."""
        try:
            resized = frame.resize((480, 320))
            arr = np.array(resized)
            black_bottom = np.mean(arr[200:, :, :] < 20) > 0.8
            upper = arr[:160, 160:480]
            tatami = (upper[:, :, 0] > 140) & (upper[:, :, 1] > 120) & (upper[:, :, 2] < 110)
            rug = (upper[:, :, 0] < 130) & (upper[:, :, 1] > 140) & (upper[:, :, 2] > 70)
            has_bedroom_floor = (np.mean(tatami) + np.mean(rug)) > 0.35
            return black_bottom and has_bedroom_floor
        except Exception:
            return False

    def _get_bedroom_advice(self, frame: Image.Image) -> str:
        """Provides precise visual grid advice to escape Red's bedroom to the stairs."""
        try:
            resized = frame.resize((480, 320))
            arr = np.array(resized)
            cap = (arr[:, :, 0] > 175) & (arr[:, :, 1] < 105) & (arr[:, :, 2] < 105)
            y_idx, x_idx = np.where(cap)
            if len(y_idx) == 0:
                return (
                    "[RED'S BEDROOM (2F) ADVISOR]:\n"
                    "- The bottom of the room is surrounded by BLACK VOID. DO NOT press DOWN!\n"
                    "- The stairs to 1F are in the FAR TOP-RIGHT corner. Walk RIGHT to Col 9, then walk UP into the stairs!"
                )
            py = int(np.median(y_idx))
            px = int(np.median(x_idx))
            col = max(0, min(9, (px - 160) // 32))
            row = max(0, min(5, py // 32))
            steps_right = max(0, 9 - col)

            is_west_void = np.mean(arr[py-10:py+10, max(0, px-40):max(1, px-15)] < 20) > 0.8
            is_south_void = np.mean(arr[min(319, py+15):min(320, py+35), max(0, px-10):min(480, px+10)] < 20) > 0.8

            lines = [
                "[RED'S BEDROOM (2F) NAVIGATION ADVISOR]:",
                f"- Player Position: Col {col}/9, Row {row}/5 (X={px}, Y={py}).",
                "- CRITICAL VOID BORDERS: This is an upstairs room surrounded by black void.",
            ]
            if is_west_void:
                lines.append("  * WEST VOID DEAD END: You are standing against the WEST WALL / BLACK VOID! DO NOT PRESS LEFT! There is NO door to the left!")
            if is_south_void:
                lines.append("  * SOUTH VOID DEAD END: You are standing against the SOUTH WALL / BLACK VOID! DO NOT PRESS DOWN! There is NO door to the south!")

            lines.extend([
                "- OBJECTIVE & EXIT: The ONLY exit downstairs to 1F is the STAIRCASE in the FAR TOP-RIGHT corner (Col 9)!",
                "  * Safety Railing: Cols 7-8 have a white staircase railing along the top that blocks moving UP.",
                "  * Stairs Landing: Walk EAST (RIGHT) across the room to Col 9, then walk NORTH (UP) to descend the stairs!"
            ])

            if col < 9:
                actions = ["RIGHT"] * min(steps_right, 4)
                lines.append(f"- Recommended Action: Walk RIGHT toward center/east of room -> actions = {actions}")
            else:
                lines.append("- Recommended Action: You have reached Col 9! Walk UP into the staircase landing -> actions = ['UP', 'UP']")
            return "\n".join(lines)
        except Exception:
            return "[RED'S BEDROOM (2F) ADVISOR]: Walk RIGHT to top-right corner (Col 9), then walk UP to go downstairs."

    def _detect_screen_mode(self, frame: Image.Image) -> str:
        """Lightweight screen mode hint to streamline Gemini prompt focus."""
        if self._is_naming_screen(frame):
            print("[Screen Mode Detected] Naming Keyboard Screen -> Waiting for user to complete naming.")
            return "NAMING SCREEN: User is entering character name manually. Return actions = ['WAIT']."

        # Check if in Red's upstairs bedroom
        if self._is_bedroom(frame):
            self.memory_manager.current_area = "Pallet Town (Red's Bedroom)"
            if "bedroom" not in (self.memory_manager.priority_quest or "").lower():
                self.memory_manager.set_priority_quest("Leave Red's bedroom, go downstairs, and talk to Mom")
            return self._get_bedroom_advice(frame)

        # If we just left the naming screen, transition back to bedroom/Pallet Town quest
        if self.memory_manager.current_area == "Intro: Character Naming Screen":
            print("[Game Transition] Naming completed! Entering Red's bedroom (2F).")
            self.memory_manager.current_area = "Pallet Town (Red's Bedroom)"
            self.memory_manager.set_priority_quest("Leave Red's bedroom, go downstairs, and talk to Mom")
            self.memory_manager.update_micro_step("Walk UP and RIGHT to staircase in top-right corner to go downstairs")

        micro = (self.memory_manager.micro_step or "").lower()
        area = self.memory_manager.current_area.lower()
        if self.in_battle or "battle" in micro or "fight" in micro or "wild" in micro or "catch" in micro:
            return "BATTLE MODE: Check move PP (PP xx/xx). NEVER select a 0-PP move! If 'There\'s no PP left for this move!' is on screen, press A/B to clear, then press RIGHT or DOWN to pick a DIFFERENT move slot, switch Pokémon, or RUN!"
        if "mart" in area or "buy" in micro or "shop" in micro or "clerk" in micro or "quit" in micro:
            return "SHOP/MENU MODE: If on clerk menu (Buy/Sell/Quit) or done shopping, press DOWN to highlight 'Quit' and press A to exit! DO NOT press A on 'Buy'!"

        # Extract real-time POIs via fast NumPy spatial detection
        poi_analysis = self.poi_detector.analyze_frame(frame, current_area=self.memory_manager.current_area)
        poi_prompt = self.poi_detector.format_poi_prompt(poi_analysis)
        return f"OVERWORLD NAVIGATION\n\n{poi_prompt}"

    def _export_overlay_state(self, action: str, thought: Optional[str] = None):
        """Exports the latest state to JSON for OBS browser source overlay."""
        state = {
            "turn": self.turn_id,
            "last_action": action,
            "latest_thought": thought or "",
            "priority_quest": self.memory_manager.priority_quest or "",
            "active_quest": self.memory_manager.priority_quest or "",
            "quests": self.memory_manager.quests,
            "current_area": self.memory_manager.current_area,
            "team": self.memory_manager.team,
            "supplies": self.memory_manager.supplies,
            "badges": self.memory_manager.badges,
            "micro_step": self.memory_manager.micro_step or "",
            "short_term_memory": self.memory_manager.short_term_memory,
            "medium_term_memory": self.memory_manager.medium_term_memory,
            "long_term_memory": self.memory_manager.long_term_memory,
            "knowledge_notes": self.memory_manager.knowledge_notes,
            "timestamp": time.time()
        }
        try:
            with open(OVERLAY_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[GameLoop] Error writing overlay state: {e}")

    def run(self, max_turns: Optional[int] = None) -> bool:
        """
        Executes the master game loop.
        Strict Guarantee: Gemini API is NEVER called unless mGBA is actively running and visible.
        """
        if self.backend in ["bionic", "lmstudio"]:
            backend_info = f"Local Bionic / LM Studio [{self.gemini_player.model_id}]"
        elif self.backend == "ollama":
            backend_info = f"Local Ollama [{self.gemini_player.model_id}]"
        else:
            backend_info = f"Gemini Flash [{self.gemini_player.model_id}]"
        print("=" * 65)
        print(" YellowAI: Pokémon Yellow Recharged Streamer")
        print(f" Active Backend: {backend_info}")
        print("=" * 65)

        # 1. HARD PRE-CHECK: Strictly verify emulator presence before starting
        print("[Safeguard] Checking for active mGBA emulator window...")
        if not self.screen_capture.is_window_alive():
            print("\n" + "!" * 65)
            print(" [SAFEGUARD BLOCKED] mGBA emulator is not running or not visible!")
            print(" Gemini API will NOT be called.")
            print(" Please launch mGBA with Pokémon Yellow Recharged loaded.")
            print(f" Target window keywords: {EMULATOR_WINDOW_TITLE_KEYWORDS}")
            print(" YellowAI is aborting execution immediately.")
            print("!" * 65 + "\n")
            return False

        print(f"[GameLoop] Successfully attached to emulator: '{self.screen_capture.target_title}' (HWND: {self.screen_capture.target_hwnd})")
        self.input_controller.set_target_hwnd(self.screen_capture.target_hwnd)

        # 2. Start background services only after emulator is verified
        if self.overlay_server:
            self.overlay_server.start()

        if self.audio_queue:
            self.audio_queue.start()

        self.is_running = True

        try:
            while self.is_running:
                # 3. PER-TURN SAFEGUARD CHECK: Ensure emulator didn't close mid-run
                if not self.screen_capture.is_window_alive():
                    print("\n" + "!" * 65)
                    print(" [SAFEGUARD TRIGGERED] Emulator window was closed or lost!")
                    print(" Gemini API call canceled. Stopping game loop immediately.")
                    print("!" * 65 + "\n")
                    break

                # 4. Capture Frame
                frame, thumb = self.screen_capture.capture_frame()
                if frame is None:
                    print("\n[SAFEGUARD TRIGGERED] Failed to capture frame from emulator.")
                    print("Gemini API call canceled. Stopping game loop.")
                    break

                # Save the latest screen capture to where logs are stored
                try:
                    frame.save(LAST_CAPTURE_FILE)
                except Exception:
                    pass

                self.turn_id += 1
                start_time = time.time()

                print(f"\n--- [TURN {self.turn_id}] ---")

                # 5. Evaluate movement from previous turn and build action history
                is_naming = self._is_naming_screen(frame)
                is_stuck, stuck_alert = self.anti_stuck.evaluate_frame(
                    thumb, is_menu_or_naming=is_naming, current_area=self.memory_manager.current_area
                )
                repeated_a, repeat_alert = self.anti_stuck.check_repeated_action("A", threshold=3)
                if repeated_a and repeat_alert:
                    stuck_alert = f"{stuck_alert}\n{repeat_alert}" if stuck_alert else repeat_alert
                    print(f"[Watchdog Notice] Repeated dialogue/action loop detected. Alerting Gemini.")

                action_history_prompt = self.anti_stuck.get_action_history_prompt()
                if is_stuck and stuck_alert:
                    print(f"[Watchdog Alert]: {stuck_alert}")

                # 6. Assemble memory context
                memory_prompt = self.memory_manager.get_memory_context_prompt()

                # Check if Last-Ditch Effort / Tactical Rescue is triggered
                is_rescue, rescue_reason = self.anti_stuck.is_rescue_needed(
                    rescue_threshold=RESCUE_STUCK_THRESHOLD
                )

                if ENABLE_RESCUE_THINKING and is_rescue:
                    print("\n" + "=" * 65)
                    print(" [LAST DITCH RESCUE TRIGGERED]")
                    print(f" Condition: {rescue_reason}")
                    print(f" Activating Medium Thinking Budget ({RESCUE_THINKING_BUDGET} tokens) + Web Search Grounding...")
                    print("=" * 65)

                    api_start = time.time()
                    rescue_result = self.gemini_player.consult_tactical_rescue(
                        image=frame,
                        current_area=self.memory_manager.current_area,
                        active_quest=self.memory_manager.active_quest or self.memory_manager.priority_quest or "Progress journey",
                        micro_step=self.memory_manager.micro_step or "Navigate forward",
                        action_history=action_history_prompt,
                        stuck_alert=stuck_alert or "",
                        rescue_reason=rescue_reason,
                        thinking_budget=RESCUE_THINKING_BUDGET,
                        enable_web_search=RESCUE_ENABLE_WEB_SEARCH,
                        emulator_verified=True
                    )
                    api_duration = time.time() - api_start

                    button = rescue_result.get("recommended_button", "B")
                    flat_actions = rescue_result.get("flat_actions", [button])
                    reasoning = f"Rescue: {rescue_result.get('situation_analysis', '')[:50]}"
                    thought = rescue_result.get("stream_thought", None)
                    cadence = rescue_result.get("cadence_tag", "TACTICAL_RESCUE")

                    clues = rescue_result.get("clues_found", "")
                    if clues:
                        print(f"[Rescue Internet Clues]: {clues}")
                        self.memory_manager.add_knowledge_note(clues.strip())
                    print(f"[Rescue Analysis]: {rescue_result.get('situation_analysis', '')}")
                    print(f"[Rescue Plan]: {rescue_result.get('tactical_plan', '')}")
                    plan_str = " -> ".join(flat_actions) if len(flat_actions) > 1 else button
                    print(f"Action Plan: [{plan_str}] | Reason: '{reasoning}' (Cadence: {cadence})")

                    # Update micro-step and short-term memory with tactical breakthrough
                    new_micro = rescue_result.get("updated_micro_step")
                    if new_micro and isinstance(new_micro, str):
                        self.memory_manager.update_micro_step(new_micro.strip())
                        print(f"[Micro-Step Updated via Rescue]: '{new_micro.strip()}'")

                    rescue_plan = rescue_result.get("tactical_plan", "")
                    if rescue_plan:
                        self.memory_manager.update_short_term(f"Tactical Rescue Plan: {rescue_plan}")

                    # Notify watchdog so cooldown is set and stuck counters reset
                    self.anti_stuck.notify_rescue_executed(cooldown_turns=RESCUE_COOLDOWN_TURNS)

                    decision = {
                        "button": button,
                        "flat_actions": flat_actions,
                        "reasoning": reasoning,
                        "thought": thought,
                        "cadence_tag": cadence
                    }
                else:
                    # 7. Call Gemini with action history, active quest, mode hint, and stuck alerts
                    mode_hint = self._detect_screen_mode(frame)

                    # Check for Battle Opening / Enemy Pokémon Switch Thinking Budget
                    trigger_battle_thinking = False
                    battle_advisor_prompt = ""
                    if ENABLE_BATTLE_THINKING and "BATTLE" in mode_hint:
                        if not self.in_battle or (self.current_opponent and self.current_opponent != self.battle_advisor_given_for_opponent):
                            trigger_battle_thinking = True
                            battle_advisor_prompt = BATTLE_TACTICAL_GUIDANCE
                            opp_label = self.current_opponent or "New Encounter"
                            print(f"\n[Battle Strategy Triggered] Allocating {BATTLE_THINKING_BUDGET} thinking tokens for {opp_label}!")

                    api_start = time.time()
                    decision = self.gemini_player.decide_action(
                        turn_id=self.turn_id,
                        image=frame,
                        previous_image=self.previous_frame,
                        memory_context=memory_prompt,
                        active_quest=self.memory_manager.active_quest,
                        action_history=action_history_prompt,
                        stuck_alert=stuck_alert or "",
                        mode_hint=mode_hint,
                        battle_thinking_budget=BATTLE_THINKING_BUDGET if trigger_battle_thinking else None,
                        battle_advisor_prompt=battle_advisor_prompt,
                        emulator_verified=True  # Verified alive and visible right above!
                    )
                    api_duration = time.time() - api_start

                    button = decision.get("button", "WAIT")
                    flat_actions = decision.get("flat_actions", [button])
                    reasoning = decision.get("reasoning", "")
                    thought = decision.get("thought", None)
                    cadence = decision.get("cadence_tag", "")

                    plan_str = " -> ".join(flat_actions) if len(flat_actions) > 1 else button
                    print(f"Action Plan: [{plan_str}] | Reason: '{reasoning}' (Cadence: {cadence})")

                # Update Battle Tracking & Enemy Changes
                opp = decision.get("opponent_pokemon")
                if opp and isinstance(opp, str) and opp.strip() and opp.strip().upper() != "NULL":
                    opp_clean = opp.strip()
                    if not self.in_battle or self.current_opponent != opp_clean:
                        print(f"[Battle Opponent Detected]: {opp_clean}")
                        self.in_battle = True
                        self.current_opponent = opp_clean
                    if trigger_battle_thinking:
                        self.battle_advisor_given_for_opponent = opp_clean
                elif "BATTLE" in mode_hint and not self.in_battle:
                    self.in_battle = True
                elif "OVERWORLD" in mode_hint and self.in_battle:
                    print("[Battle Concluded] Returning to Overworld navigation.")
                    self.in_battle = False
                    self.current_opponent = None
                    self.battle_advisor_given_for_opponent = None

                # Ingest Discovered POIs, Item Updates, and Party Changes
                poi = decision.get("discovered_poi")
                if poi and isinstance(poi, str) and poi.strip() and poi.strip().upper() != "NULL":
                    self.memory_manager.add_waypoint(self.memory_manager.current_area, poi.strip())

                items_up = decision.get("items_update")
                if items_up and isinstance(items_up, str) and items_up.strip() and items_up.strip().upper() != "NULL":
                    self.memory_manager.apply_items_update(items_up.strip())

                party_up = decision.get("party_update")
                if party_up and isinstance(party_up, str) and party_up.strip() and party_up.strip().upper() != "NULL":
                    self.memory_manager.apply_party_update(party_up.strip())

                # Process Multi-Quest Log Updates
                comp_q = decision.get("completed_quest")
                if comp_q and isinstance(comp_q, str) and comp_q.strip() and comp_q.strip().upper() != "NULL":
                    removed = self.memory_manager.complete_quest(comp_q.strip())
                    if removed:
                        print(f"[Quest Log] Completed & Removed: '{removed}'")
                        if self.memory_manager.priority_quest:
                            print(f"[Quest Log] New Top Priority: '{self.memory_manager.priority_quest}'")

                new_q = decision.get("new_quest")
                prio_q = decision.get("priority_quest")
                if new_q and isinstance(new_q, str) and new_q.strip() and new_q.strip().upper() != "NULL":
                    is_prio = (prio_q == new_q or not self.memory_manager.priority_quest)
                    self.memory_manager.add_quest(new_q.strip(), is_priority=is_prio)
                    print(f"[Quest Log] New Quest Added: '{new_q.strip()}' (Priority: {is_prio})")

                if prio_q and isinstance(prio_q, str) and prio_q.strip() and prio_q.strip().upper() != "NULL" and prio_q != new_q:
                    self.memory_manager.set_priority_quest(prio_q.strip())
                    print(f"[Quest Log] Priority Set: '{prio_q.strip()}'")

                # Process Micro-Step Updates
                micro_s = decision.get("micro_step")
                if micro_s and isinstance(micro_s, str) and micro_s.strip() and micro_s.strip().upper() != "NULL":
                    if self.memory_manager.micro_step != micro_s.strip():
                        self.memory_manager.update_micro_step(micro_s.strip())
                        print(f"[Micro-Step Updated]: '{micro_s.strip()}'")

                # Fallback support for legacy active_command field
                active_cmd = decision.get("active_command")
                if active_cmd and isinstance(active_cmd, str) and active_cmd.strip():
                    if active_cmd.strip().upper() == "COMPLETED":
                        if self.memory_manager.priority_quest:
                            self.memory_manager.complete_quest(self.memory_manager.priority_quest)
                    elif active_cmd.strip().upper() != "NULL":
                        self.memory_manager.add_quest(active_cmd.strip(), is_priority=True)

                # Event-driven memory sync on doorway transitions
                if self.anti_stuck.doorway_warp_turns == 2:
                    self.memory_manager.save()

                # 8. Emergency Menu Escape failsafe ONLY if stuck for extreme duration (8+ turns)
                if self.anti_stuck.is_emergency_stuck():
                    if self._is_naming_screen(frame):
                        print(f"[Watchdog Emergency] Stuck on Naming Screen after {self.anti_stuck.consecutive_stuck_turns} turns. Pressing START to confirm [OK]!")
                        flat_actions = ["START"]
                    else:
                        print(f"[Watchdog Emergency] Still stuck after {self.anti_stuck.consecutive_stuck_turns} turns. Triggering emergency escape!")
                        self.input_controller.emergency_menu_escape(3, try_move_away=True)
                        flat_actions = ["UP"]
                else:
                    flat_actions = decision.get("flat_actions", [button])

                if is_naming:
                    print("[Naming Screen] Waiting for user to complete character naming.")
                    flat_actions = ["WAIT"]
                elif not self.in_battle and flat_actions:
                    # Physical Void Guard: If character is against a black void wall and attempts to move into it,
                    # override immediately towards open center of room.
                    first_btn = flat_actions[0].upper().strip()
                    try:
                        arr_check = np.array(frame.resize((480, 320)))
                        cap_m = (arr_check[:, :, 0] > 175) & (arr_check[:, :, 1] < 105) & (arr_check[:, :, 2] < 105)
                        cy_i, cx_i = np.where(cap_m)
                        if len(cy_i) > 0:
                            py_i, px_i = int(np.median(cy_i)), int(np.median(cx_i))
                            is_left_void = np.mean(arr_check[py_i-10:py_i+10, max(0, px_i-40):max(1, px_i-15)] < 20) > 0.8
                            is_down_void = np.mean(arr_check[min(319, py_i+15):min(320, py_i+35), max(0, px_i-10):min(480, px_i+10)] < 20) > 0.8
                            if first_btn == "LEFT" and is_left_void:
                                print(f"[Void Guard] Blocked by West Black Void at X={px_i}! Overriding to RIGHT (moving toward center of room).")
                                flat_actions = ["RIGHT", "RIGHT"]
                            elif first_btn == "DOWN" and is_down_void:
                                print(f"[Void Guard] Blocked by South Black Void at Y={py_i}! Overriding to RIGHT (toward stairs).")
                                flat_actions = ["RIGHT", "RIGHT"]
                    except Exception:
                        pass

                    # Overworld Turning Mechanic:
                    # If Red is facing direction A and the first action is direction B,
                    # the first press only turns Red in place! A second press is required to actually move.
                    first_dir = flat_actions[0].upper().strip()
                    if first_dir in OPPOSITE_DIRECTIONS:
                        if self.last_facing_direction and self.last_facing_direction != first_dir:
                            # Prepend an extra turn press if the AI only sent a single press
                            if len(flat_actions) == 1 or flat_actions[1].upper().strip() != first_dir:
                                print(f"[Turning Assist] Changing facing from {self.last_facing_direction} to {first_dir}: prepending turn press.")
                                flat_actions = [first_dir] + flat_actions

                # 9. Multi-Action Batch Execution with Reactive Interrupt Watchdog
                executed_actions: List[str] = []
                interrupted_reason: Optional[str] = None
                current_step_thumb = thumb
                latest_frame = frame

                for step_idx, btn in enumerate(flat_actions, start=1):
                    if not self.screen_capture.is_window_alive():
                        print("\n[SAFEGUARD TRIGGERED] Emulator window was closed or lost mid-batch!")
                        self.is_running = False
                        break

                    self.input_controller.press_button(btn)
                    executed_actions.append(btn)

                    # Inter-step settle delay for character animation
                    time.sleep(self.step_settle_delay)

                    # Capture frame after step to verify outcome and watch for interrupts
                    step_frame, step_thumb = self.screen_capture.capture_frame()
                    if step_frame is not None and step_thumb is not None:
                        latest_frame = step_frame

                        if is_naming:
                            # Bypass collision interrupts completely on the naming screen
                            step_interrupted = False
                            interrupt_why = None
                            self.anti_stuck.record_action(self.turn_id, btn, reasoning, outcome="MOVED")
                        else:
                            # Check if this step should interrupt subsequent actions
                            step_interrupted, interrupt_why = self.anti_stuck.check_step_interrupt(
                                btn, current_step_thumb, step_thumb, is_menu_or_naming=False, current_area=self.memory_manager.current_area
                            )
                            step_outcome = "BLOCKED" if (step_interrupted and "COLLISION" in interrupt_why) else "MOVED"
                            self.anti_stuck.record_action(self.turn_id, btn, reasoning, outcome=step_outcome)

                            if step_interrupted and step_idx < len(flat_actions):
                                interrupted_reason = interrupt_why
                                print(f"[Interrupt Watchdog] Step {step_idx}/{len(flat_actions)} [{btn}] -> {interrupt_why}. Halting remaining batch steps.")
                                break

                        current_step_thumb = step_thumb
                    else:
                        self.anti_stuck.record_action(self.turn_id, btn, reasoning, outcome="MOVED")

                action_summary = " -> ".join(executed_actions) if executed_actions else "WAIT"
                if len(executed_actions) > 1:
                    print(f"[Batch Executed]: [{action_summary}] ({len(executed_actions)} steps)")

                # Update tracked character facing direction in overworld
                for act in reversed(executed_actions):
                    act_upper = act.upper().strip()
                    if act_upper in OPPOSITE_DIRECTIONS:
                        self.last_facing_direction = act_upper
                        break

                # Save both current and previous screen captures to disk
                try:
                    if self.previous_frame is not None:
                        self.previous_frame.save(PREVIOUS_CAPTURE_FILE)
                    latest_frame.save(LAST_CAPTURE_FILE)
                except Exception:
                    pass
                self.previous_frame = latest_frame

                # 10. Process Cadence Triggers
                if thought:
                    print(f"Thought (TTS): \"{thought}\"")
                    if self.audio_queue:
                        self.audio_queue.enqueue_thought(thought)

                if "short_term_memory" in decision:
                    new_stm = decision["short_term_memory"]
                    print(f"[Short-Term Memory Updated]: {new_stm}")
                    self.memory_manager.update_short_term(new_stm)

                if "medium_term_memory" in decision:
                    new_mtm = decision["medium_term_memory"]
                    print(f"[Medium-Term Memory Updated]: {new_mtm}")
                    self.memory_manager.update_medium_term(new_mtm)

                if "long_term_memory" in decision:
                    new_ltm = decision["long_term_memory"]
                    print(f"[Long-Term Memory Updated]: {new_ltm}")
                    self.memory_manager.update_long_term(new_ltm)

                # 11. Record History & Export Stream Overlay
                self.memory_manager.log_turn(
                    turn_id=self.turn_id,
                    action=action_summary,
                    thought=thought,
                    reasoning=reasoning
                )
                self._export_overlay_state(action=action_summary, thought=thought)

                # 12. Paced Delay between turns (allows GBA step animation to settle before next capture)
                wait_time = max(0.05, self.turn_settle_delay)
                time.sleep(wait_time)

                turn_duration = time.time() - start_time
                usage = decision.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", 0)
                cand_tok = usage.get("candidates_tokens", 0)
                tok_str = f" | Tokens: {prompt_tok} in / {cand_tok} out" if prompt_tok else ""
                print(f"[Turn {self.turn_id} Complete in {turn_duration:.2f}s] (API: {api_duration:.2f}s{tok_str} | Settle: {wait_time:.2f}s)")

                if max_turns and self.turn_id >= max_turns:
                    print(f"\n[GameLoop] Reached maximum requested turns ({max_turns}). Stopping.")
                    break

            return True

        except KeyboardInterrupt:
            print("\n[GameLoop] User interrupted execution (Ctrl+C).")
            return True
        finally:
            self.stop()

    def stop(self):
        """Stops the loop, overlay server, and releases audio queue."""
        self.is_running = False
        if self.audio_queue:
            self.audio_queue.stop()
        if self.overlay_server:
            self.overlay_server.stop()
        self.memory_manager.save()
        print("[GameLoop] Saved memory state and exited cleanly.")
