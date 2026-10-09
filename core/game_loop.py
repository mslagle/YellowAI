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
from core.screen_analyzer import ScreenAnalyzer
from core.movement_controller import MovementController
from core.battle_controller import BattleController
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
        
        # Dual-Tier Perception & Movement Controllers
        self.screen_analyzer = ScreenAnalyzer(
            target_width=CAPTURE_TARGET_WIDTH,
            target_height=CAPTURE_TARGET_HEIGHT,
            debug_poi_path="logs/last_poi_annotated.png"
        )
        self.movement_controller = MovementController(
            input_controller=self.input_controller,
            screen_capture=self.screen_capture,
            step_settle_delay=self.step_settle_delay,
            route_debug_path="logs/last_route_planned.png"
        )
        self.battle_controller = BattleController(
            input_controller=self.input_controller
        )
        
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

    def _detect_screen_mode(self, frame: Image.Image) -> Tuple[Any, str]:
        """Analyzes frame via NumPy ScreenAnalyzer and generates grounded prompt with POIs."""
        scene = self.screen_analyzer.analyze(frame, current_area_hint=self.memory_manager.current_area)
        self.memory_manager.current_area = scene.area_description

        # Transition quests upon leaving naming screen
        if scene.area_type == "BEDROOM_2F" and "bedroom" not in (self.memory_manager.priority_quest or "").lower():
            self.memory_manager.set_priority_quest("Leave Red's bedroom, go downstairs, and talk to Mom")

        if scene.is_battle or self.in_battle:
            self.in_battle = True

        prompt = self.screen_analyzer.format_grounded_prompt(scene)
        return scene, prompt

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

                latest_frame = frame

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
                    # 7. Analyze screen via NumPy ScreenAnalyzer and build grounded prompt
                    scene, mode_hint = self._detect_screen_mode(frame)

                    # Check for Battle Opening / Enemy Pokémon Switch Thinking Budget
                    trigger_battle_thinking = False
                    battle_advisor_prompt = ""
                    if ENABLE_BATTLE_THINKING and (scene.is_battle or "BATTLE" in mode_hint):
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

                    intent = decision.get("intent", "EXIT_ROOM")
                    target_poi = decision.get("target_poi")
                    battle_action = decision.get("battle_action")
                    move_slot = decision.get("move_slot")
                    target_item = decision.get("target_item")
                    switch_target_slot = decision.get("switch_target_slot")
                    reasoning = decision.get("reasoning", "")
                    thought = decision.get("thought", None)
                    cadence = decision.get("cadence_tag", "")

                    print(f"Strategic Decision: Intent=[{intent}] Target=[{target_poi or 'N/A'}] | Reason: '{reasoning}' (Cadence: {cadence})")

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
                elif (scene.is_battle or "BATTLE" in mode_hint) and not self.in_battle:
                    self.in_battle = True
                elif not scene.is_battle and "OVERWORLD" in mode_hint and self.in_battle:
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

                # Event-driven memory sync on doorway transitions
                if self.anti_stuck.doorway_warp_turns == 2:
                    self.memory_manager.save()

                # 8. Execution Routing (BattleController vs MovementController vs Dialogue)
                executed_actions: List[str] = []

                if self._is_naming_screen(frame):
                    print("[Naming Screen] Waiting for character name input.")
                    executed_actions = ["WAIT"]
                elif scene.is_battle or intent == "BATTLE_ACTION":
                    print(f"[Battle Action] Executing {battle_action or 'FIGHT'} (Move Slot {move_slot or 1})...")
                    battle_res = self.battle_controller.execute_battle_strategy(
                        battle_action=battle_action,
                        move_slot=move_slot,
                        target_item=target_item,
                        switch_target_slot=switch_target_slot
                    )
                    executed_actions = battle_res.get("executed", [])
                elif intent == "ADVANCE_DIALOGUE" or scene.is_dialogue:
                    print("[Dialogue Action] Pressing A to advance text...")
                    self.input_controller.press_button("A")
                    executed_actions = ["A"]
                elif intent == "WAIT":
                    time.sleep(self.turn_settle_delay)
                    executed_actions = ["WAIT"]
                else:
                    # Deterministic Grid Pathfinding via MovementController!
                    print(f"[Motor Execution] Pathfinding Intent=[{intent}] Target=[{target_poi or 'Auto'}]...")
                    move_res = self.movement_controller.plan_and_execute(
                        intent=intent,
                        target_poi_name=target_poi,
                        scene=scene,
                        frame=frame
                    )
                    executed_actions = move_res.get("executed_buttons", [])
                    planned = move_res.get("planned_buttons", [])
                    if not executed_actions and planned:
                        executed_actions = planned
                    print(f"[Movement Planned]: Route={move_res.get('target', '')} ({len(planned)} buttons: {' -> '.join(planned)})")

                # Grab fresh frame after actions executed
                fresh_frame, _ = self.screen_capture.capture_frame()
                if fresh_frame is not None:
                    latest_frame = fresh_frame

                # Record actions in anti-stuck watchdog
                for act in executed_actions:
                    self.anti_stuck.record_action(self.turn_id, act, reasoning, outcome="MOVED")

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
