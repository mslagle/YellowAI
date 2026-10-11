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
from core.logger import log_file_only
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
        if scene.area_type == "OVERWORLD":
            cur_lower = (self.memory_manager.current_area or "").lower()
            if not any(k in cur_lower for k in ("viridian", "route 1", "route 2", "pewter", "forest")):
                self.memory_manager.current_area = scene.area_description
        else:
            self.memory_manager.current_area = scene.area_description

        notes_joined = " ".join(scene.spatial_notes)
        if ("PIKACHU_FOLLOWING" in notes_joined or "STARTER_ALREADY_CLAIMED" in notes_joined) and not self.memory_manager.team:
            self.memory_manager.apply_party_update("Received starter Pikachu Lv5")

        # Transition quests upon room movement
        current_prio = (self.memory_manager.priority_quest or "").lower()
        current_micro = (self.memory_manager.micro_step or "").lower()
        has_starter = bool(self.memory_manager.team)

        if scene.area_type == "BEDROOM_2F" and "bedroom" not in current_prio:
            self.memory_manager.set_priority_quest("Leave Red's bedroom, go downstairs, and talk to Mom")
        elif scene.area_type == "LIVING_ROOM_1F":
            if not has_starter:
                if "bedroom" in current_prio or "go downstairs" in current_prio:
                    notes_text = " ".join(self.memory_manager.knowledge_notes).lower()
                    if "professor oak" in notes_text or "oak" in notes_text:
                        self.memory_manager.set_priority_quest("Exit house through front door to Pallet Town, then visit Professor Oak's Lab")
                        self.memory_manager.update_micro_step("Exit house through the front door mat at the bottom.")
                    else:
                        self.memory_manager.set_priority_quest("Talk to Mom in the living room, then exit the house")
            else:
                for q in list(self.memory_manager.quests):
                    if "starter" in q.lower() or "oak's lab" in q.lower():
                        self.memory_manager.complete_quest(q)
                if "viridian" not in current_prio and "route 1" not in current_prio:
                    self.memory_manager.set_priority_quest("Say goodbye to Mom in the living room, then exit the house and travel NORTH along Route 1 to Viridian City")
        elif scene.area_type == "OVERWORLD":
            # Complete any indoor exit quest once outside in Pallet Town
            for q in list(self.memory_manager.quests):
                if "exit house" in q.lower() or "living room" in q.lower() or "bedroom" in q.lower():
                    self.memory_manager.complete_quest(q)
            if has_starter:
                for q in list(self.memory_manager.quests):
                    if "choose starter" in q.lower() or "starter pokémon" in q.lower():
                        self.memory_manager.complete_quest(q)
                has_parcel = any("parcel" in q.lower() for q in self.memory_manager.quests) or any("parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
                parcel_delivered = any("pokedex" in kn.lower() or "pokédex" in kn.lower() or "delivered oak's parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
                is_viridian = any(p.name in ("POKECENTER_DOOR", "POKEMART_DOOR") for p in scene.pois) or "viridian" in self.memory_manager.current_area.lower()
                is_pallet_exit = any(p.name == "EXITED_HOUSE_DOOR" for p in scene.pois) and not is_viridian

                if has_parcel and not parcel_delivered:
                    self.memory_manager.set_priority_quest("Deliver Oak's Parcel back to Professor Oak in Pallet Town")
                    if is_viridian:
                        self.memory_manager.set_current_area("Viridian City")
                        self.memory_manager.update_micro_step("Walk SOUTH out of Viridian City and hop down Route 1 ledges to Pallet Town.")
                    else:
                        self.memory_manager.update_micro_step("Visit Professor Oak's Lab in southeast Pallet Town to deliver the parcel.")
                elif is_viridian:
                    self.memory_manager.set_current_area("Viridian City")
                    for q in list(self.memory_manager.quests):
                        if "leave pallet" in q.lower() or "route 1" in q.lower() or "heal" in q.lower():
                            self.memory_manager.complete_quest(q)
                    if "pewter" not in current_prio and "forest" not in current_prio and "brock" not in current_prio:
                        self.memory_manager.set_priority_quest("Travel NORTH from Viridian City along Route 2 through Viridian Forest to defeat Brock at Pewter City Gym")
                        self.memory_manager.update_micro_step("Walk NORTH through Viridian City to the Route 2 gatehouse.")
                elif is_pallet_exit:
                    for q in list(self.memory_manager.quests):
                        if "mom" in q.lower() or "goodbye" in q.lower():
                            self.memory_manager.complete_quest(q)
                    self.memory_manager.set_priority_quest("Leave Pallet Town via the NORTH exit and travel NORTH along Route 1 to Viridian City")
                    self.memory_manager.update_micro_step("Walk NORTH out of Pallet Town and follow Route 1 NORTH to Viridian City.")
            else:
                if "bedroom" in current_prio or "living room" in current_prio or "exit house" in current_prio:
                    self.memory_manager.set_priority_quest("Visit Professor Oak's Lab in southeast Pallet Town to choose starter Pokémon")
                if any(k in current_micro for k in ["front door", "exit house", "starter table", "downstairs", "bedroom"]):
                    self.memory_manager.update_micro_step("Walk southeast around the fence to Professor Oak's Lab entrance.")
        elif scene.area_type == "OAKS_LAB":
            has_parcel = any("parcel" in q.lower() for q in self.memory_manager.quests) or any("parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
            parcel_delivered = any("pokedex" in kn.lower() or "pokédex" in kn.lower() or "delivered oak's parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
            if has_parcel and not parcel_delivered:
                self.memory_manager.set_priority_quest("Talk to Professor Oak to deliver Oak's Parcel and receive the Pokédex")
                self.memory_manager.update_micro_step("Walk UP to Professor Oak and press A to deliver the parcel.")
            elif "LAB_TABLE_EMPTY (OAK NOT HERE)" in notes_joined:
                self.memory_manager.set_priority_quest("Exit Professor Oak's Lab and walk NORTH into the tall grass exit of Pallet Town to trigger Professor Oak")
                self.memory_manager.update_micro_step("Walk SOUTH to LAB_EXIT to leave the empty Lab, then walk NORTH to the tall grass exit of Pallet Town.")
            elif "STARTER_ALREADY_CLAIMED" in notes_joined or has_starter:
                for q in list(self.memory_manager.quests):
                    if "choose" in q.lower() or "starter" in q.lower():
                        self.memory_manager.complete_quest(q)
                self.memory_manager.set_priority_quest("Exit Professor Oak's Lab and travel NORTH via Route 1, Route 2, and Viridian Forest to defeat Brock at Pewter City Gym")
                self.memory_manager.update_micro_step("Walk SOUTH to LAB_EXIT to exit Professor Oak's Lab into Pallet Town.")
            elif not has_starter:
                self.memory_manager.set_priority_quest("Approach the Starter Table in Professor Oak's Lab and press A on the Poké Ball to claim Starter Pikachu")
                if any(k in current_micro for k in ["southeast", "door", "entrance", "outside", "pallet", "tall grass"]):
                    self.memory_manager.update_micro_step("Walk to STARTER_TABLE on the right and press A to claim Starter Pikachu.")
        elif scene.area_type == "POKECENTER":
            for q in list(self.memory_manager.quests):
                if "travel north" in q.lower() or "leave pallet" in q.lower() or "route 1" in q.lower():
                    self.memory_manager.complete_quest(q)
            if "heal" not in current_prio and "pewter" not in current_prio and "brock" not in current_prio and "parcel" not in current_prio:
                self.memory_manager.set_priority_quest("Heal Pikachu with Nurse Joy, then head NORTH via Route 2 and Viridian Forest to Pewter City Gym")
                self.memory_manager.update_micro_step("Talk to Nurse Joy at the counter to heal, then exit the Pokémon Center.")
        elif scene.area_type == "POKEMART":
            for q in list(self.memory_manager.quests):
                if "travel north" in q.lower() or "route 1" in q.lower():
                    self.memory_manager.complete_quest(q)

        if scene.is_bag_menu or scene.area_type == "BAG_MENU":
            if scene.bag_items_empty:
                if self.memory_manager.supplies.get("poke_balls", 0) > 0 or self.memory_manager.supplies.get("potions", 0) > 0:
                    self.memory_manager.update_supplies(poke_balls=0, potions=0)
                bag_warning = "BAG NOTICE: Inventory pocket is EMPTY (0 items / only 'Cancel'). You have 0 usable items or Poké Balls in this pocket! In battle, do NOT choose BAG; select FIGHT to attack or RUN to flee."
                if bag_warning not in scene.spatial_notes:
                    scene.spatial_notes.append(bag_warning)

        if scene.is_battle:
            self.in_battle = True
            if self.memory_manager.supplies.get("poke_balls", 0) == 0:
                scene.spatial_notes.append("BATTLE ALERT: You currently have 0 Poké Balls available in your active items pocket. Do NOT select BAG. Choose FIGHT to defeat the Pokémon or RUN to escape.")
        elif not scene.is_battle and self.in_battle and scene.area_type not in ("BATTLE", "BAG_MENU"):
            self.in_battle = False
            self.current_opponent = None
            self.battle_advisor_given_for_opponent = None

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

                # If caught mid-doorway or menu open black fade transition, wait for fade to finish
                for _ in range(3):
                    if thumb is not None and float(thumb.mean()) < 25.0:
                        time.sleep(0.4)
                        re_frame, re_thumb = self.screen_capture.capture_frame()
                        if re_frame is not None:
                            frame, thumb = re_frame, re_thumb
                    else:
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
                    log_file_only(f"[Watchdog Notice] Repeated dialogue/action loop detected. Alerting Gemini.")

                action_history_prompt = self.anti_stuck.get_action_history_prompt()
                if is_stuck and stuck_alert:
                    log_file_only(f"[Watchdog Alert]: {stuck_alert}")

                # 6. Assemble memory context
                memory_prompt = self.memory_manager.get_memory_context_prompt()

                memory_changed_items: List[str] = []

                # Check if Last-Ditch Effort / Tactical Rescue is triggered
                is_rescue, rescue_reason = self.anti_stuck.is_rescue_needed(
                    rescue_threshold=RESCUE_STUCK_THRESHOLD
                )

                if ENABLE_RESCUE_THINKING and is_rescue:
                    log_file_only(f"[Tactical Rescue Triggered] Condition: {rescue_reason}")

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
                        self.memory_manager.add_knowledge_note(clues.strip())
                        memory_changed_items.append(f"Clues: {clues.strip()[:60]}")

                    # Update micro-step and short-term memory with tactical breakthrough
                    new_micro = rescue_result.get("updated_micro_step")
                    if new_micro and isinstance(new_micro, str):
                        self.memory_manager.update_micro_step(new_micro.strip())
                        memory_changed_items.append(f"Step: {new_micro.strip()}")

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
                            log_file_only(f"[Battle Strategy Triggered] Allocating {BATTLE_THINKING_BUDGET} thinking tokens for {opp_label}!")

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
                        dialogue_crop=scene.dialogue_crop,
                        emulator_verified=True  # Verified alive and visible right above!
                    )
                    api_duration = time.time() - api_start

                    intent = decision.get("intent", "EXIT_ROOM")
                    target_poi = decision.get("target_poi")
                    dialogue_text = decision.get("dialogue_text")
                    battle_action = decision.get("battle_action")
                    move_slot = decision.get("move_slot")
                    target_item = decision.get("target_item")
                    switch_target_slot = decision.get("switch_target_slot")
                    reasoning = decision.get("reasoning", "")
                    thought = decision.get("thought", None)
                    cadence = decision.get("cadence_tag", "")

                    if dialogue_text and isinstance(dialogue_text, str) and dialogue_text.strip() and dialogue_text.strip().upper() != "NULL":
                        clean_diag = dialogue_text.strip()
                        log_file_only(f"[Speech Bubble Text]: \"{clean_diag}\"")
                        self.memory_manager.add_knowledge_note(f"Dialogue: {clean_diag}")
                        identified_area = self.memory_manager.ingest_sign_or_dialogue_location(clean_diag)
                        if identified_area:
                            memory_changed_items.append(f"Oriented in '{identified_area}' via sign/dialogue")

                    log_file_only(f"Strategic Decision: Intent=[{intent}] Target=[{target_poi or 'N/A'}] | Reason: '{reasoning}' (Cadence: {cadence})")

                # Update Battle Tracking & Enemy Changes
                opp = decision.get("opponent_pokemon")
                if opp and isinstance(opp, str) and opp.strip() and opp.strip().upper() != "NULL":
                    opp_clean = opp.strip()
                    if not self.in_battle or self.current_opponent != opp_clean:
                        log_file_only(f"[Battle Opponent Detected]: {opp_clean}")
                        self.in_battle = True
                        self.current_opponent = opp_clean
                    if trigger_battle_thinking:
                        self.battle_advisor_given_for_opponent = opp_clean
                elif (scene.is_battle or "BATTLE" in mode_hint) and not self.in_battle:
                    self.in_battle = True
                elif not scene.is_battle and "OVERWORLD" in mode_hint and self.in_battle:
                    log_file_only("[Battle Concluded] Returning to Overworld navigation.")
                    self.in_battle = False
                    self.current_opponent = None
                    self.battle_advisor_given_for_opponent = None

                # Ingest Discovered POIs, Item Updates, and Party Changes
                poi = decision.get("discovered_poi")
                if poi and isinstance(poi, str) and poi.strip() and poi.strip().upper() != "NULL":
                    self.memory_manager.add_waypoint(self.memory_manager.current_area, poi.strip())
                    memory_changed_items.append(f"Discovered POI: {poi.strip()}")

                items_up = decision.get("items_update")
                if items_up and isinstance(items_up, str) and items_up.strip() and items_up.strip().upper() != "NULL":
                    self.memory_manager.apply_items_update(items_up.strip())
                    memory_changed_items.append(f"Items: {items_up.strip()}")

                party_up = decision.get("party_update")
                if party_up and isinstance(party_up, str) and party_up.strip() and party_up.strip().upper() != "NULL":
                    self.memory_manager.apply_party_update(party_up.strip())
                    memory_changed_items.append(f"Party: {party_up.strip()}")

                # Process Multi-Quest Log Updates
                comp_q = decision.get("completed_quest")
                if comp_q and isinstance(comp_q, str) and comp_q.strip() and comp_q.strip().upper() != "NULL":
                    removed = self.memory_manager.complete_quest(comp_q.strip())
                    if removed:
                        memory_changed_items.append(f"Completed Quest: '{removed}'")
                        if self.memory_manager.priority_quest:
                            memory_changed_items.append(f"Priority Quest: '{self.memory_manager.priority_quest}'")

                new_q = decision.get("new_quest")
                prio_q = decision.get("priority_quest")
                if new_q and isinstance(new_q, str) and new_q.strip() and new_q.strip().upper() != "NULL":
                    is_prio = (prio_q == new_q or not self.memory_manager.priority_quest)
                    self.memory_manager.add_quest(new_q.strip(), is_priority=is_prio)
                    memory_changed_items.append(f"New Quest: '{new_q.strip()}'")

                if prio_q and isinstance(prio_q, str) and prio_q.strip() and prio_q.strip().upper() != "NULL" and prio_q != new_q:
                    self.memory_manager.set_priority_quest(prio_q.strip())
                    memory_changed_items.append(f"Priority Quest: '{prio_q.strip()}'")

                # Process Micro-Step Updates
                micro_s = decision.get("micro_step")
                if micro_s and isinstance(micro_s, str) and micro_s.strip() and micro_s.strip().upper() != "NULL":
                    if self.memory_manager.micro_step != micro_s.strip():
                        self.memory_manager.update_micro_step(micro_s.strip())
                        memory_changed_items.append(f"Micro-Step: '{micro_s.strip()}'")

                # Event-driven memory sync on doorway transitions
                if self.anti_stuck.doorway_warp_turns == 2:
                    self.memory_manager.save()

                # 8. Execution Routing (BattleController vs MovementController vs Dialogue)
                executed_actions: List[str] = []

                if self._is_naming_screen(frame) or scene.is_naming_screen:
                    log_file_only("[Naming Screen] Confirming default name (B -> START -> A)...")
                    for nb in ["B", "START", "A"]:
                        self.input_controller.press_button(nb)
                        time.sleep(0.25)
                    executed_actions = ["B", "START", "A"]
                elif scene.is_bag_menu or scene.area_type == "BAG_MENU":
                    log_file_only("[Bag Screen] In-game Bag menu is open. Pressing B to dismiss...")
                    if scene.bag_items_empty and (self.memory_manager.supplies.get("poke_balls", 0) > 0 or self.memory_manager.supplies.get("potions", 0) > 0):
                        self.memory_manager.update_supplies(poke_balls=0, potions=0)
                        memory_changed_items.append("Supplies: Discovered active pocket is empty (Poké Balls: 0)")
                    self.input_controller.press_button("B")
                    time.sleep(1.0)
                    executed_actions = ["B"]
                elif scene.is_battle or intent == "BATTLE_ACTION":
                    if battle_action == "BAG" and self.memory_manager.supplies.get("poke_balls", 0) == 0:
                        log_file_only("[Battle Fallback] Poké Balls count is 0! Overriding BAG -> FIGHT (Move Slot 1).")
                        battle_action = "FIGHT"
                        move_slot = move_slot or 1
                    log_file_only(f"[Battle Action] Executing {battle_action or 'FIGHT'} (Move Slot {move_slot or 1})...")
                    battle_res = self.battle_controller.execute_battle_strategy(
                        battle_action=battle_action,
                        move_slot=move_slot,
                        target_item=target_item,
                        switch_target_slot=switch_target_slot,
                        frame=frame
                    )
                    executed_actions = battle_res.get("executed", [])
                elif intent == "ADVANCE_DIALOGUE" or scene.is_dialogue:
                    log_file_only("[Dialogue Action] Pressing A to advance text...")
                    self.input_controller.press_button("A")
                    executed_actions = ["A"]
                elif intent == "WAIT":
                    time.sleep(self.turn_settle_delay)
                    executed_actions = ["WAIT"]
                else:
                    notes_joined = " ".join(scene.spatial_notes)
                    active_goal_text = f"{self.memory_manager.priority_quest or ''} {self.memory_manager.micro_step or ''} {reasoning or ''}".lower()

                    # When holding Oak's Parcel to deliver to Professor Oak, do NOT re-enter Pokémon Center or Poké Mart!
                    has_parcel_active = any("parcel" in q.lower() for q in self.memory_manager.quests) or any("parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
                    parcel_delivered_active = any("pokedex" in kn.lower() or "pokédex" in kn.lower() or "delivered oak's parcel" in kn.lower() for kn in self.memory_manager.knowledge_notes)
                    if (
                        scene.area_type == "OVERWORLD"
                        and has_parcel_active
                        and not parcel_delivered_active
                        and "viridian" in (self.memory_manager.current_area or "").lower()
                    ):
                        if (
                            intent in ("ENTER_BUILDING", "TALK_TO_NPC")
                            or (intent == "GO_TO_POI" and target_poi and any(k in target_poi.upper() for k in ("POKECENTER", "POKEMART", "MART", "CENTER", "DOOR")))
                        ):
                            log_file_only(f"[Parcel Safeguard] Holding Oak's Parcel in Viridian City! Overriding {intent} ({target_poi}) -> EXPLORE_SOUTH (Heading to Pallet Town)")
                            intent = "EXPLORE_SOUTH"
                            target_poi = None

                    # If Professor Oak's Lab is visible on screen BEFORE we have a starter (and not heading to tall grass),
                    # snap generic EXPLORE_* intents directly to ENTER_BUILDING -> OAKS_LAB so we never walk past it!
                    if (
                        scene.area_type == "OVERWORLD"
                        and not self.memory_manager.team
                        and "PIKACHU_FOLLOWING" not in notes_joined
                        and "tall grass" not in active_goal_text
                        and intent.startswith("EXPLORE_")
                        and any(p.name == "OAKS_LAB" for p in scene.pois)
                    ):
                        if any(k in active_goal_text for k in ("oak", "lab", "starter")) or intent in ("EXPLORE_SOUTH", "EXPLORE_SOUTHEAST", "EXPLORE_EAST"):
                            log_file_only(f"[POI Override] 'OAKS_LAB' is in frame! Overriding {intent} -> ENTER_BUILDING (Target=OAKS_LAB)")
                            intent = "ENTER_BUILDING"
                            target_poi = "OAKS_LAB"
                    elif scene.area_type == "OAKS_LAB":
                        if "LAB_TABLE_EMPTY (OAK NOT HERE)" in notes_joined or "STARTER_ALREADY_CLAIMED" in notes_joined:
                            if intent != "EXIT_ROOM":
                                log_file_only(f"[Lab Override] Lab table is empty / starter already claimed! Overriding {intent} ({target_poi}) -> EXIT_ROOM (Target=LAB_EXIT)")
                                intent = "EXIT_ROOM"
                                target_poi = "LAB_EXIT"
                        elif not self.memory_manager.team and (
                            intent in ("EXIT_ROOM", "ENTER_BUILDING", "TALK_TO_NPC")
                            or (intent.startswith("EXPLORE_") and "NORTH" not in intent)
                            or (target_poi and any(k in target_poi.upper() for k in ("EXIT", "LAB", "DOOR", "OAK")))
                        ):
                            log_file_only(f"[Lab Safeguard] Inside Oak's Lab with Starter Poké Ball on table! Overriding {intent} ({target_poi}) -> GO_TO_POI (Target=STARTER_TABLE)")
                            intent = "GO_TO_POI"
                            target_poi = "STARTER_TABLE"

                    # Pre-Execution Stale-Scene Guard:
                    # Verify the game did not finish a doorway/stairs warp or battle transition while the LLM was thinking!
                    pre_frame, pre_thumb = self.screen_capture.capture_frame()
                    if pre_frame is not None and pre_thumb is not None and float(pre_thumb.mean()) < 22.0:
                        time.sleep(0.8)
                        pre_frame, pre_thumb = self.screen_capture.capture_frame()
                    if pre_frame is not None:
                        live_scene = self.screen_analyzer.analyze(
                            pre_frame,
                            current_area_hint=self.memory_manager.current_area
                        )
                        if (
                            live_scene.area_type != scene.area_type
                            or live_scene.is_battle != scene.is_battle
                            or (live_scene.is_dialogue and not scene.is_dialogue)
                        ):
                            log_file_only(
                                f"[Stale Plan Guard] Scene changed during LLM call "
                                f"({scene.area_type} -> {live_scene.area_type})! "
                                f"Aborting stale {intent} plan and re-evaluating new room."
                            )
                            self.memory_manager.set_current_area(live_scene.area_description)
                            self.previous_frame = pre_frame
                            continue

                    # Deterministic Grid Pathfinding via MovementController!
                    log_file_only(f"[Motor Execution] Pathfinding Intent=[{intent}] Target=[{target_poi or 'Auto'}]...")
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
                    log_file_only(f"[Movement Planned]: Route={move_res.get('target', '')} ({len(planned)} buttons: {' -> '.join(planned)})")
                    if move_res.get("interrupt_reason") == "SCENE_WARP_OR_BATTLE":
                        time.sleep(0.6)

                # Grab fresh frame after actions executed
                latest_frame = frame
                fresh_frame, _ = self.screen_capture.capture_frame()
                if fresh_frame is not None:
                    latest_frame = fresh_frame

                # Record high-level intent in anti-stuck watchdog
                self.anti_stuck.record_action(self.turn_id, intent, reasoning, outcome="MOVED")

                action_summary = " -> ".join(executed_actions) if executed_actions else "WAIT"
                if len(executed_actions) > 1:
                    log_file_only(f"[Batch Executed]: [{action_summary}] ({len(executed_actions)} steps)")

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
                if "short_term_memory" in decision:
                    new_stm = decision["short_term_memory"]
                    log_file_only(f"[Short-Term Memory Updated]: {new_stm}")
                    self.memory_manager.update_short_term(new_stm)

                if "medium_term_memory" in decision:
                    new_mtm = decision["medium_term_memory"]
                    log_file_only(f"[Medium-Term Memory Updated]: {new_mtm}")
                    self.memory_manager.update_medium_term(new_mtm)

                if "long_term_memory" in decision:
                    new_ltm = decision["long_term_memory"]
                    log_file_only(f"[Long-Term Memory Updated]: {new_ltm}")
                    self.memory_manager.update_long_term(new_ltm)

                # =====================================================================
                # User-Specified Console Turn Logging (Structured & Clean)
                # 1. A small blurb about what the AI is thinking
                # 2. What buttons we need to send based on what the AI wants to do
                # 3. Any TTS if we have it
                # 4. If in a battle sequence, the strategy to defeat or capture the pokemon
                # 5. Changes to our current quests or updates to our memory
                # =====================================================================
                # 1. AI Thinking / Reasoning
                ai_thought_blurb = reasoning.strip() if reasoning else (decision.get("situation_analysis", "").strip() or "Analyzing current situation.")
                print(f"Thought: {ai_thought_blurb}")

                # 2. Buttons to send
                buttons_display = " -> ".join(executed_actions) if executed_actions else "WAIT"
                print(f"Buttons: [{buttons_display}]")

                # 3. TTS commentary
                if thought and str(thought).strip():
                    print(f"TTS: \"{thought.strip()}\"")
                    if self.audio_queue:
                        self.audio_queue.enqueue_thought(thought.strip())

                # 4. Battle Sequence Strategy
                if scene.is_battle or self.in_battle or intent == "BATTLE_ACTION":
                    opp_name = self.current_opponent or "Wild Pokémon"
                    act_type = battle_action or "FIGHT"
                    if act_type == "BAG":
                        item_label = target_item or "Poké Ball"
                        strat_desc = f"Capture strategy: Use {item_label} to catch {opp_name}"
                    elif act_type == "RUN":
                        strat_desc = f"Flee strategy: Escape from battle against {opp_name}"
                    elif act_type in ("SWITCH", "SWITCH_POKEMON"):
                        strat_desc = f"Switch strategy: Switch to party slot {switch_target_slot or 2}"
                    else:
                        strat_desc = f"Combat strategy: Attack with Move Slot {move_slot or 1} to defeat {opp_name}"
                    print(f"Battle Strategy: {strat_desc}")

                # 5. Quest & Memory Updates
                changes: List[str] = []
                if memory_changed_items:
                    changes.extend(memory_changed_items)
                if changes:
                    for chg in changes:
                        print(f"Quest/Memory: {chg}")

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
                log_file_only(f"[Turn {self.turn_id} Complete in {turn_duration:.2f}s] (API: {api_duration:.2f}s{tok_str} | Settle: {wait_time:.2f}s)")

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
