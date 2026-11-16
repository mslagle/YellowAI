"""
Compact Local Ollama Game Loop for YellowAI.
Captures mGBA frames -> saves per-turn temp screenshots -> asks Ollama Qwen 3.5 ->
archives all signs/dialogue losslessly -> updates/drops goals cleanly -> sends GBA button presses.
"""

from datetime import datetime
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import config
from agent.ollama_player import OllamaPlayer
from agent.web_search import WebSearchClient
from core.conversation_history import ConversationHistoryManager
from core.dialogue_recorder import DialogueRecorder
from core.goal_tracker import GoalTracker
from core.logger import log_file_only
from core.thinking_store import ThinkingStore
from emulator.input_controller import InputController
from emulator.screen_capture import ScreenCapture


class GameLoop:
    """Pure-local Ollama autonomous loop for Pokémon Recharged Yellow."""

    def __init__(
        self,
        ollama_base_url: str = config.OLLAMA_BASE_URL,
        ollama_model_id: str = config.OLLAMA_MODEL_ID,
        turn_settle_delay: float = config.TURN_SETTLE_DELAY,
        step_settle_delay: float = config.STEP_SETTLE_DELAY,
        discovery_on_start: bool = config.OLLAMA_DISCOVERY_ON_START,
        force_discovery: bool = False,
    ):
        self.turn_settle_delay = turn_settle_delay
        self.step_settle_delay = step_settle_delay
        self.running = False
        self.turn_id = 0
        self.max_discovery_turns = config.OLLAMA_MAX_DISCOVERY_TURNS
        self.discovery_turns = 0

        self.screen_capture = ScreenCapture(window_keywords=config.EMULATOR_WINDOW_TITLE_KEYWORDS)
        self.input_controller = InputController(key_mappings=config.KEY_MAPPINGS, hold_duration=config.INPUT_HOLD_DURATION)

        self.goal_tracker = GoalTracker()
        self.discovery_active = bool(
            force_discovery
            or (discovery_on_start and (not self.goal_tracker.discovered_facts or "Unknown" in str(self.goal_tracker.verified_profile.get("badges", ""))))
        )
        self.dialogue_recorder = DialogueRecorder()
        self.thinking_store = ThinkingStore()
        self.web_client = WebSearchClient()
        self.conversation_history = ConversationHistoryManager()

        self.player = OllamaPlayer(
            base_url=ollama_base_url,
            model_id=ollama_model_id,
            history_manager=self.conversation_history,
            web_client=self.web_client,
            thinking_store=self.thinking_store,
            goal_tracker=self.goal_tracker,
            dialogue_recorder=self.dialogue_recorder,
        )

        self.run_temp_dir = config.TEMP_SCREENSHOTS_DIR / f"run_{self.conversation_history.current_run_id}"
        self.run_temp_dir.mkdir(parents=True, exist_ok=True)
        self.previous_thumb: Optional[np.ndarray] = None
        self.previous_frame: Optional[Image.Image] = None
        self.recent_thumbs: List[np.ndarray] = []
        self.recent_dialogues: List[str] = []
        self.unchanged_turns: int = 0
        self.loop_turns: int = 0
        self.escape_idx: int = 0
        self.blocked_dirs_here: List[str] = []
        self.last_synth_area_turn = ("", -999)
        self.recent_button_log: List[str] = []

    def _compute_screen_changed(self, thumb: np.ndarray) -> bool:
        if self.previous_thumb is None:
            return True
        arr_curr = np.asarray(thumb, dtype=np.int16)
        arr_prev = np.asarray(self.previous_thumb, dtype=np.int16)
        diff = float(np.mean(np.abs(arr_curr - arr_prev)))
        return diff > 1.8

    def _detect_frame_oscillation(self, thumb: np.ndarray) -> bool:
        """Detects 2-to-8-turn visual toggle loops (e.g. repeatedly opening/closing the same poster/sign)."""
        if len(self.recent_thumbs) < 2:
            return False
        arr_curr = np.asarray(thumb, dtype=np.int16)
        for older in self.recent_thumbs[:-1]:
            if float(np.mean(np.abs(arr_curr - np.asarray(older, dtype=np.int16)))) <= 1.8:
                return True
        return False

    def _save_temp_screenshot(self, frame: Image.Image, turn_id: int) -> str:
        path = self.run_temp_dir / f"turn_{turn_id:05d}.png"
        try:
            frame.save(path)
            return str(path)
        except Exception:
            return ""

    def run(self, max_turns: Optional[int] = None) -> bool:
        hwnd = self.screen_capture.find_emulator_window()
        if not hwnd:
            print("[ERROR] mGBA window not detected! Launch mGBA first or pass --launch-mgba.")
            return False

        self.input_controller.set_target_hwnd(hwnd)
        if not self.input_controller.focus_hwnd(hwnd):
            print("[ERROR] Could not prepare mGBA window for background input.")
            return False

        if not self.player.is_server_reachable():
            print(f"[ERROR] Ollama server unreachable at {self.player.base_url}. Run 'ollama serve'.")
            return False

        print(f"[GameLoop] Attached to '{self.screen_capture.target_title}' (HWND: {hwnd})")
        print(f"[GameLoop] Model: {self.player.model_id} | Temp screenshots: {self.run_temp_dir}")

        if config.OLLAMA_AUTO_PLAN_ON_STARTUP and self.player.enable_web_search and not self.web_client.cache.get("facts"):
            log_file_only("[Startup Plan] Searching walkthrough guide for Pokémon Recharged Yellow...")
            self.player.run_web_research_session(
                objective_question=(
                    f"Pokémon Recharged Yellow walkthrough summary and progression from: "
                    f"{self.goal_tracker.current_area} ({self.goal_tracker.active_goal})"
                ),
                current_area=self.goal_tracker.current_area,
                max_tool_rounds=2,
            )

        self.running = True
        try:
            while self.running:
                if max_turns is not None and self.turn_id >= max_turns:
                    print(f"[GameLoop] Reached max turns ({max_turns}).")
                    break
                if not self.screen_capture.is_window_alive():
                    print("[GameLoop] mGBA window closed; stopping cleanly.")
                    break

                frame, thumb = self.screen_capture.capture_frame()
                if frame is None or thumb is None:
                    time.sleep(0.5)
                    continue

                self.turn_id += 1
                shot_path = self._save_temp_screenshot(frame, self.turn_id)

                screen_changed = self._compute_screen_changed(thumb)
                frame_oscillating = self._detect_frame_oscillation(thumb)
                self.unchanged_turns = 0 if screen_changed else (self.unchanged_turns + 1)
                if not screen_changed or frame_oscillating:
                    self.loop_turns += 1
                else:
                    self.loop_turns = 0

                if self.turn_id > 1:
                    prev_btns = self.conversation_history.active_turns[-1].get("buttons", []) if self.conversation_history.active_turns else []
                    self.conversation_history.update_last_turn_outcome(
                        buttons_executed=prev_btns,
                        outcome="LOOPING" if frame_oscillating else ("CHANGED" if screen_changed else "UNCHANGED"),
                    )
                    if not screen_changed or frame_oscillating:
                        for b in prev_btns:
                            if b in {"UP", "DOWN", "LEFT", "RIGHT"} and b not in self.blocked_dirs_here:
                                self.blocked_dirs_here.append(b)
                    else:
                        self.blocked_dirs_here.clear()

                stuck_alert = ""
                if self.unchanged_turns >= 3 or self.loop_turns >= 3:
                    untried = [d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d not in self.blocked_dirs_here]
                    blocked_note = (
                        f" Directions already BLOCKED here: {self.blocked_dirs_here}."
                        + (f" Try untried direction(s): {untried}!" if untried else "")
                        if self.blocked_dirs_here
                        else ""
                    )
                    stuck_alert = (
                        f"LOOP DETECTED ({max(self.unchanged_turns, self.loop_turns)} turns)! "
                        "You are revisiting the exact same screen, wall, sign, or menu. "
                        f"Press B if a textbox/menu is open, then move in a different direction.{blocked_note}"
                    )

                same_area_cooldown = (
                    self.goal_tracker.current_area == self.last_synth_area_turn[0]
                    and (self.turn_id - self.last_synth_area_turn[1]) < 30
                )
                if (
                    not same_area_cooldown
                    and (self.loop_turns == 6 or (self.goal_tracker.is_drifting(18) and self.turn_id % 24 == 0))
                    and self.player.enable_web_search
                ):
                    log_file_only("[GameLoop] Consulting web search & re-synthesizing goal after drift/loop...")
                    self.last_synth_area_turn = (self.goal_tracker.current_area, self.turn_id)
                    self.player.synthesize_goal_from_discoveries(
                        image=frame,
                        extra_context=f"Caught in UI/wall loop for {self.loop_turns} turns / {self.goal_tracker.turns_without_change} stagnant turns.",
                    )

                discovery_directive = ""
                if self.discovery_active:
                    discovery_directive = self.goal_tracker.format_discovery_directive(
                        self.discovery_turns + 1, self.max_discovery_turns
                    )

                decision = self.player.decide_action(
                    turn_id=self.turn_id,
                    image=frame,
                    previous_image=self.previous_frame,
                    discovery_directive=discovery_directive,
                    action_history=" -> ".join(self.recent_button_log[-6:]),
                    stuck_alert=stuck_alert,
                    emulator_verified=True,
                    screenshot_path=shot_path,
                )

                # Losslessly record any visible sign, speech, or menu text
                diag = decision.get("dialogue_text")
                new_dialogue_seen = False
                dialogue_repeating = False
                if isinstance(diag, str) and diag.strip() and diag.strip().upper() != "NULL":
                    clean_diag = " ".join(diag.strip().split())
                    if clean_diag.lower() in [d.lower() for d in self.recent_dialogues]:
                        dialogue_repeating = True
                    self.recent_dialogues.append(clean_diag)
                    self.recent_dialogues = self.recent_dialogues[-6:]
                    rec = self.dialogue_recorder.record(
                        text=clean_diag,
                        turn_id=self.turn_id,
                        run_id=self.conversation_history.current_run_id,
                        area=decision.get("current_area") or self.goal_tracker.current_area,
                        screenshot_path=shot_path,
                    )
                    new_dialogue_seen = rec is not None

                # Drop completed goals FIRST before updating active/new goals
                comp = decision.get("completed_goal")
                if isinstance(comp, str) and comp.strip() and comp.strip().upper() != "NULL":
                    dropped = self.goal_tracker.complete_goal(comp.strip())
                    if dropped:
                        log_file_only(f"[Goal Dropped]: '{dropped}'")

                is_looping_now = (self.loop_turns >= 2) or dialogue_repeating
                self.goal_tracker.update_from_turn(
                    area=decision.get("current_area"),
                    discovered_fact=decision.get("discovered_fact"),
                    dialogue_text=diag if isinstance(diag, str) else None,
                    screen_kind=decision.get("screen_kind"),
                    active_goal_update=decision.get("active_goal_update"),
                    next_step=decision.get("next_step"),
                    grand_goal_update=decision.get("grand_goal_update"),
                    screen_changed=screen_changed,
                    loop_detected=is_looping_now,
                    new_dialogue_seen=new_dialogue_seen,
                )

                # Only count Discovery Audit turns once past Title/Intro/Naming screens
                if self.discovery_active:
                    screen_kind = str(decision.get("screen_kind") or "").upper()
                    in_inspectable_game = screen_kind in {
                        "OVERWORLD", "START_MENU", "TRAINER_CARD", "PARTY_MENU", "BAG_MENU", "POKEDEX"
                    }
                    if in_inspectable_game:
                        self.discovery_turns += 1
                    menus_done = not self.goal_tracker.get_uninspected_menus() and screen_kind == "OVERWORLD"
                    ready_signal = bool(decision.get("ready_to_plan_goal")) and in_inspectable_game
                    if ready_signal or menus_done or self.discovery_turns >= self.max_discovery_turns:
                        self.discovery_active = False
                        self.goal_tracker.discovery_done_this_run = True
                        log_file_only("[Discovery Mode] Menu/sign audit finished -> synthesizing goal from online search + verified facts...")
                        self.player.synthesize_goal_from_discoveries(image=frame)

                # Execute returned GBA button presses directly (prefer untried directions on 4-way loop escape)
                buttons = decision.get("flat_actions") or ["A"]
                if self.loop_turns >= 7 and self.loop_turns % 4 == 3:
                    dirs = ["RIGHT", "UP", "DOWN", "LEFT"]
                    untried = [d for d in dirs if d not in self.blocked_dirs_here]
                    pool = untried or dirs
                    esc_dir = pool[self.escape_idx % len(pool)]
                    self.escape_idx += 1
                    log_file_only(f"[Loop Breaker] Breaking out of {self.loop_turns}-turn loop with ['B', '{esc_dir}', '{esc_dir}']")
                    buttons = ["B", esc_dir, esc_dir]
                    self.recent_thumbs.clear()

                for btn in buttons:
                    self.input_controller.press_button(btn)
                    time.sleep(self.step_settle_delay)

                self.recent_button_log.append(f"T{self.turn_id}:{'+'.join(buttons)}")
                self.recent_button_log = self.recent_button_log[-10:]
                log_file_only(
                    f"[Turn {self.turn_id}] Buttons={buttons} | Area={self.goal_tracker.current_area} | "
                    f"Goal='{self.goal_tracker.active_goal[:60]}' | Reason={decision.get('reasoning', '')[:120]}"
                )

                self.previous_thumb = thumb
                self.previous_frame = frame
                self.recent_thumbs.append(thumb)
                self.recent_thumbs = self.recent_thumbs[-8:]
                time.sleep(self.turn_settle_delay)

        except KeyboardInterrupt:
            print("\n[GameLoop] Stopped by user.")
        finally:
            self.stop()
        return True

    def stop(self) -> None:
        self.running = False
        try:
            self.conversation_history.finalize_run(
                current_area=self.goal_tracker.current_area,
                active_goal=self.goal_tracker.active_goal,
                team_summary=" | ".join(self.goal_tracker.discovered_facts[-4:]) or "Inspect Start menu next run",
                badges=0,
            )
            self.goal_tracker.save()
        except Exception:
            pass
        print(f"[GameLoop] Session closed at Turn {self.turn_id} ({datetime.now().strftime('%H:%M:%S')}).")
