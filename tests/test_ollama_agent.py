"""
Compact Test Suite for Local Ollama YellowAI:
- Multi-turn conversation history, compaction, & cross-run continuity
- Permanent goal dropping (completed_goals tombstones prevent repetition)
- Lossless sign/speech/dialogue archival
- Thinking trace archival & distilled insight recall
- Web search cache & tool schemas
- GBA button normalization, safeguard check, mGBA assets, & live qwen3.5:9b vision + web search
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agent.ollama_player import OllamaPlayer, TurnDecision, VALID_GBA_BUTTONS
from agent.web_search import WebSearchClient
from core.conversation_history import ConversationHistoryManager
from core.dialogue_recorder import DialogueRecorder
from core.game_loop import GameLoop
from core.goal_tracker import GoalTracker
from core.thinking_store import ThinkingStore
from emulator.mgba_launcher import find_mgba_executable, find_rom_file


class TestCompactOllamaAgent(unittest.TestCase):
    def test_conversation_history_compaction_and_cross_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_file = Path(tmpdir) / "history.json"
            h1 = ConversationHistoryManager(
                storage_path=hist_file,
                max_turns_before_compact=4,
                keep_recent_turns=2,
                compact_token_threshold=999999,
            )
            h1.current_run_id = "run_001"
            for t in range(1, 6):
                h1.record_turn(
                    turn_id=t,
                    area="Pallet Town",
                    goal="Get Pikachu",
                    user_summary=f"Turn {t}",
                    assistant_response={"reasoning": f"Step {t}", "actions": ["UP", "A"]},
                    buttons_executed=["UP", "A"],
                    outcome="CHANGED",
                    screenshot_path=f"temp/turn_{t:05d}.png",
                    prompt_tokens=200 * t,
                )
            h1.update_last_turn_outcome(["RIGHT", "A"], "CHANGED")
            self.assertTrue(h1.needs_compaction())
            summary = h1.compact_if_needed(
                pinned_goal="Reach Viridian City",
                summarizer_fn=lambda _: "Compacted T1-T3: Got Pikachu in Pallet Town.",
                completed_goals=["Get Pikachu"],
            )
            self.assertIn("Compacted T1-T3", summary)
            self.assertEqual(len(h1.active_turns), 2)
            h1.finalize_run(current_area="Pallet Town", active_goal="Get Pikachu", team_summary="Pikachu Lv5", badges=0)

            h2 = ConversationHistoryManager(storage_path=hist_file, max_turns_before_compact=4, keep_recent_turns=2)
            h2.current_run_id = "run_002"
            msgs = h2.build_ollama_messages(
                system_prompt="SYS",
                current_user_prompt="NOW",
                current_images=["fake_b64"],
                completed_goals=["Get Pikachu"],
            )
            self.assertEqual(len(msgs), 6)
            self.assertIn("ALREADY COMPLETED / DROPPED", msgs[0]["content"])
            self.assertIn("[THIS EARLIER GOAL IS NOW COMPLETED/DROPPED]", msgs[1]["content"])
            self.assertEqual(msgs[-1]["images"], ["fake_b64"])
            self.assertNotIn("images", msgs[1])

    def test_goal_tracker_permanent_goal_dropping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gfile = Path(tmpdir) / "goal.json"
            gt = GoalTracker(state_path=gfile)
            gt.update_from_turn(
                area="Daisy's House",
                discovered_fact="Spoke to Daisy",
                active_goal_update="Get Town Map from Daisy in Pallet Town",
                next_step="Press A facing Daisy",
            )
            dropped = gt.complete_goal("CURRENT")
            self.assertEqual(dropped, "Get Town Map from Daisy in Pallet Town")
            self.assertTrue(gt.is_goal_completed("Get Town Map from Daisy in Pallet Town"))

            # Re-adding finished goal must be ignored even after reload
            gt.update_from_turn(active_goal_update="Get Town Map from Daisy in Pallet Town")
            self.assertNotEqual(gt.active_goal, "Get Town Map from Daisy in Pallet Town")

            gt2 = GoalTracker(state_path=gfile)
            anchor = gt2.format_pinned_anchor()
            self.assertIn("Already Completed / Dropped Goals", anchor)
            self.assertIn("Get Town Map from Daisy in Pallet Town", anchor)

    def test_discovery_menu_profile_and_goal_synthesis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gfile = Path(tmpdir) / "goal.json"
            gt = GoalTracker(state_path=gfile)
            self.assertEqual(len(gt.get_uninspected_menus()), 4)

            gt.update_from_turn(screen_kind="TRAINER_CARD", discovered_fact="Badges: 1 (Boulder Badge) | Money: 3400")
            directive = gt.format_discovery_directive(1, 12)
            self.assertIn("TRAINER_CARD", directive)
            self.assertIn("PARTY_MENU", gt.get_uninspected_menus())

            gt.update_from_turn(screen_kind="PARTY_MENU", discovered_fact="Team: Pikachu Lv14, Mankey Lv11")
            gt.update_from_turn(screen_kind="BAG_MENU", discovered_fact="Bag: 4 Poké Balls, 2 Potions")
            gt.update_from_turn(screen_kind="POKEDEX", discovered_fact="Pokédex: Seen 6, Owned 3")
            self.assertEqual(gt.get_uninspected_menus(), [])

            gt.apply_synthesized_goal(
                new_active_goal="Travel EAST from Pewter City through Route 3 and Mt. Moon to Cerulean City",
                next_step="Exit Pewter City east onto Route 3",
                completion_criteria="Reach Cerulean City",
                confirmed_completed_goals=["Defeat Brock for Boulder Badge"],
            )
            self.assertTrue(gt.is_goal_completed("Defeat Brock for Boulder Badge"))

            gt_reloaded = GoalTracker(state_path=gfile)
            anchor = gt_reloaded.format_pinned_anchor()
            self.assertIn("Badges: 1 (Boulder Badge)", anchor)
            self.assertIn("Cerulean City", anchor)
            self.assertIn("Defeat Brock for Boulder Badge", anchor)

    def test_lossless_dialogue_and_thinking_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dr1 = DialogueRecorder(storage_path=tmp / "signs_and_speech.json")
            dr1.record("PALLET TOWN: Shades of your journey await!", turn_id=1, run_id="r1", area="Pallet Town")
            self.assertIsNone(dr1.record("PALLET TOWN: Shades of your journey await!", turn_id=2, run_id="r1", area="Pallet Town"))
            dr1.record("Trainer Tips: Press START to open Menu.", turn_id=5, run_id="r1", area="Pallet Town")

            dr2 = DialogueRecorder(storage_path=tmp / "signs_and_speech.json")
            self.assertEqual(len(dr2.entries), 2)
            self.assertTrue(dr2.search(query="Shades journey", area="Pallet Town"))

            ts1 = ThinkingStore(traces_path=tmp / "t.jsonl", insights_path=tmp / "i.json")
            ts1.record_trace("r1", 5, "Viridian Forest", "Find north exit", "Follow left trees.", 4, "Go left", ["LEFT"])
            ts1.add_insight("Follow outer edge in Viridian Forest.", area="Viridian Forest", category="GAMEPLAY", source="TURN")
            ts2 = ThinkingStore(traces_path=tmp / "t.jsonl", insights_path=tmp / "i.json")
            self.assertIn("outer edge", ts2.format_insights_prompt("Viridian Forest", "Find exit"))
            self.assertEqual(len((tmp / "t.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_buttons_schema_safeguard_and_assets(self):
        btns = OllamaPlayer.flatten_actions({"actions": ["u", "DOWN", "left", "R", "a", "B", "enter", "backspace", "BAD"]})
        self.assertEqual(btns, ["UP", "DOWN", "LEFT", "RIGHT", "A"])  # capped at MAX_BATCH_ACTIONS=5
        btns2 = OllamaPlayer.flatten_actions({"actions": ["B", "START", "SELECT"]})
        self.assertEqual(btns2, ["B", "START", "SELECT"])
        self.assertIn("actions", TurnDecision.model_json_schema()["required"])
        self.assertIsNotNone(find_mgba_executable())
        self.assertIsNotNone(find_rom_file())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            player = OllamaPlayer(
                history_manager=ConversationHistoryManager(tmp / "h.json"),
                web_client=WebSearchClient(tmp / "w.json"),
                thinking_store=ThinkingStore(tmp / "t.jsonl", tmp / "i.json"),
                goal_tracker=GoalTracker(tmp / "g.json"),
                dialogue_recorder=DialogueRecorder(tmp / "signs_and_speech.json"),
            )
            with self.assertRaises(RuntimeError):
                player.decide_action(turn_id=1, image=Image.new("RGB", (160, 144), (240, 210, 40)), emulator_verified=False)

    def test_live_ollama_vision_and_web_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            history = ConversationHistoryManager(tmp / "h.json")
            web_client = WebSearchClient(tmp / "w.json")
            player = OllamaPlayer(
                base_url=config.OLLAMA_BASE_URL,
                model_id=config.OLLAMA_MODEL_ID,
                history_manager=history,
                web_client=web_client,
                thinking_store=ThinkingStore(tmp / "t.jsonl", tmp / "i.json"),
                goal_tracker=GoalTracker(tmp / "g.json"),
                dialogue_recorder=DialogueRecorder(tmp / "signs_and_speech.json"),
            )
            if not (player.is_server_reachable() and player.is_model_available()):
                self.skipTest("Ollama daemon or model not available.")

            fixture = PROJECT_ROOT / "tests" / "fixtures" / "recharged_yellow" / "title_screen.png"
            img = Image.open(fixture).convert("RGB")
            decision = player.decide_action(turn_id=1, image=img, emulator_verified=True, screenshot_path=str(fixture))
            self.assertFalse(decision.get("fallback", False))
            for b in decision["flat_actions"]:
                self.assertIn(b, VALID_GBA_BUTTONS)
            self.assertEqual(len(history.active_turns), 1)
            search_res = web_client.search("Pokemon Recharged Yellow", max_results=2)
            self.assertIsInstance(search_res.get("results"), list)
            if not search_res.get("error"):
                self.assertGreaterEqual(len(search_res.get("results", [])), 1)

    def test_game_loop_turn_cycle_and_previous_frame_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frame = Image.new("RGB", (480, 320), (120, 180, 200))
            thumb = np.zeros((32, 32), dtype=np.uint8)
            pressed = []

            loop = GameLoop(turn_settle_delay=0.0, step_settle_delay=0.0, discovery_on_start=False)
            loop.run_temp_dir = tmp / "temp_shots"
            loop.run_temp_dir.mkdir(parents=True, exist_ok=True)
            loop.goal_tracker = GoalTracker(state_path=tmp / "goal.json")
            loop.dialogue_recorder = DialogueRecorder(storage_path=tmp / "signs.json")
            loop.thinking_store = ThinkingStore(traces_path=tmp / "traces.jsonl", insights_path=tmp / "insights.json")
            loop.web_client = WebSearchClient(cache_path=tmp / "web.json")
            loop.web_client.cache["facts"] = [{"topic": "t", "summary": "f"}]
            loop.conversation_history = ConversationHistoryManager(storage_path=tmp / "hist.json")
            loop.player.goal_tracker = loop.goal_tracker
            loop.player.dialogue_recorder = loop.dialogue_recorder
            loop.player.thinking_store = loop.thinking_store
            loop.player.web_client = loop.web_client
            loop.player.history_manager = loop.conversation_history
            loop.player.enable_web_search = False

            with (
                patch.object(loop.screen_capture, "find_emulator_window", return_value=12345),
                patch.object(loop.screen_capture, "is_window_alive", return_value=True),
                patch.object(loop.screen_capture, "capture_frame", return_value=(frame, thumb)),
                patch.object(loop.input_controller, "focus_hwnd", return_value=True),
                patch.object(loop.input_controller, "press_button", side_effect=lambda b: pressed.append(b) or True),
                patch.object(loop.player, "is_server_reachable", return_value=True),
                patch.object(
                    loop.player,
                    "_chat",
                    return_value={
                        "prompt_eval_count": 100,
                        "message": {
                            "content": json.dumps(
                                {
                                    "reasoning": "Read Pallet Town sign",
                                    "actions": ["A"],
                                    "current_area": "Pallet Town",
                                    "screen_kind": "OVERWORLD",
                                    "dialogue_text": "PALLET TOWN: Shades of your journey await!",
                                }
                            )
                        },
                    },
                ),
            ):
                ok = loop.run(max_turns=4)

            self.assertTrue(ok)
            self.assertEqual(pressed, ["A", "A", "A", "A"])
            self.assertIsInstance(loop.previous_frame, Image.Image)
            self.assertEqual(len(loop.dialogue_recorder.entries), 1)
            self.assertEqual(len(list((tmp / "temp_shots").glob("*.png"))), 4)

    def test_timeout_retries_in_fast_no_think_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            player = OllamaPlayer(
                think_mode="always",
                enable_web_search=False,
                history_manager=ConversationHistoryManager(tmp / "h.json"),
                web_client=WebSearchClient(tmp / "w.json"),
                thinking_store=ThinkingStore(tmp / "t.jsonl", tmp / "i.json"),
                goal_tracker=GoalTracker(tmp / "g.json"),
                dialogue_recorder=DialogueRecorder(tmp / "signs.json"),
            )
            calls = []

            def fake_chat(messages, think=False, json_schema=None, tools=None):
                calls.append(think)
                if len(calls) == 1:
                    raise requests.exceptions.ReadTimeout("simulated 60s timeout")
                return {
                    "prompt_eval_count": 500,
                    "message": {
                        "content": json.dumps({"reasoning": "Confirm name via START", "actions": ["START"]}),
                    },
                }

            img = Image.new("RGB", (240, 160), (200, 200, 200))
            with patch.object(player, "_chat", side_effect=fake_chat):
                res = player.decide_action(turn_id=1, image=img, emulator_verified=True)

            self.assertEqual(calls, [True, False])
            self.assertEqual(res["flat_actions"], ["START"])
            self.assertFalse(res.get("fallback", False))

    def test_empty_think_content_retries_in_fast_no_think_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            player = OllamaPlayer(
                think_mode="always",
                enable_web_search=False,
                history_manager=ConversationHistoryManager(tmp / "h.json"),
                web_client=WebSearchClient(tmp / "w.json"),
                thinking_store=ThinkingStore(tmp / "t.jsonl", tmp / "i.json"),
                goal_tracker=GoalTracker(tmp / "g.json"),
                dialogue_recorder=DialogueRecorder(tmp / "signs.json"),
            )
            calls = []

            def fake_chat(messages, think=False, json_schema=None, tools=None):
                calls.append(think)
                if len(calls) == 1:
                    return {
                        "prompt_eval_count": 600,
                        "message": {"thinking": "Long internal reasoning cut off...", "content": ""},
                    }
                return {
                    "prompt_eval_count": 500,
                    "message": {
                        "content": json.dumps({"reasoning": "Confirm name via START then A", "actions": ["START", "A"]}),
                    },
                }

            img = Image.new("RGB", (240, 160), (200, 200, 200))
            with patch.object(player, "_chat", side_effect=fake_chat):
                res = player.decide_action(turn_id=1, image=img, emulator_verified=True)

            self.assertEqual(calls, [True, False])
            self.assertEqual(res["flat_actions"], ["START", "A"])
            self.assertFalse(res.get("fallback", False))

    def test_oscillation_loop_detection_escape_and_fuzzy_fact_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt = GoalTracker(state_path=tmp / "g.json")
            gt.update_from_turn(discovered_fact="Area: Red Bedroom 2F, stairs visible on right", loop_detected=True)
            gt.update_from_turn(discovered_fact="Area: Red Bedroom 2F, stairs visible on right.", loop_detected=True)
            self.assertEqual(gt.current_area, "Red Bedroom 2F")
            self.assertEqual(len(gt.discovered_facts), 1)
            self.assertEqual(gt.turns_without_change, 2)

            # Oscillating poster toggle: Frame A -> Frame B -> Frame A -> Frame B...
            frame_a = Image.new("RGB", (480, 320), (40, 40, 40))
            frame_b = Image.new("RGB", (480, 320), (220, 220, 220))
            thumb_a = np.full((32, 32), 40, dtype=np.uint8)
            thumb_b = np.full((32, 32), 220, dtype=np.uint8)
            frames = [(frame_a, thumb_a), (frame_b, thumb_b)] * 6
            idx = {"i": 0}
            pressed_batches = []
            seen_alerts = []

            loop = GameLoop(turn_settle_delay=0.0, step_settle_delay=0.0, discovery_on_start=False)
            loop.discovery_active = False
            loop.run_temp_dir = tmp / "shots"
            loop.run_temp_dir.mkdir(parents=True, exist_ok=True)
            loop.goal_tracker = gt
            loop.dialogue_recorder = DialogueRecorder(storage_path=tmp / "signs.json")
            loop.thinking_store = ThinkingStore(traces_path=tmp / "t.jsonl", insights_path=tmp / "i.json")
            loop.web_client = WebSearchClient(cache_path=tmp / "w.json")
            loop.conversation_history = ConversationHistoryManager(storage_path=tmp / "h.json")
            loop.player.goal_tracker = loop.goal_tracker
            loop.player.dialogue_recorder = loop.dialogue_recorder
            loop.player.thinking_store = loop.thinking_store
            loop.player.web_client = loop.web_client
            loop.player.history_manager = loop.conversation_history
            loop.player.enable_web_search = False

            def next_frame():
                res = frames[idx["i"] % len(frames)]
                idx["i"] += 1
                return res

            def fake_decide(**kwargs):
                seen_alerts.append(kwargs.get("stuck_alert", ""))
                return {
                    "actions": ["A"],
                    "flat_actions": ["A"],
                    "reasoning": "Toggle poster",
                    "dialogue_text": "Fangame made with love by Jaizu.",
                    "screen_kind": "DIALOGUE",
                }

            batch_buf = []

            def rec_btn(b):
                batch_buf.append(b)
                return True

            with (
                patch.object(loop.screen_capture, "find_emulator_window", return_value=12345),
                patch.object(loop.screen_capture, "is_window_alive", return_value=True),
                patch.object(loop.screen_capture, "capture_frame", side_effect=next_frame),
                patch.object(loop.input_controller, "focus_hwnd", return_value=True),
                patch.object(loop.input_controller, "press_button", side_effect=rec_btn),
                patch.object(loop.player, "is_server_reachable", return_value=True),
                patch.object(loop.player, "decide_action", side_effect=fake_decide),
            ):
                loop.run(max_turns=9)

            self.assertTrue(any("LOOP DETECTED" in a for a in seen_alerts))
            self.assertIn("B", batch_buf)
            self.assertIn("RIGHT", batch_buf)
            self.assertGreaterEqual(gt.turns_without_change, 6)

    def test_pre_starter_menu_and_unverified_goal_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt = GoalTracker(state_path=tmp / "g.json")
            gt.update_from_turn(
                area="Red's Bedroom 2F",
                screen_kind="START_MENU",
                discovered_fact="Menu Options: Bag, Wn, Save, Option, Exit",
            )
            self.assertIn("No Pokémon yet", gt.verified_profile["party"])
            self.assertNotIn("PARTY_MENU", gt.get_uninspected_menus())

            # Unverified late-game milestones must NOT be added to completed_goals
            gt.apply_synthesized_goal(
                new_active_goal="Walk down the 2F stairs to Living Room 1F",
                next_step="Step onto orange stair mat from the right",
                confirmed_completed_goals=[
                    "Defeated Brock in Pewter Gym",
                    "Earned the Boulder Badge",
                    "Reached Viridian City",
                    "Received Starter Pokémon (Pikachu)",
                ],
            )
            self.assertFalse(gt.is_goal_completed("Defeated Brock in Pewter Gym"))
            self.assertFalse(gt.is_goal_completed("Earned the Boulder Badge"))
            self.assertFalse(gt.is_goal_completed("Received Starter Pokémon (Pikachu)"))
            self.assertEqual(gt.active_goal, "Walk down the 2F stairs to Living Room 1F")


if __name__ == "__main__":
    unittest.main()
