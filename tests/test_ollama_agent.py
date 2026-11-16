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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agent.ollama_player import OllamaPlayer, TurnDecision, VALID_GBA_BUTTONS
from agent.web_search import WebSearchClient
from core.conversation_history import ConversationHistoryManager
from core.dialogue_recorder import DialogueRecorder
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
            self.assertTrue(len(web_client.search("Pokemon Recharged Yellow", max_results=2).get("results", [])) >= 1)


if __name__ == "__main__":
    unittest.main()
