import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

import config
from core.memory_manager import MemoryManager
from core.movement_controller import MovementController
from core.screen_analyzer import (
    ScreenAnalyzer,
    SceneAnalysis,
    PlayerState,
    CELL_WALKABLE,
    CELL_OBSTACLE,
)
from emulator.input_controller import InputController


class TestHybridAndMemoryFeatures(unittest.TestCase):
    """Unit tests for canonical quest state machine, rolling dialogue log, empirical obstacles, and background input."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "memory_state.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_quest_deduplication_and_sticky_tombstones(self):
        """Verifies semantic quest deduplication, purge-all-matches completion, and sticky tombstones."""
        mem = MemoryManager(storage_path=self.state_file)
        mem.quests.clear()
        mem.completed_milestones.clear()

        # 1. Adding semantically identical quests for Oak's Parcel should deduplicate to 1 canonical quest
        mem.add_quest("Deliver Oak's Parcel to Professor Oak")
        mem.add_quest("Bring Oak's Parcel back to Prof. Oak in Pallet Town")
        mem.add_quest("Deliver Oak's Parcel to Professor Oak.")
        self.assertEqual(
            len(mem.quests),
            1,
            f"Expected 1 deduplicated quest, got: {mem.quests}",
        )

        # 2. Completing with a slightly different phrasing should remove the parcel quest, record tombstone,
        #    and automatically advance to the next canonical milestone (Pewter City).
        mem.complete_quest("Return the parcel to Oak")
        self.assertTrue(
            all("parcel" not in q.lower() for q in mem.quests),
            f"Parcel quest should be purged after completion: {mem.quests}",
        )
        self.assertIn("deliver_oaks_parcel", mem.completed_milestones)
        # Prerequisite milestones should also be rolled up automatically
        self.assertIn("reach_viridian_mart", mem.completed_milestones)
        self.assertIn("claim_pikachu_and_beat_rival", mem.completed_milestones)

        # 3. Attempting to re-add a completed milestone quest must be blocked by the tombstone
        quest_count_before = len(mem.quests)
        mem.add_quest("Deliver Oak's Parcel to Professor Oak in Pallet Town")
        mem.add_quest("Pick up Oak's Parcel from Viridian Poké Mart")
        self.assertEqual(
            len(mem.quests),
            quest_count_before,
            f"Sticky tombstones should block re-adding completed quests, got: {mem.quests}",
        )
        self.assertTrue(all("parcel" not in q.lower() for q in mem.quests))

    def test_dialogue_milestone_auto_completion(self):
        """Verifies deterministic dialogue triggers auto-complete milestones and purge stale quests."""
        mem = MemoryManager(storage_path=self.state_file)
        mem.quests = [
            "Deliver Oak's Parcel to Professor Oak in Pallet Town (high priority)",
            "Take the parcel back to Oak",
        ]
        mem.completed_milestones.clear()

        # Simulate Oak taking the parcel / giving the Pokedex in dialogue
        mem.record_dialogue(
            turn_id=42,
            text="OAK: Oh! That's the custom Poké Ball I ordered! Thank you! Here is the Pokédex!",
            area="OAKS_LAB",
        )

        self.assertIn("deliver_oaks_parcel", mem.completed_milestones)
        self.assertTrue(
            all("parcel" not in q.lower() for q in mem.quests),
            f"All parcel quests should be purged when Oak's completion dialogue triggers: {mem.quests}",
        )

    def test_rolling_dialogue_log_merging_and_filtering(self):
        """Verifies multi-page dialogue merging, battle-spam filtering, and prompt formatting."""
        mem = MemoryManager(storage_path=self.state_file)
        mem.dialogue_history.clear()

        # 1. Multi-page dialogue in the same scene within 3 turns should merge into 1 entry
        mem.record_dialogue(10, "OAK: Hey! Wait! Don't go out into the tall grass!", area="OVERWORLD")
        mem.record_dialogue(11, "OAK: It's unsafe! Wild Pokémon live in tall grass!", area="OVERWORLD")
        self.assertEqual(len(mem.dialogue_history), 1)
        self.assertIn("Don't go out", mem.dialogue_history[0]["text"])
        self.assertIn("Wild Pokémon live in tall grass", mem.dialogue_history[0]["text"])
        self.assertEqual(mem.dialogue_history[0]["turn"], 11)

        # 2. Battle spam should be filtered out
        mem.record_dialogue(15, "Wild PIDGEY appeared!", area="OVERWORLD", screen_mode="BATTLE")
        mem.record_dialogue(16, "What will PIKACHU do? FIGHT PKMN ITEM RUN", area="OVERWORLD")
        mem.record_dialogue(17, "PIKACHU used THUNDERSHOCK! It's super effective!", area="OVERWORLD")
        self.assertEqual(len(mem.dialogue_history), 1, "Battle spam should not pollute dialogue_history")

        # 3. Verify prompt context includes Recent Dialogue Log and Walkthrough Guide
        routine_prompt = mem.get_memory_context_prompt(full_dialogue=False)
        self.assertIn("[Recent Dialogue Log", routine_prompt)
        self.assertIn("Wild Pokémon live in tall grass", routine_prompt)
        self.assertIn("Walkthrough Guide:", routine_prompt)

    def test_empirical_obstacle_learning_and_raw_buttons(self):
        """Verifies MovementController learns blocked steps and supports RAW_BUTTONS passthrough."""
        mc = MovementController()

        # Construct a synthetic 10x15 walkable grid where player is at col=7, row=4
        grid = np.full((10, 15), CELL_WALKABLE, dtype=np.uint8)
        scene = SceneAnalysis(
            area_type="OVERWORLD",
            area_description="Test Overworld",
            collision_grid=grid,
            player=PlayerState(detected=True, tile_col=7, tile_row=4, pixel_x=240, pixel_y=160, facing="UP"),
            pois=[],
        )

        # If stepping UP from (col=7, row=4) to (col=7, row=3) is blocked, record_blocked_step marks (7, 3)
        blocked_tile = mc.record_blocked_step(player_tile=(7, 4), direction_btn="UP")
        self.assertEqual(blocked_tile, (7, 3))
        self.assertIn((7, 3), mc.empirical_obstacles)

        effective_grid = mc.get_effective_grid(scene)
        self.assertEqual(effective_grid[3, 7], CELL_OBSTACLE)

        # Shifting when the player moves RIGHT (dx=+1, dy=0) shifts the obstacle relative to camera by (-1, 0) -> (6, 3)
        mc.shift_empirical_obstacles("RIGHT", camera_scrolled=True)
        self.assertIn((6, 3), mc.empirical_obstacles)

        # RAW_BUTTONS intent should return RAW_BUTTONS_PASSTHROUGH so GameLoop executes raw actions directly
        dummy_frame = Image.new("RGB", (480, 320), (0, 0, 0))
        res = mc.plan_and_execute("RAW_BUTTONS", None, scene, dummy_frame)
        self.assertEqual(res["status"], "RAW_BUTTONS_PASSTHROUGH")

    def test_generic_indoor_fallback(self):
        """Verifies unseen indoor rooms with black borders classify as GENERIC_INDOOR rather than LIVING_ROOM_1F."""
        analyzer = ScreenAnalyzer(debug_poi_path=os.path.join(self.temp_dir.name, "test_annotated.png"))
        # Create a 480x320 synthetic indoor frame: black side/top pillarboxes, neutral gray interior floor
        arr = np.zeros((320, 480, 3), dtype=np.uint8)
        arr[48:272, 96:384] = (170, 170, 168)  # Neutral indoor floor without Living Room table, Mart shelves, or Bedroom rug
        frame = Image.fromarray(arr)

        analysis = analyzer.analyze(frame)
        self.assertEqual(analysis.area_type, "GENERIC_INDOOR")
        self.assertTrue(len(analysis.pois) > 0, "GENERIC_INDOOR should provide exploratory/exit POIs")

    def test_background_input_lparam_and_focus_free_mode(self):
        """Verifies InputController builds proper Win32 lParam flags and does not require foreground focus."""
        ic = InputController(key_mappings=config.KEY_MAPPINGS, input_mode="postmessage")
        # Arrow keys (0x26 = VK_UP) should set bit 24 (extended key) on WM_KEYDOWN
        lparam_down_up_arrow = ic._build_key_lparam(0x26, is_keyup=False)
        self.assertTrue((lparam_down_up_arrow & (1 << 24)) != 0, "Extended key bit 24 should be set for arrow keys")

        # Keyup should set bits 30 and 31
        lparam_up_z = ic._build_key_lparam(0x5A, is_keyup=True)
        self.assertTrue((lparam_up_z & (1 << 30)) != 0)
        self.assertTrue((lparam_up_z & (1 << 31)) != 0)

    def test_reset_progress(self):
        """Verifies reset_all_progress resets session memory and overlay state."""
        from reset_game import reset_all_progress

        # Populate memory with advanced state
        mem = MemoryManager(storage_path=self.state_file)
        mem.quests = ["Some advanced quest"]
        mem.completed_milestones = ["leave_bedroom", "deliver_oaks_parcel"]
        mem.save()

        # Run reset
        reset_all_progress()

        # Reload and check starting state
        reloaded = MemoryManager(storage_path=config.MEMORY_FILE)
        reloaded.load()
        self.assertEqual(len(reloaded.completed_milestones), 0)
        self.assertEqual(len(reloaded.quests), 1)
        self.assertIn("bedroom", reloaded.quests[0].lower())


if __name__ == "__main__":
    unittest.main()
