"""
Comprehensive Automated Verification Suite for Pokémon Recharged Yellow Perception & Pathfinding.
Tests ScreenAnalyzer, POIDetector, and MovementController across 37 curated real-game fixtures
spanning all 14 scene/UI modes (Title, Main Menu, Oak Intro, Naming Screen, Trainer Card,
Bedroom 2F at 6 scroll positions, Living Room 1F at center & doormat, Pallet Town Overworld
in Day/Dusk/Night, Oak's Lab Exterior & Interior, Route 1 Tall Grass/Ledges/Trees, Viridian
Poké Center & Poké Mart Exteriors, Poké Mart Interior & Shop Menu, and Wild Battles).

Run with:
    python -m unittest tests/test_recharged_yellow_dataset.py -v
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from core.screen_analyzer import (
    ScreenAnalyzer,
    CELL_WALKABLE,
    CELL_OBSTACLE,
)
from core.movement_controller import MovementController
from core.poi_detector import POIDetector


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recharged_yellow"


def load_fixture(name: str) -> Image.Image:
    path = FIXTURE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture: {path}")
    return Image.open(path).convert("RGB")


class TestRechargedYellowDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.analyzer = ScreenAnalyzer(debug_poi_path=str(Path(cls.tmp_dir.name) / "test_poi.png"))
        cls.mover = MovementController(
            input_controller=None,
            screen_capture=None,
            route_debug_path=str(Path(cls.tmp_dir.name) / "test_route.png"),
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_01_special_ui_screens(self):
        """Verifies Title Screen, Main Menu, Oak Intro, Naming Screen, Trainer Card, and Battle screens."""
        cases = [
            ("title_screen.png", "TITLE_SCREEN", "ADVANCE_DIALOGUE"),
            ("main_menu_continue.png", "TITLE_SCREEN", "ADVANCE_DIALOGUE"),
            ("oak_intro_speech.png", "TITLE_SCREEN", "ADVANCE_DIALOGUE"),
            ("naming_screen_player.png", "NAMING_SCREEN", "ADVANCE_DIALOGUE"),
            ("naming_screen_rival.png", "NAMING_SCREEN", "ADVANCE_DIALOGUE"),
            ("trainer_card.png", "TRAINER_CARD", "ADVANCE_DIALOGUE"),
            ("battle_wild_rattata_menu.png", "BATTLE", "BATTLE_ACTION"),
            ("battle_wild_pidgey_text.png", "BATTLE", "BATTLE_ACTION"),
            ("battle_move_select.png", "BATTLE", "BATTLE_ACTION"),
        ]
        for fixture_name, expected_area, expected_intent in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                scene = self.analyzer.analyze(img)
                self.assertEqual(scene.area_type, expected_area)
                self.assertIn(expected_intent, scene.available_intents)

                if expected_area == "TRAINER_CARD":
                    res = self.mover.plan_and_execute("EXIT_ROOM", None, scene, img)
                    self.assertTrue(res["success"])
                    self.assertEqual(res["planned_buttons"], ["B"])
                elif expected_area == "TITLE_SCREEN":
                    res = self.mover.plan_and_execute("ADVANCE_DIALOGUE", None, scene, img)
                    self.assertTrue(res["success"])
                    self.assertEqual(res["planned_buttons"], ["A"])

    def test_02_bedroom_2f_multi_scroll(self):
        """Verifies Bedroom 2F classification, dynamic green-rug-anchored POIs, and railing-aware A* pathing."""
        cases = [
            ("bedroom_2f_center.png", (7, 4), (10, 0), (11, 0)),
            ("bedroom_2f_top_scroll.png", (7, 4), (11, 0), (11, 0)),
            ("bedroom_2f_west_wall.png", (7, 4), (12, 0), (12, 0)),
            ("bedroom_2f_stairs_visible.png", (7, 4), (7, 2), (8, 2)),
            ("bedroom_2f_near_stairs.png", (7, 4), (6, 2), (7, 2)),
            ("bedroom_2f_on_stairs_mat.png", (7, 4), (6, 4), (7, 4)),
        ]
        for fixture_name, expected_player, expected_stairs, expected_mat in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                # Pass misleading hint ("Living Room 1F") to verify visual classification overrides stale hints
                scene = self.analyzer.analyze(img, current_area_hint="Mom's Living Room 1F")
                self.assertEqual(scene.area_type, "BEDROOM_2F")
                self.assertTrue(scene.player.detected)
                self.assertEqual((scene.player.tile_col, scene.player.tile_row), expected_player)

                pois = {p.name: p for p in scene.pois}
                self.assertIn("STAIRS_DOWN", pois)
                self.assertIn("STAIRS_MAT", pois)
                self.assertEqual((pois["STAIRS_DOWN"].tile_col, pois["STAIRS_DOWN"].tile_row), expected_stairs)
                self.assertEqual((pois["STAIRS_MAT"].tile_col, pois["STAIRS_MAT"].tile_row), expected_mat)

                res = self.mover.plan_and_execute("EXIT_ROOM", "STAIRS_DOWN", scene, img)
                self.assertTrue(res["success"])
                self.assertGreaterEqual(len(res["planned_buttons"]), 1)

                if fixture_name == "bedroom_2f_on_stairs_mat.png":
                    self.assertIn("LEFT", res["planned_buttons"])

    def test_03_living_room_1f_separation_and_door(self):
        """Verifies Living Room 1F separation from Bedroom 2F, Mom NPC localization, and Front Door mat routing."""
        cases = [
            ("living_room_1f_center.png", (2, 9), (2, 9), (6, 5)),
            ("living_room_1f_on_front_door.png", (7, 4), (7, 5), (11, 0)),
        ]
        for fixture_name, expected_door, expected_interact, expected_mom in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                scene = self.analyzer.analyze(img, current_area_hint="Pallet Town - Red's Bedroom (2F)")
                self.assertEqual(scene.area_type, "LIVING_ROOM_1F")
                self.assertTrue(scene.player.detected)
                self.assertEqual((scene.player.tile_col, scene.player.tile_row), (7, 4))

                pois = {p.name: p for p in scene.pois}
                self.assertIn("FRONT_DOOR", pois)
                self.assertIn("MOM", pois)
                self.assertEqual((pois["FRONT_DOOR"].tile_col, pois["FRONT_DOOR"].tile_row), expected_door)
                self.assertEqual(pois["FRONT_DOOR"].interaction_tile, expected_interact)
                self.assertEqual((pois["MOM"].tile_col, pois["MOM"].tile_row), expected_mom)

                res = self.mover.plan_and_execute("EXIT_ROOM", "FRONT_DOOR", scene, img)
                self.assertTrue(res["success"])
                self.assertGreaterEqual(len(res["planned_buttons"]), 1)
                if fixture_name == "living_room_1f_on_front_door.png":
                    # Red is facing UP on the front door mat, so exiting requires 2 DOWN presses (turn + step)
                    self.assertEqual(scene.player.facing, "UP")
                    self.assertEqual(res["planned_buttons"], ["DOWN", "DOWN"])

    def test_04_oaks_lab_interior(self):
        """Verifies Professor Oak's Lab interior across camera scrolls, dialogue, and Start Menu overlay."""
        cases = [
            ("oaks_lab_interior_oak_ahead.png", (10, 3), (7, 3), True),
            ("oaks_lab_interior_starter_table.png", (9, 6), (6, 6), False),
            ("oaks_lab_interior_dialogue.png", (9, 6), (6, 6), True),
            ("oaks_lab_interior_start_menu.png", (9, 6), (6, 6), False),
        ]
        for fixture_name, expected_table, expected_oak, expected_dialogue in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                scene = self.analyzer.analyze(img)
                self.assertEqual(scene.area_type, "OAKS_LAB")
                self.assertTrue(scene.player.detected)
                self.assertEqual((scene.player.tile_col, scene.player.tile_row), (7, 4))
                self.assertEqual(scene.is_dialogue, expected_dialogue)

                pois = {p.name: p for p in scene.pois}
                self.assertIn("STARTER_TABLE", pois)
                self.assertIn("PROFESSOR_OAK", pois)
                self.assertIn("LAB_EXIT", pois)
                self.assertEqual((pois["STARTER_TABLE"].tile_col, pois["STARTER_TABLE"].tile_row), expected_table)
                self.assertEqual((pois["PROFESSOR_OAK"].tile_col, pois["PROFESSOR_OAK"].tile_row), expected_oak)

                if not expected_dialogue or fixture_name == "oaks_lab_interior_oak_ahead.png":
                    res = self.mover.plan_and_execute("GO_TO_POI", "STARTER_TABLE", scene, img)
                    self.assertTrue(res["success"])
                    self.assertEqual(res["planned_buttons"][-1], "A")
                    self.assertIn("UP", res["planned_buttons"])

        # Also verify south entrance of Oak's Lab (when green Starter Table is off-screen to the North
        # and y=176..320 is black room void while area_hint is still 'Pallet Town / Kanto Overworld'):
        # Must classify as OAKS_LAB and route straight UP ['UP', 'UP', 'UP', 'UP'] instead of DOWN out the exit!
        img_south = load_fixture("oaks_lab_interior_oak_ahead.png").copy()
        arr_south = np.array(img_south)
        arr_south[100:160, 250:400] = [222, 222, 230]
        arr_south[176:, :] = [0, 0, 0]
        arr_south[48:112, 32:192] = [120, 110, 140]
        arr_south[48:112, 256:448] = [120, 110, 140]
        scene_south = self.analyzer.analyze(Image.fromarray(arr_south), current_area_hint="Pallet Town / Kanto Overworld")
        self.assertEqual(scene_south.area_type, "OAKS_LAB")
        res_south = self.mover.plan_and_execute("GO_TO_POI", "STARTER_TABLE", scene_south, Image.fromarray(arr_south))
        self.assertTrue(res_south["success"])
        self.assertEqual(res_south["planned_buttons"], ["UP", "UP", "UP", "UP"])
        res_enter = self.mover.plan_and_execute("ENTER_BUILDING", "OAKS_LAB", scene_south, Image.fromarray(arr_south))
        self.assertTrue(res_enter["success"])
        self.assertEqual(res_enter["planned_buttons"], ["UP", "UP", "UP", "UP"])

    def test_05_pokemart_interior(self):
        """Verifies Poké Mart interior clerk & exit localization and counter interaction."""
        cases = [
            ("pokemart_interior_entrance.png", (5, 2), (7, 5)),
            ("pokemart_interior_counter.png", (5, 4), (7, 7)),
            ("pokemart_interior_shop_menu.png", (5, 4), (7, 7)),
        ]
        for fixture_name, expected_clerk, expected_exit in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                scene = self.analyzer.analyze(img)
                self.assertEqual(scene.area_type, "POKEMART")
                self.assertTrue(scene.player.detected)
                self.assertEqual((scene.player.tile_col, scene.player.tile_row), (7, 4))

                pois = {p.name: p for p in scene.pois}
                self.assertIn("MART_CLERK", pois)
                self.assertIn("MART_EXIT", pois)
                self.assertEqual((pois["MART_CLERK"].tile_col, pois["MART_CLERK"].tile_row), expected_clerk)
                self.assertEqual((pois["MART_EXIT"].tile_col, pois["MART_EXIT"].tile_row), expected_exit)

                if fixture_name == "pokemart_interior_counter.png":
                    res = self.mover.plan_and_execute("TALK_TO_NPC", "MART_CLERK", scene, img)
                    self.assertTrue(res["success"])
                    self.assertEqual(res["planned_buttons"], ["LEFT", "A"])

    def test_06_overworld_doors_and_landmarks(self):
        """Verifies Overworld classification and door detection across Day/Dusk/Night in Pallet Town & Viridian City."""
        cases = [
            ("pallet_town_exited_house_day.png", "EXITED_HOUSE_DOOR", (7, 3)),
            ("pallet_town_exited_house_night.png", "EXITED_HOUSE_DOOR", (7, 3)),
            ("pallet_town_red_house_door_day.png", "BUILDING_DOOR", (7, 1)),
            ("pallet_town_red_house_door_dusk.png", "BUILDING_DOOR", (7, 1)),
            ("pallet_town_mailbox_dialogue.png", "BUILDING_DOOR", (9, 5)),
            ("pallet_town_rival_house_door.png", "BUILDING_DOOR", (7, 2)),
            ("pallet_town_oaks_lab_exterior.png", "OAKS_LAB", (14, 8)),
            ("pallet_town_oaks_lab_west_side_night.png", "OAKS_LAB", (11, 5)),
            ("pallet_town_oaks_lab_top_edge_night.png", "OAKS_LAB", (12, 0)),
            ("pallet_town_pond_south_of_lab_night.png", "OAKS_LAB", (12, 0)),
            ("pallet_town_east_of_oaks_lab_day.png", "OAKS_LAB", (2, 0)),
            ("viridian_pokecenter_exterior.png", "POKECENTER_DOOR", (11, 5)),
            ("viridian_pokemart_exterior.png", "POKEMART_DOOR", (11, 5)),
            ("viridian_pokemart_start_menu.png", "POKEMART_DOOR", (2, 3)),
        ]
        for fixture_name, expected_poi_name, expected_poi_tile in cases:
            with self.subTest(fixture=fixture_name):
                img = load_fixture(fixture_name)
                scene = self.analyzer.analyze(img, current_area_hint="Pallet Town - Professor Oak's Research Lab")
                self.assertEqual(scene.area_type, "OVERWORLD")
                self.assertTrue(scene.player.detected)
                self.assertEqual((scene.player.tile_col, scene.player.tile_row), (7, 4))

                pois = {p.name: p for p in scene.pois}
                self.assertIn(expected_poi_name, pois)
                self.assertEqual((pois[expected_poi_name].tile_col, pois[expected_poi_name].tile_row), expected_poi_tile)

                if expected_poi_name == "EXITED_HOUSE_DOOR":
                    res = self.mover.plan_and_execute("EXPLORE_SOUTHEAST", None, scene, img)
                    self.assertTrue(res["success"])
                    self.assertGreaterEqual(len(res["planned_buttons"]), 3)
                    res_lab = self.mover.plan_and_execute("GO_TO_POI", "OAKS_LAB", scene, img)
                    self.assertTrue(res_lab["success"])
                    self.assertEqual(res_lab["target"], "EXPLORE_SOUTHEAST")
                elif expected_poi_name == "OAKS_LAB":
                    # Even if the LLM orders EXPLORE_SOUTH while OAKS_LAB is on screen, MovementController snaps to OAKS_LAB
                    res_snap = self.mover.plan_and_execute("EXPLORE_SOUTH", None, scene, img)
                    self.assertTrue(res_snap["success"])
                    self.assertEqual(res_snap["target"], "OAKS_LAB")
                    if fixture_name == "pallet_town_oaks_lab_west_side_night.png":
                        self.assertEqual(scene.player.facing, "UP")
                        # 2 tiles DOWN starting from facing UP requires 3 DOWN presses (1 turn + 2 steps)
                        self.assertEqual(
                            res_snap["planned_buttons"],
                            ["DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "UP", "UP"]
                        )
                    elif expected_poi_tile == (12, 0):
                        self.assertEqual(res_snap["planned_buttons"][0], "UP")
                        self.assertEqual(res_snap["planned_buttons"][-1], "UP")
                    elif expected_poi_tile == (2, 0):
                        self.assertEqual(res_snap["planned_buttons"][0], "LEFT")
                        self.assertEqual(res_snap["planned_buttons"][-1], "UP")

    def test_07_overworld_tall_grass_pikachu_and_ledges(self):
        """Verifies walkable Tall Wild Grass, follower Pikachu passability, and impassable Pine Trees/Ledges."""
        img_grass = load_fixture("route_1_tall_grass_day.png")
        scene_grass = self.analyzer.analyze(img_grass)
        self.assertEqual(scene_grass.area_type, "OVERWORLD")
        self.assertEqual((scene_grass.player.tile_col, scene_grass.player.tile_row), (7, 4))
        # Tall wild grass on East edge (Col 14, Rows 2..5) is walkable, while Pine Tree at (9, 4) is obstacle
        self.assertEqual(scene_grass.collision_grid[2, 14], CELL_WALKABLE)
        self.assertEqual(scene_grass.collision_grid[3, 14], CELL_WALKABLE)
        self.assertEqual(scene_grass.collision_grid[4, 9], CELL_OBSTACLE)
        res_north = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_grass, img_grass)
        self.assertTrue(res_north["success"])
        self.assertGreaterEqual(len(res_north["planned_buttons"]), 3)

        img_pika = load_fixture("route_1_dusk_pikachu.png")
        scene_pika = self.analyzer.analyze(img_pika)
        self.assertEqual(scene_pika.area_type, "OVERWORLD")
        self.assertEqual((scene_pika.player.tile_col, scene_pika.player.tile_row), (7, 4))
        self.assertEqual(scene_pika.collision_grid[4, 6], CELL_WALKABLE)

        img_ledge = load_fixture("route_1_ledges_and_trees.png")
        scene_ledge = self.analyzer.analyze(img_ledge)
        self.assertEqual(scene_ledge.area_type, "OVERWORLD")
        self.assertEqual((scene_ledge.player.tile_col, scene_ledge.player.tile_row), (7, 4))
        res_west = self.mover.plan_and_execute("EXPLORE_WEST", None, scene_ledge, img_ledge)
        self.assertTrue(res_west["success"])
        self.assertGreaterEqual(len(res_west["planned_buttons"]), 2)

    def test_08_poi_detector_delegation(self):
        """Verifies POIDetector delegates cleanly to ScreenAnalyzer."""
        detector = POIDetector(debug_save_path=str(Path(self.tmp_dir.name) / "poi_deg.png"))
        img = load_fixture("living_room_1f_center.png")
        res = detector.analyze_frame(img)
        self.assertEqual(res["room_type"], "LIVING_ROOM_1F")
        self.assertTrue(res["player"]["detected"])
        self.assertEqual((res["player"]["col"], res["player"]["row"]), (7, 4))
        warp_names = [w["name"] for w in res["warps"]]
        self.assertIn("FRONT_DOOR", warp_names)
        prompt_block = detector.format_poi_prompt(res)
        self.assertIn("Col=7, Row=4", prompt_block)

    def test_09_end_to_end_starter_mom_route1_viridian_journey(self):
        """Verifies all real-game fixtures from Oak's Lab starter acquisition -> Mom's goodbye -> Route 1 -> Viridian City."""
        from core.battle_controller import BattleController
        bc = BattleController(input_controller=None)

        # 1. Oak's Lab South Entrance (real-game capture right after stepping in from Pallet Town)
        img_south = load_fixture("oaks_lab_interior_south_entrance.png")
        scene_south = self.analyzer.analyze(img_south, current_area_hint="Pallet Town / Kanto Overworld")
        self.assertEqual(scene_south.area_type, "OAKS_LAB")
        res_up = self.mover.plan_and_execute("ENTER_BUILDING", "OAKS_LAB", scene_south, img_south)
        self.assertEqual(res_up["planned_buttons"], ["UP", "UP", "UP", "UP"])

        # 2. Oak's Lab Empty Table before triggering Oak at North grass
        img_empty = load_fixture("oaks_lab_interior_empty_table.png")
        scene_empty = self.analyzer.analyze(img_empty)
        self.assertEqual(scene_empty.area_type, "OAKS_LAB")
        self.assertTrue(any("LAB_TABLE_EMPTY (OAK NOT HERE)" in n for n in scene_empty.spatial_notes))
        self.assertEqual(scene_empty.pois[0].name, "LAB_EXIT")

        # 3. Pallet Town North Tall Grass exit & Oak Wait dialogue
        img_north_exit = load_fixture("pallet_town_north_exit_night.png")
        scene_north_exit = self.analyzer.analyze(img_north_exit)
        self.assertEqual(scene_north_exit.area_type, "OVERWORLD")
        self.assertEqual(scene_north_exit.collision_grid[1, 7], CELL_WALKABLE)
        res_grass = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_north_exit, img_north_exit)
        self.assertTrue(res_grass["success"])
        self.assertIn("UP", res_grass["planned_buttons"])

        img_oak_wait = load_fixture("pallet_town_oak_wait_dialogue.png")
        scene_oak_wait = self.analyzer.analyze(img_oak_wait)
        self.assertEqual(scene_oak_wait.area_type, "OVERWORLD")
        self.assertTrue(scene_oak_wait.is_dialogue)

        # 4. Claiming Starter Pikachu, Naming Screen confirmation, and Post-Starter Lab Exit
        img_pika_portrait = load_fixture("oaks_lab_pikachu_portrait.png")
        scene_pika_portrait = self.analyzer.analyze(img_pika_portrait)
        self.assertEqual(scene_pika_portrait.area_type, "OAKS_LAB")
        self.assertTrue(scene_pika_portrait.is_dialogue)

        img_naming = load_fixture("naming_screen_player.png")
        scene_naming = self.analyzer.analyze(img_naming)
        res_naming = self.mover.plan_and_execute("ADVANCE_DIALOGUE", None, scene_naming, img_naming)
        self.assertEqual(res_naming["planned_buttons"], ["B", "START", "A"])

        for post_lab_fn in ("oaks_lab_after_starter_taken.png", "oaks_lab_after_congrats.png", "oaks_lab_pikachu_following.png"):
            with self.subTest(fixture=post_lab_fn):
                img_post = load_fixture(post_lab_fn)
                scene_post = self.analyzer.analyze(img_post)
                self.assertEqual(scene_post.area_type, "OAKS_LAB")
                self.assertTrue(any("STARTER_ALREADY_CLAIMED" in n for n in scene_post.spatial_notes))
                self.assertEqual(scene_post.pois[0].name, "LAB_EXIT")

        # 5. Pallet Town Three Buildings at Night with Follower Pikachu -> Disambiguates Mom's House (West) vs Rival's House (East)
        img_three = load_fixture("pallet_town_three_buildings_night.png")
        scene_three = self.analyzer.analyze(img_three)
        self.assertEqual(scene_three.area_type, "OVERWORLD")
        self.assertTrue(any("PIKACHU_FOLLOWING" in n for n in scene_three.spatial_notes))
        res_mom_door = self.mover.plan_and_execute("ENTER_BUILDING", "MOM_HOUSE", scene_three, img_three)
        self.assertTrue(res_mom_door["success"])
        self.assertIn("LEFT", res_mom_door["planned_buttons"])

        # 6. Mom's House Nighttime & Goodbye Cutscene Dialogues (Red female text + Blue Aide text)
        for mom_fn in (
            "living_room_1f_night_with_pikachu.png",
            "living_room_1f_aide_viridian_dialogue.png",
            "living_room_1f_mom_goodbye_dialogue.png",
        ):
            with self.subTest(fixture=mom_fn):
                img_mom = load_fixture(mom_fn)
                scene_mom = self.analyzer.analyze(img_mom)
                self.assertEqual(scene_mom.area_type, "LIVING_ROOM_1F")
                self.assertTrue(scene_mom.is_dialogue)

        # 7. Exiting Mom's House (with top-left Pallet Town banner) & routing around Mom's House roof peak
        img_banner = load_fixture("pallet_town_exited_house_banner_dusk.png")
        scene_banner = self.analyzer.analyze(img_banner)
        self.assertEqual(scene_banner.area_type, "OVERWORLD")
        self.assertNotIn("OAKS_LAB", [p.name for p in scene_banner.pois])
        res_leave_mom = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_banner, img_banner)
        self.assertTrue(res_leave_mom["success"])
        self.assertIn("LEFT", res_leave_mom["planned_buttons"])
        self.assertIn("UP", res_leave_mom["planned_buttons"])

        img_peak = load_fixture("pallet_town_northwest_corner_roof_peak.png")
        scene_peak = self.analyzer.analyze(img_peak)
        self.assertEqual(scene_peak.collision_grid[4, 12], CELL_WALKABLE)
        res_peak = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_peak, img_peak)
        self.assertTrue(res_peak["success"])
        self.assertEqual(res_peak["target"], "EXPLORE_EAST")
        self.assertGreaterEqual(len(res_peak["planned_buttons"]), 6)

        # 8. Route 1 First Ledge & Wild Bellsprout Battle Opening
        img_r1_ledge = load_fixture("route1_first_ledge_day.png")
        scene_r1_ledge = self.analyzer.analyze(img_r1_ledge)
        self.assertEqual(scene_r1_ledge.area_type, "OVERWORLD")
        self.assertEqual(scene_r1_ledge.collision_grid[3, 7], CELL_OBSTACLE)
        self.assertEqual(scene_r1_ledge.collision_grid[3, 1], CELL_WALKABLE)
        res_r1 = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_r1_ledge, img_r1_ledge)
        self.assertTrue(res_r1["success"])
        self.assertEqual(res_r1["planned_buttons"][0], "LEFT")

        for battle_fn in ("battle_opening_slide_in.png", "battle_wild_bellsprout_appeared.png"):
            with self.subTest(fixture=battle_fn):
                img_b = load_fixture(battle_fn)
                scene_b = self.analyzer.analyze(img_b)
                self.assertEqual(scene_b.area_type, "BATTLE")
                self.assertTrue(scene_b.is_battle)
                self.assertEqual(bc.plan_battle_action("RUN", frame=img_b), ["A"])

        # 9. Viridian City South Entrance & Central Arrival next to Viridian Pokémon Center
        img_vir_south = load_fixture("viridian_city_south_entrance_day.png")
        scene_vir_south = self.analyzer.analyze(img_vir_south)
        self.assertEqual(scene_vir_south.area_type, "OVERWORLD")
        res_vir_enter = self.mover.plan_and_execute("EXPLORE_NORTH", None, scene_vir_south, img_vir_south)
        self.assertTrue(res_vir_enter["success"])
        self.assertEqual(res_vir_enter["planned_buttons"][0], "LEFT")

        img_vir_center = load_fixture("viridian_city_pikachu_following_day.png")
        scene_vir_center = self.analyzer.analyze(img_vir_center)
        self.assertEqual(scene_vir_center.area_type, "OVERWORLD")
        self.assertTrue(any("PIKACHU_FOLLOWING" in n for n in scene_vir_center.spatial_notes))
        pois_vir = {p.name: p for p in scene_vir_center.pois}
        self.assertIn("POKECENTER_DOOR", pois_vir)
        self.assertEqual((pois_vir["POKECENTER_DOOR"].tile_col, pois_vir["POKECENTER_DOOR"].tile_row), (12, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
