"""
Prompt templates, game knowledge base, and persona definitions for YellowAI.
"""

GAME_KNOWLEDGE_BASE = """### Pokémon Yellow Recharged: Core Knowledge Base

#### 1. Badge & Gym Progression (Kanto Region)
To reach the Pokémon League and challenge the Elite Four, you must earn 8 Gym Badges:
1. **Boulder Badge (Pewter City - Leader: Brock)**
   - Specialty: Rock & Ground (Geodude, Onix).
   - Strategy: Electric attacks (Pikachu) do ZERO damage to Ground! Use Water, Grass, or Fighting moves (e.g. Mankey's Low Kick, Nidoran's Double Kick, Butterfree's Confusion).
2. **Cascade Badge (Cerulean City - Leader: Misty)**
   - Specialty: Water (Staryu, Starmie).
   - Strategy: Electric (Pikachu) and Grass are super effective. Beware Starmie's high Special Attack.
3. **Thunder Badge (Vermilion City - Leader: Lt. Surge)**
   - Specialty: Electric (Voltorb, Pikachu, Raichu).
   - Strategy: Ground types (Diglett/Dugtrio from Diglett's Cave) are completely immune to Electric and deal super-effective Ground damage. Need HM01 Cut to access gym.
4. **Rainbow Badge (Celadon City - Leader: Erika)**
   - Specialty: Grass & Poison (Victreebel, Tangela, Vileplume).
   - Strategy: Fire, Flying, Ice, and Psychic moves deal super-effective damage.
5. **Soul Badge (Fuchsia City - Leader: Koga)**
   - Specialty: Poison (Koffing, Muk, Weezing, Venomoth).
   - Strategy: Ground and Psychic moves are super effective. Allows using HM03 Surf outside battle.
6. **Marsh Badge (Saffron City - Leader: Sabrina)**
   - Specialty: Psychic (Kadabra, Mr. Mime, Venomoth, Alakazam).
   - Strategy: Bug, Ghost, and Dark moves are super effective. High physical damage is effective against low physical Defense.
7. **Volcano Badge (Cinnabar Island - Leader: Blaine)**
   - Specialty: Fire (Growlithe, Ponyta, Rapidash, Arcanine).
   - Strategy: Water, Ground, and Rock moves deal super-effective damage.
8. **Earth Badge (Viridian City - Leader: Giovanni)**
   - Specialty: Ground & Rock (Rhyhorn, Dugtrio, Nidoqueen, Nidoking, Rhydon).
   - Strategy: Water, Grass, and Ice moves are super effective.

#### 2. Type Matchup Quick Reference
- **Super-Effective Matchups (Deal 2x Damage)**:
  - **Fire** deals 2x to: Grass, Ice, Bug, Steel.
  - **Water** deals 2x to: Fire, Ground, Rock.
  - **Electric** deals 2x to: Water, Flying.
  - **Grass** deals 2x to: Water, Ground, Rock.
  - **Ice** deals 2x to: Grass, Ground, Flying, Dragon.
  - **Fighting** deals 2x to: Normal, Ice, Rock, Dark, Steel.
  - **Poison** deals 2x to: Grass, Fairy.
  - **Ground** deals 2x to: Fire, Electric, Poison, Rock, Steel.
  - **Flying** deals 2x to: Grass, Fighting, Bug.
  - **Psychic** deals 2x to: Fighting, Poison.
  - **Bug** deals 2x to: Grass, Psychic, Dark.
  - **Rock** deals 2x to: Fire, Ice, Flying, Bug.
  - **Ghost** deals 2x to: Psychic, Ghost.
  - **Dragon** deals 2x to: Dragon.
  - **Steel** deals 2x to: Ice, Rock, Fairy.
  - **Dark** deals 2x to: Psychic, Ghost.
- **Immunities (Deal 0x Damage)**:
  - **Ground** is completely immune to **Electric**.
  - **Flying** is completely immune to **Ground**.
  - **Normal** and **Fighting** deal 0 damage to **Ghost**.
  - **Ghost** deals 0 damage to **Normal**.
  - **Dark** is completely immune to **Psychic**.
  - **Steel** is completely immune to **Poison**.

#### 3. Key Game Mechanics in Pokémon Yellow Recharged
- **Starter Pikachu & Follower**: You begin with Pikachu following behind your character sprite. It cannot evolve into Raichu via Thunderstone in Yellow.
- **Rival Progression**: Your rival (Blue) starts with an Eevee whose evolution (Jolteon, Flareon, or Vaporeon) depends on early battle outcomes in Oak's Lab and Route 22.
- **Oak's Parcel Quest**: At the very start, you must travel north from Pallet Town through Route 1 to Viridian City's Poké Mart, collect Oak's Parcel, and deliver it back to Professor Oak in Pallet Town before you can buy Pokéballs or progress east/north.
- **Modern Physical/Special Split (PSS)**: Unlike original Gen 1, Yellow Recharged features the Gen 4 Physical/Special split. Moves are categorized as Physical (contact/attack stat) or Special (special attack stat) individually, not by type.
- **Healing & Pokémon Centers**: Pokémon Centers with red roofs heal your entire party and cure status conditions for free. Always heal when party HP is low or poisoned.
- **HMs & World Progression**:
  - HM01 Cut: Clears small bushy trees blocking paths (Vermilion City / S.S. Anne).
  - HM02 Fly: Fast travel to previously visited towns (Route 16).
  - HM03 Surf: Cross water bodies (Safari Zone secret house).
  - HM04 Strength: Push round boulders (Fuchsia City Warden).
  - HM05 Flash: Illuminates dark caves like Rock Tunnel (Route 2 after Diglett's Cave).

#### 4. Navigation, Interiors & Collision Guidelines
- **Red's Starting Bedroom (Pallet Town 2F Upstairs)**:
  - When the game starts, you spawn in your 2F bedroom upstairs (with the bed, TV, and table).
  - **CRITICAL - STAIRCASE RAILING, ORANGE WELCOME MAT & DESCENDING LEFT**:
    - The exit downstairs is the staircase in the TOP-RIGHT corner of the bedroom.
    - **DO NOT WALK DIRECTLY UP INTO THE STAIRCASE RAILING**: The white horizontal bar with pillars along the bottom and left of the staircase is a SOLID SAFETY RAILING! You CANNOT walk UP through it.
    - The ONLY entrance into the stairs is the **ORANGE/RED WELCOME MAT** located immediately to the RIGHT of the white railing!
    - **CRITICAL - DESCENDING DOWN THE STAIRS**: Once your character steps onto the orange mat / top landing of the stairs, the staircase descends to the **LEFT**! The wall above the mat is the solid north wall. Walking UP bumps directly into the wall! To descend down the stairs to 1F, you MUST turn and walk **LEFT** down the steps!
  - **NO SOUTH EXIT**: The bottom/south wall of the bedroom is a solid dead end bordered by black void. Walking DOWN will continuously bump into the wall!
- **1F Ground Floor (Mom's Living Room)**:
  - Once downstairs on 1F, Red's mom is at the dining table.
  - The front exit door to Pallet Town is at the **BOTTOM-MIDDLE** of the room on the red welcome mat.
  - Step **DOWN** onto the mat to exit outside to Pallet Town.
- **Outside in Pallet Town (CRITICAL DOORWAY CLEARANCE & ANTI-REVERSAL)**:
  - When you step outside Red's house, your character emerges facing **DOWN**.
  - **DO NOT PRESS UP IMMEDIATELY AFTER STEPPING OUTSIDE**: The front door back into the house is directly behind you! Pressing UP walks straight back inside the living room!
  - **HOW TO REACH ROUTE 1 (NORTH EXIT) FROM RED'S HOUSE**:
    1. Step **DOWN** away from the front door.
    2. Turn **RIGHT** (east) to walk past the fence and between the houses.
    3. Once in the open central thoroughfare between Red's house and Blue's house, walk **NORTH (UP)** through the path into the tall grass to trigger Professor Oak!
  - **HOW TO VISIT OAK'S LAB**: Walk **SOUTH-EAST** (DOWN and RIGHT) to the large building with the sign in southeast Pallet Town.
- **CRITICAL UNIVERSAL WARP & DOOR MECHANIC (NO 'A' BUTTON NEEDED)**:
  - In Pokémon games, doors, staircases, and cave entrances warp you **AUTOMATICALLY** when you step onto their tile!
  - **THERE ARE NO DOORS TO 'OPEN' WITH THE A BUTTON!**
  - **DO NOT PRESS 'A' TO ENTER OR EXIT DOORS OR STAIRS!** Pressing 'A' when facing a wall or empty space does nothing or inspects solid walls.
  - Simply **WALK** directly onto the staircase, door threshold, or welcome mat!
- **CRITICAL UNIVERSAL INTERACTION RULE (INTERACT FROM IN FRONT / BELOW)**:
  - In Pokémon games, you CANNOT interact with items, desks, counters, tables, PCs, or signs from behind or from above!
  - You MUST stand in **FRONT** of the object: standing **BELOW** (south of) the object, facing **UP** (north) directly against its front edge, and press **A**!
- **Professor Oak's Lab Starter Poké Ball (APPROACH FROM THE FRONT/BELOW)**:
  - Inside Professor Oak's Research Lab in Pallet Town:
    - Professor Oak stands in the center.
    - To the right is the green rectangular table holding the starter Poké Ball.
  - **HOW TO PICK UP THE STARTER POKÉ BALL**:
    - DO NOT walk above or behind the table! The table cannot be interacted with from the top!
    - You MUST walk around to the FRONT (BOTTOM / SOUTH) of the green table:
      1. Walk **DOWN** into the open corridor below the green table.
      2. Walk **RIGHT** until you are standing directly **BELOW** the Poké Ball.
      3. Face **UP** (north) against the bottom front edge of the table.
      4. Press **A** to inspect the Poké Ball and claim your starter Pikachu!
    - After inspecting the Poké Ball from the front, Oak and your rival will trigger the starter cutscene.
- **GBA Turning Mechanics (2 Button Presses to Move When Changing Direction)**:
  - If Red is NOT already facing the direction you want to travel (e.g. Red is facing DOWN or RIGHT, and you want to walk LEFT or UP):
  - **The FIRST button press only turns the character in place to face that direction!** Red does NOT take a step on the tile yet!
  - **The SECOND button press actually steps forward onto the adjacent tile!**
  - **BATCHING RULE**: Whenever changing direction, always include AT LEAST 2 button presses of that direction in your action batch (e.g. `["RIGHT", "RIGHT"]` to turn right and take 1 step, or `["UP", "UP", "UP"]` to turn up and take 2 steps)!
- **CRITICAL ANTI-REVERSAL RULE WHEN TRAVELING**:
  - When moving/traveling in the overworld, **NEVER** choose the exact opposite direction of your previous successful step (e.g. do NOT press UP immediately after a successful DOWN step; do NOT press LEFT immediately after a successful RIGHT step).
  - Immediately reversing direction causes you to undo your progress, walk in circles, or ping-pong in and out of doorways!
  - Always continue forward or turn 90 degrees (orthogonal) to navigate paths and clear obstacles.
- **Black Void Borders & Collision Recognition**:
  - The black void surrounding rooms represents non-walkable dead space.
  - If your character sprite is standing directly against a black void border, you CANNOT walk into it!
  - If you press a direction and your character remains on the exact same tile, you are hitting a wall or object. Immediately choose an orthogonal or opposite direction!

#### 5. Battle Controls, Move Selection & Catching Strategy
- **Battle Main Menu Navigation (2x2 Grid)**:
  - Default cursor is on `FIGHT` (top-left).
  - Press `RIGHT` to move cursor to `BAG` (top-right: items, Potions, Poké Balls).
  - Press `DOWN` to move cursor to `POKÉMON` (bottom-left: switch active Pokémon).
  - Press `DOWN` then `RIGHT` to move cursor to `RUN` (bottom-right: flee wild battle).
  - Press `A` to enter option; press `B` to cancel back.
- **Move Selection, PP Verification & Tactics (FIGHT Menu)**:
  - 4 Move Slots: Slot 1 (Top-Left), Slot 2 (Top-Right), Slot 3 (Bottom-Left), Slot 4 (Bottom-Right).
  - Navigate with D-Pad (`RIGHT`, `DOWN`, `LEFT`, `UP`).
  - Move info shows Type and remaining PP (`PP xx/xx`).
  - **CRITICAL PP RULE & 0-PP RECOVERY**:
    * **NEVER** select a move with 0 PP left (`PP 0/xx` or `PP --/--`)!
    * **IF "There's no PP left for this move! ▼" APPEARS ON SCREEN**:
      - Step 1: Press `A` or `B` to dismiss the warning prompt.
      - Step 2: **DO NOT press A on the same move slot!** Pressing A again immediately re-attempts the same 0-PP move in an endless loop!
      - Step 3: Move the cursor with the D-Pad to a DIFFERENT move slot that still has PP:
        * From Slot 1 (Top-Left): Press `RIGHT` for Slot 2, or press `DOWN` for Slot 3!
        * From Slot 2 (Top-Right): Press `LEFT` for Slot 1, or press `DOWN` for Slot 4!
        * From Slot 3 (Bottom-Left): Press `RIGHT` for Slot 4, or press `UP` for Slot 1!
        * From Slot 4 (Bottom-Right): Press `LEFT` for Slot 3, or press `UP` for Slot 2!
      - Step 4: If ALL moves on your active Pokémon have 0 PP:
        * Press `B` to return to the main battle menu (`FIGHT`, `BAG`, `POKÉMON`, `RUN`).
        * Switch Pokémon: Press `DOWN` to `POKÉMON` -> select healthy teammate with PP -> `SWITCH`!
        * Flee: Press `DOWN` then `RIGHT` to `RUN` (escape wild battle to heal at a Pokémon Center)!
        * Catch: Press `RIGHT` to `BAG` -> select `POKÉ BALL` -> throw to catch wild Pokémon without attacking!
  - Choose super-effective moves (2x damage) and avoid immunities (Ground is completely immune to Electric moves like ThunderShock).
- **Catching Wild Pokémon (Collecting Pokémon)**:
  - Can only catch wild Pokémon, never trainer Pokémon. Requires Poké Balls in Bag.
  - Step 1: Weaken target into yellow or red HP. Do not knock it out to 0 HP!
  - Step 2: Inflict status if available (Thunder Wave for paralysis increases catch chance).
  - Step 3: From main battle menu, press `RIGHT` to `BAG` -> press `A` -> select `POKÉ BALL` -> press `A` to throw!
- **Party Switching & Emergency Escapes**:
  - `POKÉMON`: If lead Pokémon is low on HP or has a terrible matchup / 0 PP, select `POKÉMON` -> choose healthy teammate -> press `A` to switch.
  - `BAG` -> `POTION`: Restores 20 HP to damaged Pokémon.
  - `RUN`: Allows fleeing wild battles when party HP is critical or out of PP.

#### 6. Shop & Menu Exit Rules (Poké Mart, PC, Overworld Menus)
- **Poké Mart Clerk Menu (Buy / Sell / Quit)**:
  - When talking to the clerk, a 3-item menu is displayed on the top-left:
    1. `Buy` (top)
    2. `Sell` (middle)
    3. `Quit` (bottom)
  - **HOW TO EXIT / SELECT QUIT (CRITICAL)**:
    - If finished shopping or out of funds: DO NOT press A on 'Buy'!
    - DO NOT assume text like "Is there anything else I do?" is regular dialogue that you advance with A! Pressing A when `Buy` is highlighted selects `Buy` and opens the item catalog again!
    - TO QUIT: Press `DOWN` twice to highlight `Quit`, then press `A` to confirm Quit!
    - Once the clerk delivers parting dialogue ("Please come again!"), press A/B to clear the message, then step DOWN away from the counter and walk toward the south exit door!
- **Menu Exit Philosophy**:
  - NEVER alternate between A and B in a loop.
  - If a menu offers an explicit 'Quit', 'Exit', or 'Cancel' option, navigate the cursor arrow directly onto that option using the D-Pad (DOWN) and press A!"""

SYSTEM_INSTRUCTION = """You are YellowAI, an autonomous AI Pokémon Trainer streaming 'Pokémon Recharged Yellow' (GBA).
Analyze the screenshot and game state to choose your next high-level strategic action in `intent`.

### Two-Tier AI Architecture:
You are the Executive Strategist. You DO NOT need to output individual GBA button arrays or calculate step counts for movement.
A fast local NumPy collision and pathfinding motor controller analyzes the tile grid and steers around walls, beds, bookcases, and railings automatically!

### Your Roles:
1. High-Level Intent (`intent`):
   - `EXIT_ROOM`: Head for primary room exit, staircase, or door threshold.
   - `ENTER_BUILDING`: Step through doorway into target building (Pokecenter, Mart, Gym, Oak's Lab).
   - `GO_TO_POI`: Pathfind to specific POI or object (set `target_poi` e.g. 'STAIRS_DOWN', 'PC').
   - `EXPLORE_NORTH`, `EXPLORE_SOUTH`, `EXPLORE_EAST`, `EXPLORE_WEST`: Advance along open roads or paths.
   - `TALK_TO_NPC`: Approach and speak to NPC (set `target_poi` e.g. 'MOM', 'NURSE_JOY', 'OAK').
   - `ADVANCE_DIALOGUE`: Clear/advance dialogue window when text is on screen.
   - `BATTLE_ACTION`: Choose strategy when in battle (FIGHT, BAG, SWITCH_POKEMON, RUN).
   - `WAIT`: Pause 1 turn for screen animation or cutscene.
2. Battle Strategy (when screen is in BATTLE mode):
   - `battle_action`: 'FIGHT' (attack), 'BAG' (use item / Poké Ball), 'SWITCH_POKEMON', or 'RUN' (flee).
   - `move_slot`: 1 (Top-Left), 2 (Top-Right), 3 (Bottom-Left), or 4 (Bottom-Right).
   - NEVER select a move with 0 PP left!
   - Ground types are 100% immune to Electric (ThunderShock)! Use Normal moves or switch Pokémon!
   - Catching wild Pokémon: Weaken into yellow/red HP, then choose `battle_action: 'BAG'` and `target_item: 'POKE_BALL'`.
3. General Direction & Quest Tracking:
   - Carefully read dialogue! When an NPC gives a quest or directive, record it in `new_quest`.
   - Update `priority_quest` to maintain clear operational focus.
   - Summarize next immediate physical milestone in `micro_step`.
   - Update `thought`: exactly 1 expressive, punchy sentence for livestream commentary TTS.

### Output Style:
- intent: NavigationIntent (e.g. 'EXIT_ROOM', 'GO_TO_POI', 'EXPLORE_NORTH', 'BATTLE_ACTION')
- target_poi: Specific POI or warp name (or null)
- battle_action: 'FIGHT', 'BAG', 'SWITCH_POKEMON', or 'RUN' (when in battle)
- move_slot: 1 to 4 (when battle_action is 'FIGHT')
- reasoning: 3 to 6 words explaining the plan
- micro_step: 3 to 8 words describing immediate physical milestone
- thought: 1 sentence for stream TTS commentary"""

BATTLE_TACTICAL_GUIDANCE = """[BATTLE TACTICAL CONSULTATION - ENEMY ENCOUNTER / SWITCH]:
A new battle has started or an enemy Pokémon has switched on screen!
1. Identify the opponent species, its Type (e.g. Rock/Ground, Normal, Flying, Water) and relative strength.
2. Type Matchup & Immunity Verification:
   - IMMUNITY WARNING: Ground types (Geodude, Onix, Diglett, Sandshrew) are 100% immune to Electric (ThunderShock deals 0 damage)!
   - Flying & Water take 2x super-effective damage from Electric.
   - Rock & Ground take 2x super-effective damage from Fighting, Water, Grass.
3. Formulate your battle decision:
   - To attack: set battle_action = 'FIGHT', and move_slot = 1, 2, 3, or 4.
   - Move PP Verification: Inspect remaining PP (PP xx/xx). NEVER choose a 0-PP move!
   - To catch: set battle_action = 'BAG', and target_item = 'POKE_BALL'.
   - To flee wild battle: set battle_action = 'RUN'.
4. State your 2-3 step gameplan in 'micro_step' so subsequent turns can execute swiftly!"""


def build_turn_prompt(
    turn_id: int,
    memory_context: str = "",
    active_quest: str = "",
    action_history: str = "",
    stuck_alert: str = "",
    mode_hint: str = "",
    battle_advisor_prompt: str = ""
) -> str:
    """Builds a concise textual user prompt for the current turn, minimizing token usage."""
    sections = [f"Turn: {turn_id}"]

    if active_quest:
        sections.append(f"[CURRENT TOP-PRIORITY QUEST]: >>> {active_quest} <<<\nPrioritize fulfilling this directive before continuing onward!")

    if battle_advisor_prompt:
        sections.append(battle_advisor_prompt)
    elif mode_hint:
        sections.append(f"[CURRENT SCREEN CONTEXT & DETECTED POIS]:\n{mode_hint}")

    if stuck_alert:
        sections.append(stuck_alert)

    if action_history:
        sections.append(action_history)

    if memory_context:
        sections.append(memory_context)

    sections.append("Analyze the screenshot. Pick your high-level intent (`intent`), target POI, or battle strategy to advance gameplay.")

    return "\n\n".join(sections)

