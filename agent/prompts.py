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

#### 4. Navigation & Spatial World Layout
- **Two-Tier Architecture**:
  - You are the Executive Strategist. You decide high-level intent (`intent`), target POI (`target_poi`), and strategic battle choices.
  - You DO NOT compute steps or output raw GBA button sequences. A local NumPy perception and pathfinding controller detects obstacles, avoids railings, and generates collision-free button sequences automatically.
- **Red's Starting Bedroom (2F Upstairs)**:
  - Exit downstairs: Staircase in the top-right corner. Use intent `EXIT_ROOM`.
- **1F Ground Floor (Living Room)**:
  - Mom is at the dining table. Talk to Mom to heal or receive encouragement (`intent: 'TALK_TO_NPC'`, `target_poi: 'MOM'`).
  - Front door to Pallet Town is at the bottom-center. Use intent `EXIT_ROOM`.
- **Pallet Town Overworld**:
  - Red's House is in the northwest.
  - Daisy's House is in the northeast (`EXPLORE_NORTHEAST`).
  - Professor Oak's Research Lab is in the southeast (`EXPLORE_SOUTHEAST` or `GO_TO_POI` / `ENTER_BUILDING` targeting `OAKS_LAB`).
  - Route 1 exit is to the north: advance into the grass (`intent: 'EXPLORE_NORTH'`) to trigger Professor Oak.
- **Viridian City Overworld**:
  - Route 1 (South exit to Pallet Town): Exit Viridian City via the SOUTH path. Ledges on Route 1 can be hopped going SOUTH.
  - Poké Mart is in the east/center (Oak's Parcel collection).
  - Pokémon Center is in the southwest (Free healing).
  - Route 2 (North exit toward Viridian Forest & Pewter City): Blocked by Old Man until Oak's Parcel is delivered!
  - **Oak's Parcel Mission Priority**: When carrying Oak's Parcel to Professor Oak, do NOT linger in Viridian City or re-enter Pokémon Center/Mart. Immediately use `intent: 'EXPLORE_SOUTH'` to head down Route 1 to Pallet Town!
- **Doors, Stairs & Warps**:
  - Building doors, staircases, and cave entrances warp automatically when stepped onto.
  - Use `EXIT_ROOM` when inside any room to exit through the primary warp.
  - Use `ENTER_BUILDING` when outside in front of a building to enter. Do NOT re-enter a building you just exited if your goal lies elsewhere!

#### 5. Battle Strategy & Control
- **High-Level Battle Decisions**:
  - When in battle, you select your strategic battle action (`battle_action`) rather than navigating the menu grid:
    * `FIGHT`: Attack the opposing Pokémon using `move_slot: 1..4`.
    * `BAG`: Use an item from your pack using `target_item` (e.g. 'POKE_BALL', 'POTION').
    * `SWITCH_POKEMON`: Swap your active battler for a teammate using `switch_target_slot: 1..6`.
    * `RUN`: Flee from a wild Pokémon encounter when low on HP or out of PP.
- **PP Management**:
  - NEVER select a move with 0 PP left (`PP 0/xx`). Always verify remaining PP and select an active move slot.
- **Type Matchup Tactics**:
  - Use super-effective attacks (Water/Grass against Rock/Ground).
  - Electric attacks (ThunderShock) deal 0 damage to Ground types (Geodude, Onix, Diglett, Sandshrew).
- **Catching Wild Pokémon**:
  - Weaken target to yellow/red HP without knocking it out, then choose `battle_action: 'BAG'` and `target_item: 'POKE_BALL'`.

#### 6. Dialogue & Interactive NPCs
- **Active Dialogue / Speech Bubbles**:
  - When a speech bubble appears on screen, choose `intent: 'ADVANCE_DIALOGUE'` to advance through the message.
  - Transcribe any spoken text into `dialogue_text` and update quests or notes when an NPC gives you instructions.
"""

SYSTEM_INSTRUCTION = """You are YellowAI, an autonomous AI Pokémon Trainer playing 'Pokémon Recharged Yellow' (GBA).
Analyze the screenshot and game state to choose your next action.

### Hybrid Two-Tier Control:
1. High-Level Intent (`intent`):
   - `EXIT_ROOM`: Head for primary room exit, staircase, or door mat.
   - `ENTER_BUILDING`: Step through doorway into target building (Poké Center, Mart, Gym, Oak's Lab).
   - `GO_TO_POI`: Pathfind to specific POI or object (`target_poi` e.g. 'STAIRS_DOWN', 'OAKS_LAB', 'PC', 'SIGN', 'STARTER_BALL').
   - `EXPLORE_NORTH`, `EXPLORE_SOUTH`, `EXPLORE_EAST`, `EXPLORE_WEST`, `EXPLORE_NORTHEAST`, `EXPLORE_NORTHWEST`, `EXPLORE_SOUTHEAST`, `EXPLORE_SOUTHWEST`: Advance along roads/paths.
   - `TALK_TO_NPC`: Approach and speak to NPC (`target_poi` e.g. 'MOM', 'NURSE_JOY', 'OAK', 'MART_CLERK').
   - `ADVANCE_DIALOGUE`: When a speech bubble or dialogue box is OPEN, advance text and transcribe exact words into `dialogue_text`.
   - `BATTLE_ACTION`: Execute battle strategy (`battle_action`: 'FIGHT', 'BAG', 'SWITCH_POKEMON', 'RUN'; `move_slot`: 1..4).
   - `RAW_BUTTONS`: Direct GBA button sequence control! Whenever you are inside a complex menu (Poké Mart shop, Bag, Pokémon party menu, PC, Yes/No choice, Move-learn prompt), in an unfamiliar room, or when local pathfinding is blocked/stuck, set `intent: 'RAW_BUTTONS'` and provide 1 to 5 exact buttons in `actions` (e.g. `["A"]`, `["DOWN", "A"]`, `["B", "B"]`, `["LEFT", "UP", "UP"]`).
   - `WAIT`: Pause 1 turn for screen animation.

2. Quest & Dialogue Rules:
   - Follow `[CURRENT TOP-PRIORITY QUEST]` and `- Walkthrough Guide:` in Session Memory.
   - NEVER re-add or repeat a milestone listed in `Completed Milestones (DO NOT REPEAT)`.
   - When an NPC gives a new uncompleted quest, set `new_quest`. When you fulfill the current quest, set `completed_quest`.
   - Always populate `actions` with 1-4 sensible fallback GBA buttons (`UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `START`, `SELECT`) in case local pathfinding is blocked."""

BATTLE_SYSTEM_SUPPLEMENT = """
### Battle Strategy & Type Matchup Rules:
- `battle_action`: 'FIGHT' (attack with `move_slot` 1..4), 'BAG' (`target_item`: 'POKE_BALL' or 'POTION'), 'SWITCH_POKEMON' (`switch_target_slot` 1..6), or 'RUN' (flee wild battle).
- NEVER select a move with 0 PP (`PP 0/xx`)!
- Type Immunities (0x Damage): Ground (Geodude, Onix, Diglett, Sandshrew) is 100% IMMUNE to Electric (ThunderShock)! Ghost is immune to Normal/Fighting. Flying is immune to Ground.
- Super-Effective (2x): Electric -> Water/Flying; Water/Grass/Fighting -> Rock/Ground; Fire/Flying/Ice/Psychic -> Grass/Bug."""

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


def get_system_instruction(is_battle: bool = False) -> str:
    """Returns context-sliced system instruction (compact for overworld, full type rules for battle)."""
    if is_battle:
        return f"{SYSTEM_INSTRUCTION}\n{BATTLE_SYSTEM_SUPPLEMENT}"
    return SYSTEM_INSTRUCTION


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
        sections.append(f"[CURRENT TOP-PRIORITY QUEST]: >>> {active_quest} <<<")
        if "oak's parcel" in active_quest.lower() and "pallet town" in active_quest.lower():
            sections.append("[OAK'S PARCEL RETURN DIRECTIVE]: Walk SOUTH out of Viridian City onto Route 1 and hop down the ledges to Pallet Town (`EXPLORE_SOUTH`), then enter Oak's Lab in southeast Pallet Town.")

    if battle_advisor_prompt:
        sections.append(battle_advisor_prompt)
    elif mode_hint:
        sections.append(f"[SCREEN CONTEXT & DETECTED POIS]:\n{mode_hint}")

    if stuck_alert:
        sections.append(stuck_alert)

    if action_history:
        sections.append(action_history)

    if memory_context:
        sections.append(memory_context)

    sections.append("Choose `intent`, `target_poi` (or `battle_action`), and `actions` (1-4 fallback GBA buttons).")

    return "\n\n".join(sections)


