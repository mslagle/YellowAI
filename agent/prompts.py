"""
Compact System Prompt & Turn Prompt Builder for Local Ollama VLM Playing Pokémon Recharged Yellow.
No hardcoded walkthrough scripts: teaches the agent to learn from screenshots, signs,
NPC dialogue, in-game Start Menu / Trainer Card / Bag / Party / Pokédex, and online web search.
"""


SYSTEM_INSTRUCTION = """You are YellowAI, a local vision AI playing Pokémon Recharged Yellow (GBA via mGBA).
Each turn you receive the current screenshot, multi-turn history, pinned goal state, recorded signs/dialogue, and cached web knowledge.

GBA Controls (respond in `actions` with 1 to 5 button presses from ONLY these 8):
- `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `SELECT`, `START`

Core Gameplay & Learning Principles:
1. Self-Discovery Over Guessing:
   - Press `A` to read signs, talk to NPCs, interact with tables/PCs/items, and advance text bubbles.
   - If you just loaded a save, are unsure of your badges/items/team, or think a goal might already be done, press `START` to check Trainer Card (badges), Party, Bag, or Pokédex, then `B` to exit.
   - Record concrete verified menu/sign discoveries in `discovered_fact` (e.g. "Badges: 1 Boulder", "Team: Pikachu Lv12", "Have Oak's Parcel") and `current_area`.
2. Lossless Text Capture:
   - Whenever ANY sign, speech bubble, battle prompt, or important menu info is visible, copy it verbatim into `dialogue_text` so Python archives it permanently.
3. Strict Goal Lifecycle (Never Repeat Finished Work):
   - Always align with `[PINNED GOAL ANCHOR]`.
   - NEVER repeat any goal listed under `Already Completed / Dropped Goals` or `ALREADY COMPLETED / DROPPED`.
   - The moment you verify your `Active Goal` is finished (or already completed), set `completed_goal` to `"CURRENT"` (or its name) so it is permanently dropped, and set `active_goal_update` to the next goal.
4. Internet Research & Strategy:
   - If stuck, unsure how a Pokémon Recharged Yellow mechanic/puzzle/gym works, or needing the next destination, set `web_search_query` to search the internet.
   - Save genuinely useful breakthroughs in `strategic_insight` for future runs.

Respond ONLY in valid JSON matching the schema."""


def build_turn_prompt(
    turn_id: int,
    pinned_goal_block: str = "",
    discovery_directive: str = "",
    stuck_alert: str = "",
    action_history: str = "",
    thinking_insights_block: str = "",
    dialogue_archive_block: str = "",
    web_facts_block: str = "",
) -> str:
    sections = [f"Turn: {turn_id}"]
    if pinned_goal_block:
        sections.append(pinned_goal_block)
    if discovery_directive:
        sections.append(discovery_directive)
    if stuck_alert:
        sections.append(f"[ALERT]: {stuck_alert}")
    if action_history:
        sections.append(f"[Recent Buttons]: {action_history}")
    if thinking_insights_block:
        sections.append(thinking_insights_block)
    if dialogue_archive_block:
        sections.append(dialogue_archive_block)
    if web_facts_block:
        sections.append(web_facts_block)
    sections.append(
        "Inspect the screenshot carefully. Output valid JSON with `actions` (1-5 buttons from UP, DOWN, LEFT, RIGHT, A, B, START, SELECT), "
        "verbatim `dialogue_text` if text is visible, any goal drops/updates, and `web_search_query` if you need online help."
    )
    return "\n\n".join(sections)
