"""
Gemini Vision Player for YellowAI.
Queries Gemini 3.5 Flash with game screenshots, dynamic response schemas, and latency optimization.
Strict safeguard: Refuses to send requests if emulator is not verified alive.
"""

import io
import json
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Type, List
from PIL import Image
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from agent.prompts import SYSTEM_INSTRUCTION, build_turn_prompt


class GBAButton(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    A = "A"
    B = "B"
    START = "START"
    SELECT = "SELECT"
    WAIT = "WAIT"


class NavigationIntent(str, Enum):
    EXIT_ROOM = "EXIT_ROOM"              # Head for primary room exit / stairs / door
    ENTER_BUILDING = "ENTER_BUILDING"    # Step into target building in overworld
    GO_TO_POI = "GO_TO_POI"              # Pathfind to specific POI or object
    EXPLORE_NORTH = "EXPLORE_NORTH"      # Journey north along road or path
    EXPLORE_SOUTH = "EXPLORE_SOUTH"      # Journey south along road or path
    EXPLORE_EAST = "EXPLORE_EAST"        # Journey east along road or path
    EXPLORE_WEST = "EXPLORE_WEST"        # Journey west along road or path
    EXPLORE_NORTHEAST = "EXPLORE_NORTHEAST"  # Journey northeast
    EXPLORE_NORTHWEST = "EXPLORE_NORTHWEST"  # Journey northwest
    EXPLORE_SOUTHEAST = "EXPLORE_SOUTHEAST"  # Journey southeast
    EXPLORE_SOUTHWEST = "EXPLORE_SOUTHWEST"  # Journey southwest
    TALK_TO_NPC = "TALK_TO_NPC"          # Approach and speak to NPC
    ADVANCE_DIALOGUE = "ADVANCE_DIALOGUE"# Advance/dismiss dialogue or text on screen
    OPEN_MENU = "OPEN_MENU"              # Open Start menu
    BATTLE_ACTION = "BATTLE_ACTION"      # Execute battle strategy
    WAIT = "WAIT"


class BattleActionType(str, Enum):
    FIGHT = "FIGHT"
    BAG = "BAG"
    SWITCH_POKEMON = "SWITCH_POKEMON"
    RUN = "RUN"


class StandardTurnResponse(BaseModel):
    intent: NavigationIntent = Field(
        default=NavigationIntent.EXIT_ROOM,
        description="High-level intent: EXIT_ROOM, ENTER_BUILDING, GO_TO_POI, EXPLORE_NORTH, EXPLORE_SOUTH, EXPLORE_EAST, EXPLORE_WEST, EXPLORE_NORTHEAST, EXPLORE_NORTHWEST, EXPLORE_SOUTHEAST, EXPLORE_SOUTHWEST, TALK_TO_NPC, ADVANCE_DIALOGUE, BATTLE_ACTION, WAIT."
    )
    target_poi: Optional[str] = Field(
        default=None,
        description="Target POI name if navigating to an object or NPC (e.g. 'STAIRS_DOWN', 'FRONT_DOOR', 'MOM', 'NURSE_JOY', 'OAK', 'PC')."
    )
    battle_action: Optional[BattleActionType] = Field(
        default=None,
        description="If in battle: FIGHT, BAG, SWITCH_POKEMON, or RUN."
    )
    move_slot: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="If battle_action is FIGHT, which move slot (1 to 4) to execute."
    )
    target_item: Optional[str] = Field(
        default=None,
        description="If battle_action is BAG, item to use (e.g. 'POKE_BALL', 'POTION')."
    )
    switch_target_slot: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description="If battle_action is SWITCH_POKEMON, party member slot (1 to 6) to switch into."
    )
    reasoning: str = Field(
        description="Brief explanation of your strategic plan."
    )
    micro_step: Optional[str] = Field(
        default=None,
        description="Immediate tactical milestone summary (e.g. 'Walk down stairs to 1F', 'Talk to Mom', 'Exit outside to Pallet Town')."
    )
    new_quest: Optional[str] = Field(
        default=None,
        description="If an NPC gives a new quest, command, or goal, state it here to add to the quest log."
    )
    completed_quest: Optional[str] = Field(
        default=None,
        description="If an active quest was just fulfilled, state which quest was completed to remove it."
    )
    priority_quest: Optional[str] = Field(
        default=None,
        description="If top-priority focus should be set or changed, state which quest is highest priority."
    )
    discovered_poi: Optional[str] = Field(
        default=None,
        description="If you discover a new landmark or important NPC, state it here."
    )
    items_update: Optional[str] = Field(
        default=None,
        description="If you bought or used items, summarize here (e.g. '+5 Poké Balls')."
    )
    party_update: Optional[str] = Field(
        default=None,
        description="If you caught a Pokémon or leveled up, state it here."
    )
    opponent_pokemon: Optional[str] = Field(
        default=None,
        description="If in a battle, state opposing Pokémon species on screen (e.g. 'Pidgey', 'Geodude')."
    )
    dialogue_text: Optional[str] = Field(
        default=None,
        description="If a speech bubble or dialogue window is open on screen, transcribe the exact words spoken or displayed."
    )


# Schema 2: Every 5 Turns (Thought for TTS)
class ThoughtTurnResponse(StandardTurnResponse):
    thought: str = Field(
        description="Exactly ONE expressive, punchy sentence about your current feelings, tactical plans, or struggles for your livestream audience."
    )


# Schema 3: Every 10 Turns (+ Short-Term Memory)
class ShortMemoryTurnResponse(ThoughtTurnResponse):
    short_term_memory: str = Field(
        description="Updated 1-2 sentence tactical summary of the immediate situation: room, battle state, dialogue, or obstacle."
    )


# Schema 4: Every 50 Turns (+ Medium-Term Memory)
class MediumMemoryTurnResponse(ShortMemoryTurnResponse):
    medium_term_memory: str = Field(
        description="Updated 2-3 sentence summary of current journey milestone: town/route, gym target, party condition."
    )


# Schema 5: Every 100 Turns (+ Long-Term Memory)
class LongMemoryTurnResponse(MediumMemoryTurnResponse):
    long_term_memory: str = Field(
        description="Updated 3-5 sentence strategic summary of the entire playthrough: badges won, team roster, rival arc, and grand objective."
    )


class GeminiPlayer:
    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-3.5-flash",
        thinking_budget: Optional[int] = 0
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.thinking_budget = thinking_budget
        self.consecutive_errors: int = 0
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()

    def select_schema_for_turn(self, turn_id: int) -> Tuple[Type[BaseModel], str]:
        """Determines response model schema and cadence tag based on turn counter."""
        if turn_id > 0 and turn_id % 100 == 0:
            return LongMemoryTurnResponse, "CADENCE_LONG_MEMORY"
        elif turn_id > 0 and turn_id % 50 == 0:
            return MediumMemoryTurnResponse, "CADENCE_MEDIUM_MEMORY"
        elif turn_id > 0 and turn_id % 10 == 0:
            return ShortMemoryTurnResponse, "CADENCE_SHORT_MEMORY"
        elif turn_id > 0 and turn_id % 5 == 0:
            return ThoughtTurnResponse, "CADENCE_THOUGHT"
        else:
            return StandardTurnResponse, "CADENCE_STANDARD"

    @staticmethod
    def flatten_actions(result_data: Dict[str, Any]) -> List[str]:
        raw_actions = result_data.get("actions") or result_data.get("recommended_actions")
        flat: List[str] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if isinstance(item, dict):
                    btn = item.get("button", "WAIT")
                    rep = item.get("repeat", 1)
                    try:
                        rep = int(rep)
                    except Exception:
                        rep = 1
                    rep = max(1, min(8, rep))
                    for _ in range(rep):
                        flat.append(str(btn).upper().strip())
                elif isinstance(item, str):
                    flat.append(item.upper().strip())
        elif isinstance(raw_actions, str):
            flat.append(raw_actions.upper().strip())

        if not flat:
            fallback_btn = result_data.get("button") or result_data.get("recommended_button")
            if fallback_btn:
                flat.append(str(fallback_btn).upper().strip())
            else:
                flat.append("WAIT")
        return flat

    def decide_action(
        self,
        turn_id: int,
        image: Image.Image,
        previous_image: Optional[Image.Image] = None,
        memory_context: str = "",
        active_quest: Optional[str] = None,
        action_history: str = "",
        stuck_alert: str = "",
        mode_hint: str = "",
        battle_thinking_budget: Optional[int] = None,
        battle_advisor_prompt: str = "",
        dialogue_crop: Optional[Image.Image] = None,
        emulator_verified: bool = False
    ) -> Dict[str, Any]:
        """
        Sends frame, memory context, active quest, action history, and collision alerts to Gemini 3.5 Flash.
        HARD SAFEGUARD: Raises error and refuses to call API if emulator_verified is False.
        """
        if not emulator_verified:
            raise RuntimeError(
                "[SAFEGUARD BLOCKED] Gemini API call was blocked! The emulator is not verified to be running."
            )

        schema_cls, cadence_tag = self.select_schema_for_turn(turn_id)
        prompt_text = build_turn_prompt(
            turn_id=turn_id,
            memory_context=memory_context,
            active_quest=active_quest or "",
            action_history=action_history,
            stuck_alert=stuck_alert,
            mode_hint=mode_hint,
            battle_advisor_prompt=battle_advisor_prompt
        )

        # Convert PIL current image to JPEG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        image_part = types.Part.from_bytes(
            data=img_byte_arr.getvalue(),
            mime_type="image/jpeg"
        )

        contents = []
        if previous_image is not None:
            prev_byte_arr = io.BytesIO()
            previous_image.save(prev_byte_arr, format='JPEG', quality=85)
            prev_part = types.Part.from_bytes(
                data=prev_byte_arr.getvalue(),
                mime_type="image/jpeg"
            )
            contents.append("[PREVIOUS SCREENSHOT (Before last turn's actions)]:")
            contents.append(prev_part)

        contents.append("[CURRENT SCREENSHOT (Live game state to act upon)]:")
        contents.append(image_part)

        if dialogue_crop is not None:
            crop_byte_arr = io.BytesIO()
            dialogue_crop.convert("RGB").save(crop_byte_arr, format='JPEG', quality=95)
            crop_part = types.Part.from_bytes(
                data=crop_byte_arr.getvalue(),
                mime_type="image/jpeg"
            )
            contents.append("[FOCUSED SPEECH BUBBLE / DIALOGUE CROP]:")
            contents.append(crop_part)

        contents.append(prompt_text)

        try:
            config_params = {
                "system_instruction": SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": schema_cls,
                "temperature": 0.3,
            }

            effective_thinking = battle_thinking_budget if (battle_thinking_budget is not None and battle_thinking_budget > 0) else self.thinking_budget
            if effective_thinking is not None and effective_thinking > 0:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=effective_thinking
                )

            config = types.GenerateContentConfig(**config_params)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )

            # Parse JSON output
            result_data = json.loads(response.text)
            if battle_thinking_budget and battle_thinking_budget > 0:
                cadence_tag = f"{cadence_tag}_BATTLE_THINKING"
            result_data["cadence_tag"] = cadence_tag

            flat_buttons = self.flatten_actions(result_data)
            result_data["flat_actions"] = flat_buttons
            result_data["button"] = flat_buttons[0] if flat_buttons else "WAIT"

            # Capture token usage metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                result_data["usage"] = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "candidates_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
                }

            self.consecutive_errors = 0
            return result_data

        except Exception as e:
            self.consecutive_errors += 1
            print(f"[GeminiPlayer] Generation error on Turn {turn_id} (consecutive: {self.consecutive_errors}): {e}")

            err_str = str(e).upper()
            if "RESOURCE_EXHAUSTED" in err_str or "402" in err_str:
                if self.consecutive_errors >= 3:
                    raise RuntimeError(
                        f"[YellowAI CRITICAL]: Prepayment credits are depleted (402 RESOURCE_EXHAUSTED). "
                        f"Please visit https://ai.studio/projects to manage billing/credits before resuming."
                    )

            # Alternate fallback buttons between B and WAIT so we never spam A into 0-PP loops or dialog locks
            fallback_btn = "B" if self.consecutive_errors % 2 == 1 else "WAIT"
            return {
                "button": fallback_btn,
                "flat_actions": [fallback_btn],
                "actions": [{"button": fallback_btn, "repeat": 1}],
                "reasoning": "Fallback due to API error.",
                "thought": "Hold on chat, my brain had a quick hiccup!",
                "cadence_tag": "ERROR_FALLBACK"
            }

    def consult_tactical_rescue(
        self,
        image: Image.Image,
        current_area: str,
        active_quest: str,
        micro_step: str,
        action_history: str,
        stuck_alert: str,
        rescue_reason: str,
        thinking_budget: int = 2048,
        enable_web_search: bool = True,
        emulator_verified: bool = False
    ) -> Dict[str, Any]:
        """
        LAST DITCH EFFORT / TACTICAL RESCUE CONSULTATION:
        When the AI detects it is lost, unable to move, or cannot find the objective:
        Enables a medium-sized thinking budget (e.g. 2048 tokens) and Google Search grounding
        to analyze the current visual screenshot, deduce the true path/solution, and search the
        internet for walkthroughs, maps, or ROM-hack-specific clues (Pokmon Yellow Recharged).
        """
        if not emulator_verified:
            raise RuntimeError(
                "[SAFEGUARD BLOCKED] Tactical rescue call was blocked! Emulator is not verified to be running."
            )

        rescue_prompt = f"""### EMERGENCY TACTICAL RESCUE & STRATEGIC CONSULTATION (POKMON YELLOW RECHARGED)

[SITUATION NOTICE]: The player character is currently LOST, UNABLE TO MOVE, or CANNOT FIND THE OBJECTIVE.
Rescue Trigger: {rescue_reason}
Current Area: {current_area}
Active Priority Quest: {active_quest}
Previous Micro-Step Target: {micro_step}

[Recent Action History]:
{action_history}

[Collision / Watchdog Alert]:
{stuck_alert}

You are an expert Pokmon Speedrunner and Game Guide navigating 'Pokmon Recharged Yellow' (GBA ROM hack created by Jaizu, built on the Pokmon Emerald engine with modernized Gen 4 physical/special mechanics).

Your mission is to perform a deep tactical diagnosis using the provided screenshot and internet search (if needed):
1. VISUAL INSPECTION:
   - Identify exactly where Red and Pikachu are on the screen.
   - Note what physical tiles, walls, trees, ledges, water, NPCs, or dialogue boxes are surrounding Red.
   - Detect what is blocking Red's movement in previous attempts.
2. INTERNET SEARCH & GAME KNOWLEDGE:
   - If unsure about the map layout, quest trigger, or building entrance in Pokmon Yellow Recharged, consult the web for clues.
3. ESCAPE PLAN & NEXT OBJECTIVE:
   - Formulate a clear 2-3 step physical sequence to escape this impasse.
   - Specify the EXACT single button to press right now to initiate the rescue.

You MUST respond strictly in valid JSON matching this schema:
{{
  "clues_found": "1-2 sentences of key clues from web search or game knowledge about this location/quest",
  "situation_analysis": "Deep 1-2 sentence analysis explaining why Red was stuck/lost and what path is open",
  "recommended_button": "ONE of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT",
  "tactical_plan": "Step-by-step 2 to 3 step physical maneuver to reach the objective",
  "updated_micro_step": "3 to 8 words specifying the immediate physical maneuver (e.g. 'Step DOWN to dirt road then turn RIGHT')",
  "stream_thought": "One witty, confident sentence for the livestream audience explaining the tactical pivot"
}}"""

        # Convert PIL image to JPEG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        image_part = types.Part.from_bytes(
            data=img_byte_arr.getvalue(),
            mime_type="image/jpeg"
        )

        config_params: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }

        if thinking_budget and thinking_budget > 0:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        if enable_web_search:
            config_params["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        try:
            config = types.GenerateContentConfig(**config_params)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[rescue_prompt, image_part],
                config=config,
            )

            result_data = json.loads(response.text)
            flat_buttons = self.flatten_actions(result_data)
            btn = flat_buttons[0] if flat_buttons else "B"
            if btn not in GBAButton.__members__:
                btn = "B"
            result_data["flat_actions"] = flat_buttons
            result_data["recommended_button"] = btn
            result_data["cadence_tag"] = "TACTICAL_RESCUE"
            return result_data

        except Exception as e:
            print(f"[GeminiPlayer] Tactical rescue error: {e}")
            return {
                "clues_found": "Web consultation encountered an exception.",
                "situation_analysis": f"Emergency fallback triggered due to error: {e}",
                "recommended_button": "B",
                "flat_actions": ["B"],
                "actions": [{"button": "B", "repeat": 1}],
                "tactical_plan": "Press B and step away from barrier.",
                "updated_micro_step": "Back out from obstacle",
                "stream_thought": "Chat, let me reset my bearings real quick!",
                "cadence_tag": "TACTICAL_RESCUE_FALLBACK"
            }
