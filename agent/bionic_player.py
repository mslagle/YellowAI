"""
Bionic / LM Studio Local Vision Player for YellowAI.
Connects to local OpenAI-compatible vision servers (LM Studio, BionicGPT, vLLM, LocalAI)
hosted at http://localhost:1234/v1 with zero API token costs.
"""

import io
import json
import base64
from typing import Optional, Dict, Any, Tuple, Type, List
from PIL import Image
import requests
from pydantic import BaseModel

import config as app_config
from agent.gemini_player import (
    GBAButton,
    StandardTurnResponse,
    ThoughtTurnResponse,
    ShortMemoryTurnResponse,
    MediumMemoryTurnResponse,
    LongMemoryTurnResponse
)
from agent.prompts import SYSTEM_INSTRUCTION, get_system_instruction, build_turn_prompt


class BionicPlayer:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_id: str = "default",
        temperature: float = 0.2,
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.temperature = temperature
        self.timeout = timeout
        self.consecutive_errors: int = 0

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
                if isinstance(item, str):
                    flat.append(item.upper().strip())
                elif isinstance(item, dict):
                    btn = item.get("button", "WAIT")
                    rep = item.get("repeat", 1)
                    try:
                        rep = int(rep)
                    except Exception:
                        rep = 1
                    for _ in range(max(1, min(8, rep))):
                        flat.append(str(btn).upper().strip())
        elif isinstance(raw_actions, str):
            flat.append(raw_actions.upper().strip())

        if not flat:
            fallback_btn = result_data.get("button") or result_data.get("recommended_button")
            if fallback_btn:
                flat.append(str(fallback_btn).upper().strip())
            else:
                flat.append("WAIT")
        return flat

    def is_server_reachable(self) -> bool:
        """Verifies if the Bionic / LM Studio local server endpoint is active."""
        urls_to_try = [
            f"{self.base_url}/models",
            f"{self.base_url.rstrip('/v1')}/v1/models",
            self.base_url
        ]
        for url in urls_to_try:
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code in [200, 404]:
                    return True
            except Exception:
                pass
        return False

    def get_loaded_models(self) -> List[str]:
        """Fetches models loaded on the local OpenAI-compatible server."""
        try:
            url = f"{self.base_url}/models" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/models"
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [m.get("id", "") for m in data if m.get("id")]
        except Exception:
            pass
        return []

    def _resolve_model_id(self) -> str:
        """Auto-detects active model if model_id is 'default' or unspecified."""
        if self.model_id and self.model_id.lower() != "default":
            return self.model_id
        loaded = self.get_loaded_models()
        if loaded:
            return loaded[0]
        return "default"

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
        Sends frame and game state to local Bionic / LM Studio server via standard OpenAI Chat Completions Vision API.
        HARD SAFEGUARD: Refuses to send requests if emulator is not running.
        """
        if not emulator_verified:
            raise RuntimeError("[SAFEGUARD BLOCKED] Local VLM call blocked! Emulator not verified running.")

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

        # Convert PIL current image to base64 JPEG
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{img_b64}"

        user_content = []
        include_prev = previous_image is not None and (
            bool(stuck_alert) or not getattr(app_config, "SEND_PREVIOUS_IMAGE_ONLY_WHEN_STUCK", True)
        )
        if include_prev:
            prev_buffer = io.BytesIO()
            previous_image.save(prev_buffer, format="JPEG", quality=85)
            prev_b64 = base64.b64encode(prev_buffer.getvalue()).decode("utf-8")
            user_content.extend([
                {"type": "text", "text": "[PREVIOUS SCREENSHOT (Before last turn's actions)]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"}}
            ])

        user_content.extend([
            {"type": "text", "text": "[CURRENT SCREENSHOT (Live game state to act upon)]:"},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ])

        if dialogue_crop is not None:
            crop_buffer = io.BytesIO()
            dialogue_crop.convert("RGB").save(crop_buffer, format="JPEG", quality=95)
            crop_b64 = base64.b64encode(crop_buffer.getvalue()).decode("utf-8")
            user_content.extend([
                {"type": "text", "text": "[FOCUSED SPEECH BUBBLE / DIALOGUE CROP]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
            ])

        user_content.append({"type": "text", "text": prompt_text})

        is_battle = bool(battle_advisor_prompt) or "BATTLE" in (mode_hint or "").upper()
        schema_json = schema_cls.model_json_schema()
        system_msg = f"{get_system_instruction(is_battle=is_battle)}\nYou MUST respond STRICTLY in valid JSON matching this schema:\n{json.dumps(schema_json)}"

        # Prefill assistant response with ```json\n{ to fast-track generation and skip reasoning delays
        messages = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": "```json\n{"
            }
        ]

        payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "max_tokens": 500,
            "temperature": self.temperature
        }

        chat_url = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"

        try:
            resp = requests.post(chat_url, json=payload, timeout=self.timeout)
            # If server rejects assistant prefill with 400, retry without it
            if resp.status_code == 400:
                payload["messages"] = messages[:2]
                resp = requests.post(chat_url, json=payload, timeout=self.timeout)

            resp.raise_for_status()

            data = resp.json()
            message_obj = data.get("choices", [{}])[0].get("message", {})
            raw_content = message_obj.get("content", "").strip()

            # Clean markdown codeblocks and restore leading bracket if prefilled
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3].strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:].strip()
            if not raw_content.startswith("{"):
                raw_content = "{" + raw_content

            result_data = json.loads(raw_content.strip())
            flat_buttons = self.flatten_actions(result_data)
            result_data["flat_actions"] = flat_buttons
            result_data["button"] = flat_buttons[0] if flat_buttons else "WAIT"
            result_data["cadence_tag"] = cadence_tag

            # Report tokens from local server metrics
            usage = data.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", 0)
            cand_tok = usage.get("completion_tokens", 0)
            result_data["usage"] = {
                "prompt_tokens": prompt_tok,
                "candidates_tokens": cand_tok,
                "total_tokens": prompt_tok + cand_tok
            }

            self.consecutive_errors = 0
            return result_data

        except Exception as e:
            self.consecutive_errors += 1
            print(f"[BionicPlayer] Generation error on Turn {turn_id} (consecutive: {self.consecutive_errors}): {e}")

            fallback_btn = "B" if self.consecutive_errors % 2 == 1 else "WAIT"
            return {
                "button": fallback_btn,
                "flat_actions": [fallback_btn],
                "actions": [fallback_btn],
                "reasoning": f"Bionic fallback ({e})",
                "thought": "Processing with my local Bionic brain!",
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
        enable_web_search: bool = False,
        emulator_verified: bool = False
    ) -> Dict[str, Any]:
        """Emergency tactical rescue via local Bionic / LM Studio endpoint."""
        if not emulator_verified:
            raise RuntimeError("[SAFEGUARD BLOCKED] Local rescue call blocked! Emulator not verified.")

        rescue_prompt = f"""### EMERGENCY TACTICAL RESCUE (POKÉMON YELLOW RECHARGED)
Trigger: {rescue_reason}
Area: {current_area}
Quest: {active_quest}
Previous Micro-Step: {micro_step}
Recent History:
{action_history}
Alerts:
{stuck_alert}

Analyze Red's position and obstacles. Return JSON with 'recommended_button', 'situation_analysis', 'tactical_plan', 'updated_micro_step', and 'stream_thought'."""

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{img_b64}"

        payload = {
            "model": self._resolve_model_id(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a speedrunner analyzing a stuck screen. Respond strictly in JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": rescue_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            "temperature": 0.2
        }

        chat_url = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"

        try:
            resp = requests.post(chat_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()

            data = resp.json()
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            result_data = json.loads(raw_content.strip())
            flat_buttons = self.flatten_actions(result_data)
            btn = flat_buttons[0] if flat_buttons else "B"
            result_data["flat_actions"] = flat_buttons
            result_data["recommended_button"] = btn
            result_data["cadence_tag"] = "TACTICAL_RESCUE"
            return result_data

        except Exception as e:
            print(f"[BionicPlayer] Tactical rescue error: {e}")
            return {
                "situation_analysis": f"Local rescue error: {e}",
                "recommended_button": "B",
                "flat_actions": ["B"],
                "tactical_plan": "Step back and try orthogonal direction.",
                "updated_micro_step": "Back out from obstacle",
                "stream_thought": "Resetting my bearings with Bionic!",
                "cadence_tag": "TACTICAL_RESCUE_FALLBACK"
            }
