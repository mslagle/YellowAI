"""
Ollama Vision Player for YellowAI.
Executes local multimodal models (such as Qwen2.5-VL-7B, Qwen2.5-VL-3B, or Llama 3.2 Vision)
with zero API token costs via Ollama's native REST API.
"""

import io
import json
import base64
from typing import Optional, Dict, Any, Tuple, Type, List
from PIL import Image
import requests
from pydantic import BaseModel

from agent.gemini_player import (
    GBAButton,
    StandardTurnResponse,
    ThoughtTurnResponse,
    ShortMemoryTurnResponse,
    MediumMemoryTurnResponse,
    LongMemoryTurnResponse
)
from agent.prompts import SYSTEM_INSTRUCTION, build_turn_prompt


class OllamaPlayer:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "qwen2.5-vl:7b",
        temperature: float = 0.2,
        timeout: float = 45.0
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
        """Verifies if the local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Checks if the target model is downloaded locally in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                # Match full name (e.g. qwen2.5-vl:7b) or prefix
                return any(self.model_id in m or m.startswith(self.model_id) for m in models)
            return False
        except Exception:
            return False

    def decide_action(
        self,
        turn_id: int,
        image: Image.Image,
        memory_context: str,
        active_quest: Optional[str] = None,
        action_history: str = "",
        stuck_alert: str = "",
        mode_hint: str = "",
        battle_thinking_budget: Optional[int] = None,
        battle_advisor_prompt: str = "",
        emulator_verified: bool = False
    ) -> Dict[str, Any]:
        """
        Queries local Ollama VLM with image, prompt, and JSON schema.
        HARD SAFEGUARD: Refuses to send requests if emulator is not verified alive.
        """
        if not emulator_verified:
            raise RuntimeError(
                "[SAFEGUARD BLOCKED] Local VLM call was blocked! Emulator is not verified running."
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

        # Convert PIL image to base64 JPEG
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": f"{SYSTEM_INSTRUCTION}\nYou MUST respond ONLY in valid JSON conforming to the requested schema."
                },
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [img_b64]
                }
            ],
            "format": schema_cls.model_json_schema(),
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            url = f"{self.base_url}/api/chat"
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()

            data = resp.json()
            raw_text = data.get("message", {}).get("content", "").strip()

            result_data = json.loads(raw_text)
            flat_buttons = self.flatten_actions(result_data)
            result_data["flat_actions"] = flat_buttons
            result_data["button"] = flat_buttons[0] if flat_buttons else "WAIT"
            result_data["cadence_tag"] = cadence_tag

            # Token usage reporting from Ollama metrics
            prompt_tok = data.get("prompt_eval_count", 0)
            cand_tok = data.get("eval_count", 0)
            result_data["usage"] = {
                "prompt_tokens": prompt_tok,
                "candidates_tokens": cand_tok,
                "total_tokens": prompt_tok + cand_tok
            }

            self.consecutive_errors = 0
            return result_data

        except Exception as e:
            self.consecutive_errors += 1
            print(f"[OllamaPlayer] Generation error on Turn {turn_id} (consecutive: {self.consecutive_errors}): {e}")

            fallback_btn = "B" if self.consecutive_errors % 2 == 1 else "WAIT"
            return {
                "button": fallback_btn,
                "flat_actions": [fallback_btn],
                "actions": [fallback_btn],
                "reasoning": f"Local VLM fallback ({e})",
                "thought": "Thinking offline with my local brain!",
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
        """Local tactical rescue using Ollama."""
        if not emulator_verified:
            raise RuntimeError("[SAFEGUARD BLOCKED] Local rescue call blocked! Emulator not verified.")

        rescue_prompt = f"""### EMERGENCY TACTICAL RESCUE (POKÉMON YELLOW RECHARGED)
Trigger: {rescue_reason}
Area: {current_area}
Quest: {active_quest}
Previous micro-step: {micro_step}
Recent History:
{action_history}
Alerts:
{stuck_alert}

Analyze Red's position, surroundings, and obstacles. Formulate an escape plan."""

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        rescue_schema = {
            "type": "object",
            "properties": {
                "situation_analysis": {"type": "string"},
                "recommended_button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]},
                "tactical_plan": {"type": "string"},
                "updated_micro_step": {"type": "string"},
                "stream_thought": {"type": "string"}
            },
            "required": ["situation_analysis", "recommended_button", "tactical_plan", "updated_micro_step", "stream_thought"]
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Pokémon speedrunner escaping a stuck position. Respond strictly in JSON."
                },
                {
                    "role": "user",
                    "content": rescue_prompt,
                    "images": [img_b64]
                }
            ],
            "format": rescue_schema,
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            result_data = json.loads(data.get("message", {}).get("content", "{}"))

            flat_buttons = self.flatten_actions(result_data)
            btn = flat_buttons[0] if flat_buttons else "B"
            result_data["flat_actions"] = flat_buttons
            result_data["recommended_button"] = btn
            result_data["cadence_tag"] = "TACTICAL_RESCUE"
            return result_data
        except Exception as e:
            print(f"[OllamaPlayer] Tactical rescue error: {e}")
            return {
                "situation_analysis": f"Local rescue error: {e}",
                "recommended_button": "B",
                "flat_actions": ["B"],
                "tactical_plan": "Step back and try orthogonal direction.",
                "updated_micro_step": "Back out from obstacle",
                "stream_thought": "Let me reset my bearings real quick!",
                "cadence_tag": "TACTICAL_RESCUE_FALLBACK"
            }
