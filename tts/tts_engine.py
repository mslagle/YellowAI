"""
TTS Engine for YellowAI.
Synthesizes speech for live streaming commentary.
"""

import asyncio
import io
import soundfile as sf
import numpy as np
from typing import Tuple, Optional

class TTSEngine:
    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+10%"):
        self.voice = voice
        self.rate = rate

    async def _synthesize_edge_tts(self, text: str) -> Tuple[np.ndarray, int]:
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        data, samplerate = sf.read(audio_stream)
        return data, samplerate

    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Converts text to an audio numpy array and sample rate.
        Synchronous wrapper around async edge_tts synthesis.
        """
        if not text or not text.strip():
            return None, 24000

        try:
            # Clean text for speech
            clean_text = text.strip().replace('"', '').replace("'", "")
            return asyncio.run(self._synthesize_edge_tts(clean_text))
        except Exception as e:
            print(f"[TTSEngine] Error synthesizing speech: {e}")
            return None, 24000
