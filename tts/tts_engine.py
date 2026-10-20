"""
TTS Engine for YellowAI.
Synthesizes speech for live streaming commentary.
"""

import asyncio
import io
import os
import subprocess
import tempfile
import soundfile as sf
import numpy as np
from typing import Tuple, Optional

class TTSEngine:
    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+10%", provider: Optional[str] = None):
        import config
        self.voice = voice
        self.rate = rate
        self.provider = (provider or getattr(config, "TTS_PROVIDER", "edge")).lower().strip()
        self._edge_disabled: bool = (self.provider == "sapi")

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

    def _synthesize_windows_sapi(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        """Offline fallback using Windows built-in SAPI speech synthesizer."""
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name

            # Sanitize single quotes for PowerShell
            clean = text.replace("'", "''").replace('"', '')
            ps_script = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SetOutputToWaveFile('{tmp_wav}'); "
                f"$s.Speak('{clean}'); "
                f"$s.Dispose()"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                timeout=8.0
            )
            if res.returncode == 0 and os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 0:
                data, samplerate = sf.read(tmp_wav)
                return data, samplerate
        except Exception as e:
            print(f"[TTSEngine] Windows SAPI fallback error: {e}")
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
        return None, 24000

    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Converts text to an audio numpy array and sample rate.
        Supports 'edge' (with automatic sticky Windows SAPI fallback on error) and 'sapi' (100% offline).
        """
        if not text or not text.strip() or self.provider == "none":
            return None, 24000

        clean_text = text.strip().replace('"', '').replace("'", "")

        # Direct Windows SAPI (offline / instant)
        if self._edge_disabled or self.provider == "sapi":
            return self._synthesize_windows_sapi(clean_text)

        # Primary Edge TTS
        try:
            return asyncio.run(self._synthesize_edge_tts(clean_text))
        except Exception as e:
            print(f"[TTSEngine] Edge TTS failed ({e}), switching to Windows SAPI for this session...")
            self._edge_disabled = True
            sapi_data, sapi_rate = self._synthesize_windows_sapi(clean_text)
            if sapi_data is not None and len(sapi_data) > 0:
                return sapi_data, sapi_rate
            return None, 24000
