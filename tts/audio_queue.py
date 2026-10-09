"""
Asynchronous Audio Queue for YellowAI.
Ensures TTS commentary plays smoothly in the background without blocking the game loop.
"""

import queue
import threading
import time
from typing import Optional
import sounddevice as sd
from tts.tts_engine import TTSEngine

class AudioQueueManager:
    def __init__(self, tts_engine: Optional[TTSEngine] = None):
        self.tts_engine = tts_engine or TTSEngine()
        self.queue = queue.Queue(maxsize=5)  # Drop older thoughts if backlog grows
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background audio playback thread."""
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[AudioQueue] Background TTS player started.")

    def stop(self):
        """Stops the audio worker cleanly."""
        self.is_running = False
        # Put sentinel to unblock worker
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        print("[AudioQueue] Background TTS player stopped.")

    def enqueue_thought(self, thought: str):
        """Adds a thought quote to the speech queue."""
        if not thought or not thought.strip() or not self.is_running:
            return
        
        # If queue is full, drop oldest item to maintain real-time commentary
        if self.queue.full():
            try:
                dropped = self.queue.get_nowait()
                print(f"[AudioQueue] Dropping stale thought to keep commentary fresh.")
            except queue.Empty:
                pass

        try:
            self.queue.put_nowait(thought.strip())
        except queue.Full:
            pass

    def _worker_loop(self):
        while self.is_running:
            try:
                thought = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if thought is None or not self.is_running:
                break

            try:
                print(f"\n[TTS Commentator Speaking]: \"{thought}\"")
                audio_data, sample_rate = self.tts_engine.synthesize(thought)
                if audio_data is not None and len(audio_data) > 0:
                    sd.play(audio_data, sample_rate)
                    sd.wait()  # Wait for speech playback to finish before next queued line
            except Exception as e:
                print(f"[AudioQueue] Playback error: {e}")
            finally:
                self.queue.task_done()
