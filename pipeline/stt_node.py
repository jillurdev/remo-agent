import asyncio
from typing import Callable, Awaitable
from livekit import rtc
from livekit.plugins import deepgram, silero
from livekit.agents import stt, vad
from utils.logger import logger


class STTNode:
    def __init__(self, speaker_track: rtc.AudioTrack, source_lang: str):
        self.speaker_track = speaker_track
        self.source_lang = source_lang
        # Map our internal lang codes to Deepgram lang codes if necessary
        dg_lang = source_lang if source_lang != "no-translate" else "en"
        self._stt = deepgram.STT(model="nova-3", language=dg_lang)
        self._stt_stream = self._stt.stream()
        self._vad = silero.VAD.load()
        self._vad_stream = self._vad.stream()
        self._task = None
        self._read_task = None
        self._audio_stream = None
        self._on_segment = None
        self._running = False

    async def start(self, on_segment: Callable[[str, bool], Awaitable[None]]):
        self._on_segment = on_segment
        self._running = True
        self._audio_stream = rtc.AudioStream(self.speaker_track)
        self._task = asyncio.create_task(self._process_audio())
        self._read_task = asyncio.create_task(self._read_stt())
        logger.info(
            f"[STTNode] Started STT for {self.speaker_track.sid} in {self.source_lang}"
        )

    async def _process_audio(self):
        is_speaking = False
        try:
            async for event in self._audio_stream:
                if not self._running:
                    break
                frame = event.frame
                self._vad_stream.push_frame(frame)

                # Drain VAD events to determine if speaking
                # In livekit-agents, vad_stream is an async iterable or we can just iterate over it
                # Wait, this might block if we await it here. Let's run VAD loop concurrently.
                # Actually, in this simple version, let's just push frame to STT always for now,
                # or use VAD properly.
                if True:  # Placeholder for VAD logic
                    self._stt_stream.push_frame(frame)
        except Exception as e:
            logger.error(f"[STTNode] Audio processing error: {e}", exc_info=True)

    async def _read_stt(self):
        retries = 0
        while self._running and retries < 5:
            try:
                async for event in self._stt_stream:
                    if not self._running:
                        break

                    # Only forward FINAL transcripts downstream. Interim
                    # transcripts are noisy/partial ("ami", "ami jacchi",
                    # "ami jacchi bazar-e") and the frontend renders each
                    # one as a separate bubble instead of overwriting a
                    # single "live" line — that's what was causing the
                    # duplicated / broken-up sentences in the UI. Simplest
                    # fix: stop emitting interim segments at the source.
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        text = event.alternatives[0].text if event.alternatives else ""
                        if text.strip():
                            await self._on_segment(text, True)
                retries = 0  # reset retries on successful stream exit
            except Exception as e:
                retries += 1
                logger.warning(f"[STTNode] STT stream error, retry {retries}/5: {e}")
                await asyncio.sleep(2**retries)
                if retries < 5 and self._running:
                    self._stt_stream = self._stt.stream()

        if retries >= 5:
            logger.error("[STTNode] STT node failed after max retries.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._read_task:
            self._read_task.cancel()
        try:
            self._stt_stream.flush()
            await self._stt_stream.aclose()
        except Exception:
            pass
        try:
            await self._vad_stream.aclose()
        except Exception:
            pass
        logger.info("[STTNode] Stopped")
