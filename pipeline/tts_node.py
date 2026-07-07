import asyncio
from livekit import rtc
from livekit.plugins import cartesia
from livekit.agents import tts
from utils.logger import logger


class TTSNode:
    def __init__(
        self, target_lang: str, room: rtc.Room, track_name: str, speaker_id: str
    ):
        self.target_lang = target_lang
        self.room = room
        self.track_name = track_name
        self.speaker_id = speaker_id
        cartesia_lang = target_lang.split("-")[0].lower()
        self._tts = cartesia.TTS(model="sonic-3.5", language=cartesia_lang)
        self._audio_source = None
        self._track = None
        self._running = False

    async def start(self):
        self._running = True
        logger.info(f"[TTSNode] Started TTS for {self.track_name} ({self.target_lang})")

    async def synthesise_and_publish(self, text: str):
        if not self._running or not text.strip():
            return

        if not self._track:
            self._audio_source = rtc.AudioSource(
                self._tts.sample_rate, self._tts.num_channels
            )
            self._track = rtc.LocalAudioTrack.create_audio_track(
                self.track_name, self._audio_source
            )
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_UNKNOWN, name=self.track_name
            )

            try:
                pub = await self.room.local_participant.publish_track(
                    self._track, options
                )
                logger.info(f"[TTSNode] Published translated track {self.track_name}")
            except Exception as e:
                logger.error(f"[TTSNode] Failed to publish track: {e}")
                return

        retries = 0
        while retries < 5 and self._running:
            try:
                logger.debug(f"[TTSNode] Synthesising: {text[:30]}...")
                async for event in self._tts.synthesize(text):
                    if event.type == tts.SynthesizeEventType.AUDIO:
                        await self._audio_source.capture_frame(event.frame)
                break
            except Exception as e:
                retries += 1
                logger.warning(f"[TTSNode] TTS synthesis error, retry {retries}/5: {e}")
                await asyncio.sleep(2**retries)

        if retries >= 5:
            logger.error(f"[TTSNode] TTS node failed after max retries.")

    async def stop(self):
        self._running = False
        if self._track:
            try:
                await self.room.local_participant.unpublish_track(self._track.sid)
            except Exception:
                pass
        logger.info(f"[TTSNode] Stopped TTS for {self.track_name}")
