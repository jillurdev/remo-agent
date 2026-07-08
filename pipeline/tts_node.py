import asyncio
from livekit import rtc
from livekit.plugins import cartesia
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
        # Serializes access to _audio_source.capture_frame(). Without this,
        # overlapping synthesise_and_publish() calls (e.g. triggered by STT
        # interim transcripts firing faster than TTS can finish) race on the
        # same audio source and LiveKit raises InvalidState - failed to
        # capture frame, and previously in-flight partial audio gets mixed
        # with new audio -> repeated / garbled speech.
        self._synth_lock = asyncio.Lock()
        # Monotonically increasing token to let a *newer* synthesis request
        # cancel/ignore results from an older, still-running one (relevant if
        # you synthesize per-interim-transcript instead of per-final).
        self._current_gen = 0

    async def start(self):
        self._running = True
        logger.info(f"[TTSNode] Started TTS for {self.track_name} ({self.target_lang})")

    async def synthesise_and_publish(self, text: str):
        if not self._running or not text.strip():
            return

        # Bump generation so any older queued/looping synthesis for this
        # speaker knows it's stale and should stop early.
        self._current_gen += 1
        my_gen = self._current_gen

        if not self._track:
            self._audio_source = rtc.AudioSource(
                self._tts.sample_rate, self._tts.num_channels
            )
            self._track = rtc.LocalAudioTrack.create_audio_track(
                self.track_name, self._audio_source
            )
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_UNKNOWN,
            )

            try:
                pub = await self.room.local_participant.publish_track(
                    self._track, options
                )
                logger.info(f"[TTSNode] Published translated track {self.track_name}")
            except Exception as e:
                logger.error(f"[TTSNode] Failed to publish track: {e}")
                return

        # Serialize: only one synthesis/capture sequence writes to
        # _audio_source at a time. If a newer call is already waiting on the
        # lock, this one will find my_gen stale after acquiring and bail out
        # instead of writing overlapping/outdated audio.
        async with self._synth_lock:
            if my_gen != self._current_gen:
                logger.debug(
                    f"[TTSNode] Skipping stale synthesis for {self.track_name} "
                    f"(gen {my_gen} superseded by {self._current_gen})"
                )
                return

            retries = 0
            while retries < 5 and self._running:
                try:
                    logger.debug(f"[TTSNode] Synthesising: {text[:30]}...")
                    async for synthesized_audio in self._tts.synthesize(text):
                        if my_gen != self._current_gen:
                            # A newer request superseded us mid-stream; stop
                            # writing frames instead of finishing this one.
                            logger.debug(
                                f"[TTSNode] Aborting mid-stream synthesis for "
                                f"{self.track_name} (superseded)"
                            )
                            return
                        if synthesized_audio.frame is not None:
                            await self._audio_source.capture_frame(
                                synthesized_audio.frame
                            )
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"[TTSNode] TTS synthesis error, retry {retries}/5: {e}"
                    )
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
