import asyncio
import json
import time
import uuid
from livekit import rtc
from utils.logger import logger
from .language_pipeline import LanguagePipeline
from .stt_node import STTNode


class SpeakerPipeline:
    """
    Owns exactly ONE STT stream per speaker, no matter how many target
    languages are active in the room. Segments from that single STT stream
    are fanned out to every active LanguagePipeline (MT + TTS only) and the
    transcript is published exactly once, in the speaker's own language.
    """

    # If two FINAL segments arrive within this many seconds, we treat them
    # as the same continuous sentence/turn (Deepgram often finalizes on a
    # brief pause mid-sentence, not just at the true end of an utterance).
    # Keeping the same turn id + growing the text lets the frontend replace
    # the previous bubble instead of adding a new one — fixing the
    # "sentence shows up broken into 2-3 messages" issue.
    _TURN_GAP_SECONDS = 2.0

    def __init__(self, speaker: rtc.RemoteParticipant, room: rtc.Room):
        self.speaker = speaker
        self.room = room
        self.language_pipelines = {}  # target_lang -> LanguagePipeline
        self.speaker_track = None
        self.stt_node = None

        self._last_sent_segment = ""
        self._last_final_segment = None

        # Turn-accumulation state
        self._current_turn_id = None
        self._current_turn_text = ""
        self._last_final_at = 0.0

    async def set_speaker_track(self, track: rtc.AudioTrack):
        if self.speaker_track == track:
            return
        self.speaker_track = track
        logger.info(f"[SpeakerPipeline] Speaker track set for {self.speaker.identity}")

        # Start the single shared STT stream for this speaker the first time
        # we get an audio track. Adding/removing target languages later never
        # spins up another one.
        if self.stt_node is None:
            self.stt_node = STTNode(track, self._get_source_lang())
            await self.stt_node.start(self.on_segment)

    def _get_source_lang(self) -> str:
        if self.speaker.metadata:
            try:
                metadata = json.loads(self.speaker.metadata)
                return metadata.get("user_lang", "en")
            except:
                pass
        return "en"

    async def on_segment(self, text: str, is_final: bool):
        if not text.strip():
            return

        if is_final:
            # Always let a final segment through once, even if it matches the last interim.
            if text == self._last_final_segment:
                return
            self._last_final_segment = text
        else:
            if text == self._last_sent_segment:
                return
            self._last_sent_segment = text

        # Decide whether this final continues the current turn or starts a
        # new one, and publish the accumulated turn text (not just this
        # fragment) so the frontend can overwrite a single growing bubble.
        if is_final:
            now = time.time()
            starts_new_turn = (
                self._current_turn_id is None
                or (now - self._last_final_at) > self._TURN_GAP_SECONDS
            )
            if starts_new_turn:
                self._current_turn_id = str(uuid.uuid4())
                self._current_turn_text = text
            else:
                self._current_turn_text = f"{self._current_turn_text} {text}".strip()
            self._last_final_at = now

            await self._publish_transcript(
                self._current_turn_text, is_final, self._current_turn_id
            )
        else:
            # Interim path kept for safety in case STTNode ever emits one
            # again; not expected to fire since STTNode only forwards finals.
            await self._publish_transcript(text, is_final, self._current_turn_id)

        # Only fan FINAL segments out to translation + TTS. Forwarding
        # interim segments here caused each partial ("ami", "ami jacchi",
        # "ami jacchi school", ...) to be independently translated and
        # spoken — producing repeated, broken-sentence audio and racing
        # writes into TTSNode's audio source (InvalidState errors).
        if is_final:
            for pl in list(self.language_pipelines.values()):
                if pl:
                    asyncio.create_task(pl.handle_segment(text, is_final))

    async def _publish_transcript(self, text: str, is_final: bool, turn_id: str):
        payload = {
            "type": "transcript",
            "id": turn_id,
            "from": {"identity": self.speaker.identity},
            "speakerId": self.speaker.identity,
            "message": text,
            "original": text,
            "isFinal": is_final,
            "lang": self._get_source_lang(),
            "timestamp": int(time.time() * 1000),
        }
        try:
            await self.room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="transcription_data",
            )
        except Exception as e:
            logger.error(f"[SpeakerPipeline] Failed to publish transcript: {e}")

    async def ensure_language(self, lang_code: str) -> None:
        if lang_code in self.language_pipelines:
            return

        logger.info(
            f"[SpeakerPipeline] Ensuring language {lang_code} for {self.speaker.identity}"
        )
        source_lang = self._get_source_lang()

        # LanguagePipeline no longer needs the audio track directly — it only
        # does MT + TTS, fed by the segments this SpeakerPipeline forwards.
        pl = LanguagePipeline(self.speaker, source_lang, lang_code, self.room)
        self.language_pipelines[lang_code] = pl
        await pl.start()

    async def remove_language(self, lang_code: str) -> None:
        if lang_code not in self.language_pipelines:
            return

        pl = self.language_pipelines.pop(lang_code)
        if pl:
            await pl.stop()
        logger.info(
            f"[SpeakerPipeline] Removed language {lang_code} for {self.speaker.identity}"
        )

    async def shutdown(self) -> None:
        if self.stt_node:
            await self.stt_node.stop()
        for lang, pl in self.language_pipelines.items():
            if pl:
                await pl.stop()
        self.language_pipelines.clear()
        logger.info(f"[SpeakerPipeline] Shutdown complete for {self.speaker.identity}")
