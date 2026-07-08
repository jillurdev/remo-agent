
import re
from livekit import rtc
from utils.logger import logger
from .mt_node import MTNode
from .tts_node import TTSNode


class LanguagePipeline:
    """
    One of these exists per active target language for a given speaker.
    It no longer runs its own STT or publishes transcripts — SpeakerPipeline
    owns the single shared STT stream and calls handle_segment() here with
    already-transcribed text. This pipeline's only job is: translate that
    text into target_lang, then speak it via TTS.
    """

    def __init__(
        self,
        speaker: rtc.RemoteParticipant,
        source_lang: str,
        target_lang: str,
        room: rtc.Room,
    ):
        self.speaker = speaker
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.room = room

        self.mt_node = MTNode(source_lang, target_lang)

        # Track name e.g. trl_alice_en
        clean_id = re.sub(r"[^a-zA-Z0-9]", "_", speaker.identity)
        track_name = f"trl_{clean_id}_{target_lang}"
        self.tts_node = TTSNode(target_lang, room, track_name, speaker.identity)

        self._running = False

    async def start(self):
        self._running = True
        await self.tts_node.start()
        logger.info(
            f"[LanguagePipeline] Started {self.source_lang} -> {self.target_lang} for {self.speaker.identity}"
        )

    async def handle_segment(self, text: str, is_final: bool):
        """Called by SpeakerPipeline with a segment from the shared STT stream."""
        if not self._running or not text.strip():
            return

        try:
            translated = await self.mt_node.translate(text)
            if translated and translated.strip():
                await self.tts_node.synthesise_and_publish(translated)
        except Exception as e:
            logger.error(f"[LanguagePipeline] Pipeline error: {e}")

    async def stop(self):
        self._running = False
        await self.tts_node.stop()
        logger.info(
            f"[LanguagePipeline] Stopped {self.target_lang} for {self.speaker.identity}"
        )
 