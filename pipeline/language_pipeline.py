
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


# import asyncio
# import json
# import time
# import re
# from livekit import rtc
# from utils.logger import logger
# from .stt_node import STTNode
# from .mt_node import MTNode
# from .tts_node import TTSNode


# class LanguagePipeline:
#     def __init__(
#         self,
#         speaker: rtc.RemoteParticipant,
#         speaker_track: rtc.AudioTrack,
#         source_lang: str,
#         target_lang: str,
#         room: rtc.Room,
#     ):
#         self.speaker = speaker
#         self.speaker_track = speaker_track
#         self.source_lang = source_lang
#         self.target_lang = target_lang
#         self.room = room

#         self.stt_node = STTNode(speaker_track, source_lang)
#         self.mt_node = MTNode(source_lang, target_lang)

#         # Track name e.g. trl_alice_en
#         clean_id = re.sub(r"[^a-zA-Z0-9]", "_", speaker.identity)
#         track_name = f"trl_{clean_id}_{target_lang}"
#         self.tts_node = TTSNode(target_lang, room, track_name, speaker.identity)

#         self._last_sent_segment = ""
#         self._last_final_segment = None
#         self._running = False

#     async def start(self):
#         self._running = True
#         await self.tts_node.start()
#         await self.stt_node.start(self.on_segment)
#         logger.info(
#             f"[LanguagePipeline] Started {self.source_lang} -> {self.target_lang} for {self.speaker.identity}"
#         )

#     async def on_segment(self, text: str, is_final: bool):
#         if not self._running or not text.strip():
#             return

#         if is_final:
#             # Always let a final segment through once, even if it matches the last interim.
#             if text == self._last_final_segment:
#                 return
#             self._last_final_segment = text
#         else:
#             if text == self._last_sent_segment:
#                 return
#             self._last_sent_segment = text

#         try:
#             translated = await self.mt_node.translate(text)
#             if translated and translated.strip():
#                 asyncio.create_task(self.tts_node.synthesise_and_publish(translated))
#                 await self._publish_transcript(text, translated, is_final)
#         except Exception as e:
#             logger.error(f"[LanguagePipeline] Pipeline error: {e}")

#     async def _publish_transcript(self, original: str, translated: str, is_final: bool):
#         payload = {
#             "type": "transcript",
#             "from": {"identity": self.speaker.identity},
#             "speakerId": self.speaker.identity,
#             "message": translated,
#             "original": original,
#             "isFinal": is_final,
#             "lang": self.target_lang,
#             "timestamp": int(time.time() * 1000),
#         }
#         try:
#             await self.room.local_participant.publish_data(
#                 payload=json.dumps(payload).encode("utf-8"),
#                 reliable=True,
#                 topic="transcription_data",
#             )
#         except Exception as e:
#             logger.error(f"[LanguagePipeline] Failed to publish transcript: {e}")

#     async def stop(self):
#         self._running = False
#         await self.stt_node.stop()
#         await self.tts_node.stop()
#         logger.info(
#             f"[LanguagePipeline] Stopped {self.target_lang} for {self.speaker.identity}"
#         )
