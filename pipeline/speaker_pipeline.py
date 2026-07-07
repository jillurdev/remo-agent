import asyncio
import json
import time
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

    def __init__(self, speaker: rtc.RemoteParticipant, room: rtc.Room):
        self.speaker = speaker
        self.room = room
        self.language_pipelines = {}  # target_lang -> LanguagePipeline
        self.speaker_track = None
        self.stt_node = None

        self._last_sent_segment = ""
        self._last_final_segment = None

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

        # Publish the transcript exactly once per segment, in the speaker's own
        # language. Each viewer's client is responsible for translating this
        # into whichever "Transcript Language" they've personally selected.
        await self._publish_transcript(text, is_final)

        # Fan the same segment out to every active target-language pipeline
        # for translation + TTS. STT itself is never repeated per language.
        for pl in list(self.language_pipelines.values()):
            if pl:
                asyncio.create_task(pl.handle_segment(text, is_final))

    async def _publish_transcript(self, text: str, is_final: bool):
        payload = {
            "type": "transcript",
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



# import asyncio
# import json
# from livekit import rtc
# from utils.logger import logger
# from .language_pipeline import LanguagePipeline

# class SpeakerPipeline:
#     def __init__(self, speaker: rtc.RemoteParticipant, room: rtc.Room):
#         self.speaker = speaker
#         self.room = room
#         self.language_pipelines = {} # target_lang -> LanguagePipeline
#         self.speaker_track = None
        
#     async def set_speaker_track(self, track: rtc.AudioTrack):
#         if self.speaker_track == track:
#             return
#         self.speaker_track = track
#         # If track changes, we should restart pipelines, but normally it doesn't change.
#         # Wait, if we had pipelines waiting for track, we can start them now.
#         logger.info(f"[SpeakerPipeline] Speaker track set for {self.speaker.identity}")
#         for lang, pl in self.language_pipelines.items():
#             if not pl:
#                 # Re-create pipeline with track
#                 new_pl = LanguagePipeline(self.speaker, self.speaker_track, self._get_source_lang(), lang, self.room)
#                 self.language_pipelines[lang] = new_pl
#                 await new_pl.start()

#     def _get_source_lang(self) -> str:
#         if self.speaker.metadata:
#             try:
#                 metadata = json.loads(self.speaker.metadata)
#                 return metadata.get("user_lang", "en")
#             except:
#                 pass
#         return "en"

#     async def ensure_language(self, lang_code: str) -> None:
#         if lang_code in self.language_pipelines:
#             return
            
#         logger.info(f"[SpeakerPipeline] Ensuring language {lang_code} for {self.speaker.identity}")
#         source_lang = self._get_source_lang()
        
#         if not self.speaker_track:
#             # Store a placeholder until track is available
#             logger.warning(f"[SpeakerPipeline] No audio track for {self.speaker.identity} yet, queueing {lang_code}")
#             self.language_pipelines[lang_code] = None
#             return
            
#         pl = LanguagePipeline(self.speaker, self.speaker_track, source_lang, lang_code, self.room)
#         self.language_pipelines[lang_code] = pl
#         await pl.start()

#     async def remove_language(self, lang_code: str) -> None:
#         if lang_code not in self.language_pipelines:
#             return
            
#         pl = self.language_pipelines.pop(lang_code)
#         if pl:
#             await pl.stop()
#         logger.info(f"[SpeakerPipeline] Removed language {lang_code} for {self.speaker.identity}")

#     async def shutdown(self) -> None:
#         for lang, pl in self.language_pipelines.items():
#             if pl:
#                 await pl.stop()
#         self.language_pipelines.clear()
#         logger.info(f"[SpeakerPipeline] Shutdown complete for {self.speaker.identity}")
