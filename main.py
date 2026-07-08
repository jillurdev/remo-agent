
import asyncio
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobRequest, WorkerOptions, cli
from pipeline.speaker_pipeline import SpeakerPipeline
from utils.logger import logger

# Store SpeakerPipelines by participant SID
speaker_pipelines = {}
# Keep track of requested target languages
target_languages = {}  # participant_sid -> lang_code


async def entrypoint(ctx: JobContext):
    logger.info(f"🤖 AI agent joining room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"✅ AI agent connected to room: {ctx.room.name}")

    from livekit.plugins import silero

    silero.VAD.load()
    logger.info("🔊 VAD (voice activity detection) model loaded")

    def _lang_from_metadata(metadata: str | None, field: str, default: str) -> str:
        if not metadata:
            return default
        try:
            return json.loads(metadata).get(field, default)
        except Exception:
            return default

    def update_pipelines():
        active_langs = set(target_languages.values())
        for sp_sid, sp in speaker_pipelines.items():
            for lang in list(sp.language_pipelines.keys()):
                if lang not in active_langs:
                    logger.info(
                        f"🌐 No more listeners for '{lang}' from {sp.speaker.identity} — stopping that translation pipeline"
                    )
                    asyncio.create_task(sp.remove_language(lang))
            for lang in active_langs:
                source_lang = sp._get_source_lang()
                if (
                    lang != source_lang
                    and lang != "no-translate"
                    and lang not in sp.language_pipelines
                ):
                    logger.info(
                        f"🌐 Starting translation pipeline for {sp.speaker.identity}: {source_lang} -> {lang}"
                    )
                    asyncio.create_task(sp.ensure_language(lang))

    def on_participant_connected(participant: rtc.RemoteParticipant):
        # Guard against duplicate SpeakerPipeline creation. This can happen
        # because we call this function once up-front for every participant
        # already in the room, AND it's also registered as the
        # "participant_connected" event handler — if a participant connects
        # right around agent startup, both paths can fire for the same sid.
        # Without this guard, the old SpeakerPipeline (with its own live
        # STTNode) never gets stopped, so the room ends up with two
        # independent STT streams transcribing the same audio -> every final
        # segment gets published twice (duplicate transcript bubbles) and
        # translated/spoken twice.
        if participant.sid in speaker_pipelines:
            logger.debug(
                f"[main] on_participant_connected called again for already-tracked "
                f"participant {participant.identity} ({participant.sid}) — ignoring"
            )
            return

        lang = _lang_from_metadata(participant.metadata, "target_lang", "no-translate")
        spoken_lang = _lang_from_metadata(participant.metadata, "user_lang", "en")
        logger.info(
            f"👤 Participant joined: {participant.identity}  "
            f"(speaks: {spoken_lang}, wants to hear: {lang})"
        )
        speaker_pipelines[participant.sid] = SpeakerPipeline(participant, ctx.room)
        target_languages[participant.sid] = lang
        update_pipelines()

    ctx.room.on("participant_connected", on_participant_connected)

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"👋 Participant left: {participant.identity}")
        if participant.sid in speaker_pipelines:
            sp = speaker_pipelines.pop(participant.sid)
            asyncio.create_task(sp.shutdown())

        if participant.sid in target_languages:
            del target_languages[participant.sid]
            update_pipelines()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            if participant.sid in speaker_pipelines:
                logger.info(f"🎙️  Listening to {participant.identity}'s microphone")
                asyncio.create_task(
                    speaker_pipelines[participant.sid].set_speaker_track(track)
                )

    @ctx.room.on("participant_metadata_changed")
    def on_participant_metadata_changed(
        participant: rtc.RemoteParticipant, old_metadata: str
    ):
        lang = _lang_from_metadata(participant.metadata, "target_lang", "no-translate")

        if target_languages.get(participant.sid) != lang:
            logger.info(
                f"🔄 {participant.identity} changed their listening language: "
                f"{target_languages.get(participant.sid)} -> {lang}"
            )
            target_languages[participant.sid] = lang
            update_pipelines()

    for participant in ctx.room.remote_participants.values():
        on_participant_connected(participant)
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                on_track_subscribed(pub.track, pub, participant)

    try:
        await asyncio.Event().wait()
    finally:
        logger.info(f"🛑 Agent shutting down for room: {ctx.room.name}")
        for sp in speaker_pipelines.values():
            await sp.shutdown()


async def request_fnc(req: JobRequest) -> None:
    logger.info(f"📥 Job request received for room: {req.room.name}")
    await req.accept()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        )
    )
    
# import asyncio
# import os
# import json
# import logging
# from dotenv import load_dotenv

# load_dotenv()
# from livekit import rtc
# from livekit.agents import AutoSubscribe, JobContext, JobRequest, WorkerOptions, cli
# from pipeline.speaker_pipeline import SpeakerPipeline
# from utils.logger import logger

# # Store SpeakerPipelines by participant SID
# speaker_pipelines = {}
# # Keep track of requested target languages
# target_languages = {}  # participant_sid -> lang_code


# async def entrypoint(ctx: JobContext):
#     logger.info(f"🤖 AI agent joining room: {ctx.room.name}")
#     await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#     logger.info(f"✅ AI agent connected to room: {ctx.room.name}")

#     from livekit.plugins import silero

#     silero.VAD.load()
#     logger.info("🔊 VAD (voice activity detection) model loaded")

#     def _lang_from_metadata(metadata: str | None, field: str, default: str) -> str:
#         if not metadata:
#             return default
#         try:
#             return json.loads(metadata).get(field, default)
#         except Exception:
#             return default

#     def update_pipelines():
#         active_langs = set(target_languages.values())
#         for sp_sid, sp in speaker_pipelines.items():
#             for lang in list(sp.language_pipelines.keys()):
#                 if lang not in active_langs:
#                     logger.info(
#                         f"🌐 No more listeners for '{lang}' from {sp.speaker.identity} — stopping that translation pipeline"
#                     )
#                     asyncio.create_task(sp.remove_language(lang))
#             for lang in active_langs:
#                 source_lang = sp._get_source_lang()
#                 if (
#                     lang != source_lang
#                     and lang != "no-translate"
#                     and lang not in sp.language_pipelines
#                 ):
#                     logger.info(
#                         f"🌐 Starting translation pipeline for {sp.speaker.identity}: {source_lang} -> {lang}"
#                     )
#                     asyncio.create_task(sp.ensure_language(lang))

#     @ctx.room.on("participant_connected")
#     def on_participant_connected(participant: rtc.RemoteParticipant):
#         lang = _lang_from_metadata(participant.metadata, "target_lang", "no-translate")
#         spoken_lang = _lang_from_metadata(participant.metadata, "user_lang", "en")
#         logger.info(
#             f"👤 Participant joined: {participant.identity}  "
#             f"(speaks: {spoken_lang}, wants to hear: {lang})"
#         )
#         speaker_pipelines[participant.sid] = SpeakerPipeline(participant, ctx.room)
#         target_languages[participant.sid] = lang
#         update_pipelines()

#     @ctx.room.on("participant_disconnected")
#     def on_participant_disconnected(participant: rtc.RemoteParticipant):
#         logger.info(f"👋 Participant left: {participant.identity}")
#         if participant.sid in speaker_pipelines:
#             sp = speaker_pipelines.pop(participant.sid)
#             asyncio.create_task(sp.shutdown())

#         if participant.sid in target_languages:
#             del target_languages[participant.sid]
#             update_pipelines()

#     @ctx.room.on("track_subscribed")
#     def on_track_subscribed(
#         track: rtc.Track,
#         publication: rtc.RemoteTrackPublication,
#         participant: rtc.RemoteParticipant,
#     ):
#         if track.kind == rtc.TrackKind.KIND_AUDIO:
#             if participant.sid in speaker_pipelines:
#                 logger.info(f"🎙️  Listening to {participant.identity}'s microphone")
#                 asyncio.create_task(
#                     speaker_pipelines[participant.sid].set_speaker_track(track)
#                 )

#     @ctx.room.on("participant_metadata_changed")
#     def on_participant_metadata_changed(
#         participant: rtc.RemoteParticipant, old_metadata: str
#     ):
#         lang = _lang_from_metadata(participant.metadata, "target_lang", "no-translate")

#         if target_languages.get(participant.sid) != lang:
#             logger.info(
#                 f"🔄 {participant.identity} changed their listening language: "
#                 f"{target_languages.get(participant.sid)} -> {lang}"
#             )
#             target_languages[participant.sid] = lang
#             update_pipelines()

#     for participant in ctx.room.remote_participants.values():
#         on_participant_connected(participant)
#         for pub in participant.track_publications.values():
#             if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
#                 on_track_subscribed(pub.track, pub, participant)

#     try:
#         await asyncio.Event().wait()
#     finally:
#         logger.info(f"🛑 Agent shutting down for room: {ctx.room.name}")
#         for sp in speaker_pipelines.values():
#             await sp.shutdown()


# async def request_fnc(req: JobRequest) -> None:
#     logger.info(f"📥 Job request received for room: {req.room.name}")
#     await req.accept()


# if __name__ == "__main__":
#     cli.run_app(
#         WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             request_fnc=request_fnc,
#         )
#     )
