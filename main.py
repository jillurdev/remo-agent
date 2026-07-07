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
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    from livekit.plugins import silero

    silero.VAD.load()

    def update_pipelines():
        active_langs = set(target_languages.values())
        for sp_sid, sp in speaker_pipelines.items():
            for lang in list(sp.language_pipelines.keys()):
                if lang not in active_langs:
                    asyncio.create_task(sp.remove_language(lang))
            for lang in active_langs:
                source_lang = sp._get_source_lang()
                if lang != source_lang and lang != "no-translate":
                    asyncio.create_task(sp.ensure_language(lang))

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")
        speaker_pipelines[participant.sid] = SpeakerPipeline(participant, ctx.room)

        lang = "no-translate"
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
                lang = meta.get("target_lang", "no-translate")
            except:
                pass
        target_languages[participant.sid] = lang
        update_pipelines()

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
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
                logger.info(f"Subscribed to audio track for {participant.identity}")
                asyncio.create_task(
                    speaker_pipelines[participant.sid].set_speaker_track(track)
                )

    @ctx.room.on("participant_metadata_changed")
    def on_participant_metadata_changed(
        participant: rtc.RemoteParticipant, old_metadata: str
    ):
        logger.info(f"Participant metadata changed for {participant.identity}")
        lang = "no-translate"
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
                lang = meta.get("target_lang", "no-translate")
            except:
                pass

        if target_languages.get(participant.sid) != lang:
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
        for sp in speaker_pipelines.values():
            await sp.shutdown()


async def request_fnc(req: JobRequest) -> None:
    logger.info(f"Accepting job for room {req.room.name}")
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        )
    )
