# ============================================================
#  agent.py  —  LiveKit Multi-User Transcriber + Translator
#  Per-participant translation: each user hears in their chosen language
#  AI Mode is personal — only users who enable it get translation
# ============================================================

import asyncio
import logging
import json
import time
import re
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RoomInputOptions,
    RoomIO,
    RoomOutputOptions,
    StopResponse,
    WorkerOptions,
    cli,
    llm,
    utils,
)
from livekit.plugins import deepgram, elevenlabs, noise_cancellation
from translate import Translator

load_dotenv()

logger = logging.getLogger("transcriber")

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


# ================================================================
#  HEALTH CHECK SERVER (required for Render)
# ================================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server running on port {port}")
    server.serve_forever()


# ================================================================
#  LISTENER AUDIO PUBLISHER
# ================================================================

class ListenerAudioPublisher:
    def __init__(self, participant_identity: str, room: rtc.Room, voice_id: str):
        self.participant_identity = participant_identity
        self.room = room
        self.target_lang = "no-translate"
        self.voice_id = voice_id

        self._tts = elevenlabs.TTS(
            model="eleven_multilingual_v2",
            voice_id=self.voice_id
        )
        self._audio_source = rtc.AudioSource(
            self._tts.sample_rate,
            self._tts.num_channels
        )
        self._track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant_identity}",
            self._audio_source
        )
        self._track_published = False
        self._lock = asyncio.Lock()

    def update_settings(self, target_lang: str, voice_id: str):
        self.target_lang = target_lang

        if voice_id != self.voice_id:
            self.voice_id = voice_id
            self._tts = elevenlabs.TTS(
                model="eleven_multilingual_v2",
                voice_id=self.voice_id
            )
            self._audio_source = rtc.AudioSource(
                self._tts.sample_rate,
                self._tts.num_channels
            )
            self._track = rtc.LocalAudioTrack.create_audio_track(
                f"translation_{self.participant_identity}",
                self._audio_source
            )
            self._track_published = False

    async def speak(self, text: str):
        logger.info(f"🔊 SPEECH REQUEST: {self.participant_identity}")
        if not text.strip():
            return

        if self.target_lang == "no-translate":
            logger.info(f"🚫 SKIP (no-translate): {self.participant_identity}")
            return

        try:
            translator = Translator(to_lang=self.target_lang)
            translated_text = translator.translate(text)
        except Exception as e:
            logger.error(f"Translation error for {self.participant_identity}: {e}")
            return

        logger.info(
            f"[for {self.participant_identity}] "
            f"(auto->{self.target_lang}) "
            f"{text!r} -> {translated_text!r}"
        )

        async with self._lock:
            try:
                if not self._track_published:
                    await self.room.local_participant.publish_track(self._track)
                    self._track_published = True

                async for synthesized in self._tts.synthesize(translated_text):
                    await self._audio_source.capture_frame(synthesized.frame)

            except Exception as e:
                logger.error(f"TTS/publish error for {self.participant_identity}: {e}")


# ================================================================
#  SPEAKER TRANSCRIBER
# ================================================================

class SpeakerTranscriber(Agent):
    def __init__(self, *, participant_identity: str, room: rtc.Room, on_transcript):
        self.participant_identity = participant_identity
        self.room = room
        self.on_transcript = on_transcript

        self.stt_plugin = deepgram.STT(
            model="nova-2",
            language="multi",
            smart_format=True
        )

        self.tts_plugin = elevenlabs.TTS(
            model="eleven_multilingual_v2",
            voice_id=DEFAULT_VOICE_ID
        )

        super().__init__(
            instructions="not-needed",
            stt=self.stt_plugin,
            tts=self.tts_plugin
        )

    async def on_user_turn_completed(self, _, new_message: llm.ChatMessage):
        logger.info("🔥 STT TURN COMPLETED TRIGGERED")
        user_transcript = new_message.text_content

        logger.info(f"🎙️ RAW TRANSCRIPTION EVENT TRIGGERED")

        if not user_transcript.strip():
            logger.info("⚠️ EMPTY TRANSCRIPT")
            raise StopResponse()

        logger.info(
        f"📝 FINAL TRANSCRIPT [{self.participant_identity}]: {user_transcript}"
    )

        await self.on_transcript(self.participant_identity, user_transcript)

        logger.info("📤 TRANSCRIPT SENT TO MANAGER")

        raise StopResponse()


# ================================================================
#  MULTI-USER TRANSLATION MANAGER
# ================================================================

class MultiUserTranslationManager:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._speaker_sessions: dict[str, AgentSession] = {}
        self._speaker_agents: dict[str, SpeakerTranscriber] = {}
        self._listeners: dict[str, ListenerAudioPublisher] = {}
        self._tasks: set[asyncio.Task] = set()
        self._user_target_lang: dict[str, str] = {}
        self._user_voice: dict[str, str] = {}

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.on("participant_metadata_changed", self.on_metadata_changed)
        

    async def aclose(self):
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.off("participant_metadata_changed", self.on_metadata_changed)
        await utils.aio.cancel_and_wait(*self._tasks)
        await asyncio.gather(
            *[self._close_session(s) for s in self._speaker_sessions.values()]
        )

    def _parse_metadata(self, participant: rtc.RemoteParticipant):
        target_lang = "no-translate"
        voice_id = DEFAULT_VOICE_ID

        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                target_lang = metadata.get("target_lang", "no-translate")
                voice_id = metadata.get("voice_id", DEFAULT_VOICE_ID)
            except Exception as e:
                logger.warning(f"Metadata parse error for {participant.identity}: {e}")
                logger.info(
    f"🔄 METADATA UPDATED: {participant.identity} -> {participant.metadata}"
)

        return target_lang, voice_id

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        target_lang, voice_id = self._parse_metadata(participant)
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id

        if participant.identity in self._listeners:
            self._listeners[participant.identity].update_settings(target_lang, voice_id)
            logger.info(f"Updated {participant.identity}: target_lang={target_lang}")

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🟢 PARTICIPANT CONNECTED: {participant.identity}")
        logger.info(f"🔔 EVENT HIT: {participant.identity}")

        logger.info(f"📦 METADATA: {participant.metadata}")
        if (
            participant.identity in self._speaker_sessions
            or participant.identity.startswith("agent-")
        ):
            logger.info(f"⛔ SKIPPED PARTICIPANT: {participant.identity}")
            return
        
        logger.info(f"🎯 STARTING SPEAKER SESSION FOR: {participant.identity}")

        target_lang, voice_id = self._parse_metadata(participant)
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id

        listener = ListenerAudioPublisher(participant.identity, self.ctx.room, voice_id)
        listener.update_settings(target_lang, voice_id)
        self._listeners[participant.identity] = listener

        task = asyncio.create_task(self._start_speaker_session(participant))
        self._tasks.add(task)

        def on_done(t: asyncio.Task):
            try:
                session, agent = t.result()
                self._speaker_sessions[participant.identity] = session
                self._speaker_agents[participant.identity] = agent
            except Exception as e:
                logger.error(f"Session failed for {participant.identity}: {e}")
            finally:
                self._tasks.discard(t)

        task.add_done_callback(on_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🔴 PARTICIPANT DISCONNECTED: {participant.identity}")
        self._user_target_lang.pop(participant.identity, None)
        self._user_voice.pop(participant.identity, None)
        self._listeners.pop(participant.identity, None)
        self._speaker_agents.pop(participant.identity, None)

        session = self._speaker_sessions.pop(participant.identity, None)
        if session is None:
            return

        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def on_transcript(self, speaker_identity: str, transcript: str):
        logger.info(f"📩 MANAGER RECEIVED TRANSCRIPT FROM: {speaker_identity}")
        logger.info(f"💬 TEXT: {transcript}")
        raw_identity = speaker_identity
        clean_name = re.sub(r'_{2,}[a-zA-Z0-9]+$', '', raw_identity)

        payload = {
            "message": transcript,
            "timestamp": int(time.time() * 1000),
            "id": f"transcript-{raw_identity}-{time.time()}",
            "from": {
                "identity": raw_identity,
                "name": clean_name,
                "isLocal": False
            }
        }

        try:
            await self.ctx.room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="transcription_data"
            )
        except Exception as e:
            logger.error(f"Failed to publish transcript: {e}")

        # শুধু যাদের agent mode ON (target_lang != no-translate) তারা translation পাবে
        listeners_to_speak = [
            listener
            for identity, listener in self._listeners.items()
            if identity != speaker_identity
            and listener.target_lang != "no-translate"
        ]

        if not listeners_to_speak:
            return

        await asyncio.gather(
            *[listener.speak(transcript) for listener in listeners_to_speak],
            return_exceptions=True
        )

    async def _start_speaker_session(self, participant: rtc.RemoteParticipant):
        logger.info(f"⚙️ SESSION CREATION START: {participant.identity}")
        
        session = AgentSession()

        agent = SpeakerTranscriber(
            participant_identity=participant.identity,
            room=self.ctx.room,
            on_transcript=self.on_transcript,
        )

        logger.info(f"🧠 AGENT CREATED: {participant.identity}")

        room_io = RoomIO(
            agent_session=session,
            room=self.ctx.room,
            participant=participant,
            input_options=RoomInputOptions(
                text_enabled=False,
                noise_cancellation=noise_cancellation.BVC()
            ),
            output_options=RoomOutputOptions(
                transcription_enabled=True,
                audio_enabled=False
            ),
        )

        logger.info(f"🔗 ROOM IO STARTING: {participant.identity}")

        await room_io.start()

        logger.info(f"🎧 ROOM IO STARTED: {participant.identity}")

        await session.start(agent=agent)

        logger.info(f"🚀 SESSION STARTED: {participant.identity}")

        return session, agent

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.aclose()


# ================================================================
#  ENTRYPOINT
# ================================================================

async def entrypoint(ctx: JobContext):
    logger.info("🚀 ENTRYPOINT STARTED")

    manager = MultiUserTranslationManager(ctx)
    manager.start()

    def on_join(participant):
        logger.info(f"🔥 JOIN EVENT: {participant.identity}")
        manager.on_participant_connected(participant)

    ctx.room.on("participant_connected", on_join)
    ctx.room.on("participant_disconnected", manager.on_participant_disconnected)
    ctx.room.on("participant_metadata_changed", manager.on_metadata_changed)

    await ctx.connect(auto_subscribe=AutoSubscribe.ALL)

    logger.info("✅ CONNECTED TO LIVEKIT")

    while True:
        await asyncio.sleep(1)


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))