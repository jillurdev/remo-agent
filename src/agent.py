# ============================================================
#  agent.py  —  LiveKit Multi-User Transcriber + Translator
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
from livekit.plugins import deepgram, elevenlabs
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import av as pyav
import numpy as np

load_dotenv()

logger = logging.getLogger("transcriber")

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


# ================================================================
#  HEALTH CHECK SERVER
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

        self._audio_source = rtc.AudioSource(24000, 1)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant_identity}", self._audio_source
        )
        self._track_published = False
        self._lock = asyncio.Lock()

    def update_settings(self, target_lang: str, voice_id: str):
        self.target_lang = target_lang
        self.voice_id = voice_id

    
    async def speak(self, text: str):
        if not text.strip():
            return
        if self.target_lang == "no-translate":
            return

        try:
            translated_text = GoogleTranslator(
                source="auto", target=self.target_lang
            ).translate(text)
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
                    await self.room.local_participant.set_metadata(
                        json.dumps(
                            {
                                "is_agent": True,
                                "translated_user": self.participant_identity,
                            }
                        )
                    )
                    await self.room.local_participant.publish_track(self._track)
                    self._track_published = True

                tts = gTTS(text=translated_text, lang=self.target_lang[:2], slow=False)
                mp3_buffer = io.BytesIO()
                tts.write_to_fp(mp3_buffer)
                mp3_buffer.seek(0)

                container = pyav.open(mp3_buffer, format="mp3")
                resampler = pyav.AudioResampler(format="s16", layout="mono", rate=24000)

                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        pcm = resampled.to_ndarray().flatten()
                        audio_frame = rtc.AudioFrame(
                            data=pcm.astype(np.int16).tobytes(),
                            sample_rate=24000,
                            num_channels=1,
                            samples_per_channel=len(pcm),
                        )
                        await self._audio_source.capture_frame(audio_frame)

            except Exception as e:
                logger.error(f"TTS/publish error for {self.participant_identity}: {e}")

# ================================================================
#  SPEAKER TRANSCRIBER
# ================================================================


NOVA2_SUPPORTED = {
    "en",
    "es",
    "fr",
    "de",
    "hi",
    "pt",
    "zh",
    "ja",
    "ko",
    "it",
    "nl",
    "pl",
    "ru",
    "tr",
    "id",
    "vi",
    "uk",
    "sv",
    "no",
    "da",
    "fi",
    "cs",
    "ro",
    "bg",
    "sk",
    "hu",
    "el",
    "ms",
}


class SpeakerTranscriber(Agent):
    def __init__(
        self,
        *,
        participant_identity: str,
        room: rtc.Room,
        on_transcript,
        user_lang: str = "multi",
    ):
        self.participant_identity = participant_identity
        self.room = room
        self.on_transcript = on_transcript

        stt_lang = user_lang if user_lang in NOVA2_SUPPORTED else "multi"
        logger.info(f"🎤 STT LANG: {user_lang} -> {stt_lang}")

        self.stt_plugin = deepgram.STT(
            model="nova-2", language=stt_lang, smart_format=True
        )

        self.tts_plugin = elevenlabs.TTS(
            model="eleven_multilingual_v2",
            voice_id=DEFAULT_VOICE_ID,
            api_key=os.environ.get("ELEVEN_API_KEY")
            or os.environ.get("ELEVENLABS_API_KEY"),
        )

        super().__init__(
            instructions="not-needed", stt=self.stt_plugin, tts=self.tts_plugin
        )

    async def on_user_turn_completed(self, _, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content

        if not user_transcript.strip():
            raise StopResponse()

        logger.info(f"📝 TRANSCRIPT [{self.participant_identity}]: {user_transcript!r}")

        await self.on_transcript(self.participant_identity, user_transcript)

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
        self._user_lang: dict[str, str] = {}  # speaking language

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._tasks)
        await asyncio.gather(
            *[self._close_session(s) for s in self._speaker_sessions.values()]
        )

    def _parse_metadata(self, participant: rtc.RemoteParticipant):
        target_lang = "no-translate"
        voice_id = DEFAULT_VOICE_ID
        user_lang = "multi"  # default — auto detect

        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                target_lang = metadata.get("target_lang", "no-translate")
                voice_id = metadata.get("voice_id", DEFAULT_VOICE_ID)
                user_lang = metadata.get("user_lang", "multi")
            except Exception as e:
                logger.warning(f"Metadata parse error for {participant.identity}: {e}")

        return target_lang, voice_id, user_lang

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        target_lang, voice_id, user_lang = self._parse_metadata(participant)
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id
        self._user_lang[participant.identity] = user_lang

        if participant.identity in self._listeners:
            self._listeners[participant.identity].update_settings(target_lang, voice_id)
            logger.info(
                f"🔄 UPDATED {participant.identity}: target_lang={target_lang}, user_lang={user_lang}"
            )

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🟢 PARTICIPANT CONNECTED: {participant.identity}")

        if (
            participant.identity in self._speaker_sessions
            or participant.identity.startswith("agent-")
        ):
            logger.info(f"⛔ SKIPPED: {participant.identity}")
            return

        target_lang, voice_id, user_lang = self._parse_metadata(participant)
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id
        self._user_lang[participant.identity] = user_lang

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
                logger.info(f"✅ SESSION READY: {participant.identity}")
            except Exception as e:
                logger.error(f"Session failed for {participant.identity}: {e}")
            finally:
                self._tasks.discard(t)

        task.add_done_callback(on_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🔴 DISCONNECTED: {participant.identity}")
        self._user_target_lang.pop(participant.identity, None)
        self._user_voice.pop(participant.identity, None)
        self._user_lang.pop(participant.identity, None)
        self._listeners.pop(participant.identity, None)
        self._speaker_agents.pop(participant.identity, None)

        session = self._speaker_sessions.pop(participant.identity, None)
        if session is None:
            return

        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def handle_language_update(self, payload: bytes):
        try:
            data = json.loads(payload.decode("utf-8"))
            identity = data.get("identity")
            target_lang = data.get("target_lang", "no-translate")
            voice_id = data.get("voice_id", DEFAULT_VOICE_ID)

            logger.info(f"🌐 LANGUAGE UPDATE: {identity} -> {target_lang}")

            if identity in self._listeners:
                self._listeners[identity].update_settings(target_lang, voice_id)
                self._user_target_lang[identity] = target_lang
        except Exception as e:
            logger.error(f"Language update error: {e}")

    async def on_transcript(self, speaker_identity: str, transcript: str):
        logger.info(f"📩 TRANSCRIPT FROM: {speaker_identity} — {transcript!r}")

        raw_identity = speaker_identity
        clean_name = re.sub(r"_{2,}[a-zA-Z0-9]+$", "", raw_identity)

        payload = {
            "message": transcript,
            "timestamp": int(time.time() * 1000),
            "id": f"transcript-{raw_identity}-{time.time()}",
            "from": {"identity": raw_identity, "name": clean_name, "isLocal": False},
        }

        try:
            await self.ctx.room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="transcription_data",
            )
        except Exception as e:
            logger.error(f"Failed to publish transcript: {e}")

        listeners_to_speak = [
            listener
            for identity, listener in self._listeners.items()
            if identity != speaker_identity and listener.target_lang != "no-translate"
        ]

        logger.info(
            f"🔊 LISTENERS TO SPEAK: {[l.participant_identity for l in listeners_to_speak]}"
        )

        if not listeners_to_speak:
            return

        await asyncio.gather(
            *[listener.speak(transcript) for listener in listeners_to_speak],
            return_exceptions=True,
        )

    async def _start_speaker_session(self, participant: rtc.RemoteParticipant):
        logger.info(f"⚙️ STARTING SESSION: {participant.identity}")

        session = AgentSession()

        user_lang = self._user_lang.get(participant.identity, "multi")
        logger.info(f"🗣️ USER LANG for {participant.identity}: {user_lang}")

        agent = SpeakerTranscriber(
            participant_identity=participant.identity,
            room=self.ctx.room,
            on_transcript=self.on_transcript,
            user_lang=user_lang,
        )

        room_io = RoomIO(
            agent_session=session,
            room=self.ctx.room,
            participant=participant,
            input_options=RoomInputOptions(
                text_enabled=False,
            ),
            output_options=RoomOutputOptions(
                transcription_enabled=True, audio_enabled=False
            ),
        )

        await room_io.start()
        await session.start(agent=agent)

        logger.info(f"🚀 SESSION LIVE: {participant.identity}")

        return session, agent

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.aclose()


# ================================================================
#  ENTRYPOINT
# ================================================================


async def entrypoint(ctx: JobContext):
    logger.info("🚀 ENTRYPOINT STARTED")

    manager = MultiUserTranslationManager(ctx)

    def on_data_received(data_packet):
        try:
            topic = data_packet.topic
            payload = bytes(data_packet.data)
            if topic != "language_update":
                return
            asyncio.create_task(manager.handle_language_update(payload))
        except Exception as e:
            logger.error(f"data_received error: {e}")

    ctx.room.on("participant_connected", manager.on_participant_connected)
    ctx.room.on("participant_disconnected", manager.on_participant_disconnected)
    ctx.room.on("participant_metadata_changed", manager.on_metadata_changed)
    ctx.room.on("data_received", on_data_received)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("✅ CONNECTED TO LIVEKIT")

    for p in ctx.room.remote_participants.values():
        manager.on_participant_connected(p)

    while True:
        await asyncio.sleep(1)


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            load_threshold=0.9, 
        )
    )
