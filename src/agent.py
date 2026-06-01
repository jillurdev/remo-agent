# ============================================================
#  agent.py  —  LiveKit Multi-User Transcriber + Translator
#  Render-ready version (free Web Service deployment)
# ============================================================

import asyncio
import logging
import json
import time
import re
import os                          # needed to read PORT env variable on Render
import threading                   # needed to run health-check server in background
from http.server import HTTPServer, BaseHTTPRequestHandler  # built-in HTTP server for Render health check
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

# Load environment variables from .env file (works locally)
# On Render, env vars are set from the dashboard — no .env file needed
load_dotenv()

logger = logging.getLogger("transcriber")

# Default ElevenLabs voice ID — can be overridden per-user via metadata
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


# ================================================================
#  HEALTH CHECK SERVER
#  Render requires a running HTTP server on the assigned PORT.
#  Without this, Render marks the service as "failed" and
#  continuously restarts it. This tiny server just responds "ok".
# ================================================================

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — responds 200 OK to any GET request."""

    def do_GET(self):
        # Render only checks for a 200 status, response body doesn't matter
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        # Suppress default HTTP access logs to avoid terminal spam on every health-check ping
        pass


def run_health_server():
    """
    Start the health check HTTP server.
    Render assigns a PORT via environment variable.
    Default fallback is 8000 for local testing.
    """
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server running on port {port}")
    server.serve_forever()


# ================================================================
#  TRANSCRIBER AGENT
#  One Transcriber instance is created per remote participant.
#  It handles STT (speech recognition), translation, and TTS
#  (text-to-speech) for that specific user.
# ================================================================

class Transcriber(Agent):
    def __init__(
        self,
        *,
        participant_identity: str,
        translator: Translator,
        room: rtc.Room,
        from_lang: str,
        target_lang: str,
        voice_id: str,
    ):
        self.participant_identity = participant_identity
        self.translator = translator
        self.room = room
        self.from_lang = from_lang       # language the user speaks
        self.target_lang = target_lang   # language to translate into
        self.voice_id = voice_id         # ElevenLabs voice ID for TTS output

        # --- STT (Speech-to-Text) setup using Deepgram ---
        self.stt_plugin = deepgram.STT(
            model="nova-2",
            language=self.from_lang,
            smart_format=True  # auto punctuation & formatting
        )

        # --- TTS (Text-to-Speech) setup using ElevenLabs ---
        self.tts_plugin = elevenlabs.TTS(
            model="eleven_multilingual_v2",  # supports many languages including Bengali
            voice_id=self.voice_id
        )

        # Initialize Agent base class — no LLM needed, just STT + TTS pipeline
        super().__init__(
            instructions="not-needed",  # no LLM, so instructions are irrelevant
            stt=self.stt_plugin,
            tts=self.tts_plugin
        )

        # Audio source and track for publishing translated audio back to the room
        self.audio_source = rtc.AudioSource(
            self.tts_plugin.sample_rate,
            self.tts_plugin.num_channels
        )

        self.track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant_identity}",
            self.audio_source
        )

        # Lazy publish flag — only publish the audio track on first actual translation
        self._track_published = False

    async def update_agent_metadata(self):
        """
        Publish agent metadata to the LiveKit room so the frontend
        knows which agent is handling which participant's translation.
        """
        logger.info(
            f"Agent metadata updated for {self.participant_identity}: "
            f"{self.from_lang} -> {self.target_lang}"
        )

        agent_metadata = {
            "is_agent": True,
            "input_lang": self.from_lang,
            "output_lang": self.target_lang,
            "translated_user": self.participant_identity
        }

        await self.room.local_participant.set_metadata(json.dumps(agent_metadata))

    async def update_settings(
        self,
        new_from_lang: str,
        new_target_lang: str,
        new_voice_id: str,
        new_translator: Translator,
    ):
        """
        Live settings update — called when a participant changes their metadata.
        Allows language/voice changes without restarting the agent session.
        """
        logger.info(
            f"Updating settings for {self.participant_identity}: "
            f"{new_from_lang} -> {new_target_lang}"
        )

        self.translator = new_translator
        self.from_lang = new_from_lang
        self.target_lang = new_target_lang

        # Re-initialize STT with the new source language
        self.stt_plugin = deepgram.STT(
            model="nova-3-general",
            language=self.from_lang,
            smart_format=True
        )
        self._stt = self.stt_plugin  # update the agent's internal STT reference

        # Only re-initialize TTS if the voice ID actually changed
        if self.voice_id != new_voice_id:
            self.voice_id = new_voice_id
            self.tts_plugin = elevenlabs.TTS(
                model="eleven_multilingual_v2",
                voice_id=self.voice_id
            )
            self._tts = self.tts_plugin  # update the agent's internal TTS reference

        await self.update_agent_metadata()

    async def on_user_turn_completed(self, _, new_message: llm.ChatMessage):
        """
        Called automatically by the agent framework when a user finishes speaking.
        This is where we receive the transcript, translate it, and publish TTS audio.
        """
        user_transcript = new_message.text_content

        # Nothing to process if STT returned empty text
        if not user_transcript.strip():
            return

        try:
            # --- Step 1: Clean participant identity for display name ---
            # Identities may have a random suffix like "__abc123" — strip it
            raw_identity = self.participant_identity
            clean_name = re.sub(r'_{2,}[a-zA-Z0-9]+$', '', raw_identity)

            # --- Step 2: Build the transcript payload ---
            payload = {
                "message": user_transcript,
                "timestamp": int(time.time() * 1000),  # milliseconds epoch
                "id": f"transcript-{raw_identity}-{time.time()}",
                "from": {
                    "identity": raw_identity,
                    "name": clean_name,
                    "isLocal": False
                }
            }

            data = json.dumps(payload).encode("utf-8")

            # --- Step 3: Publish raw transcript to all room participants ---
            # Frontend can receive this and display it as subtitles/transcript UI
            await self.room.local_participant.publish_data(
                payload=data,
                reliable=True,          # guaranteed delivery (like TCP, not UDP)
                topic="transcription_data"
            )

            # --- Step 4: Translate and synthesize speech (only if translation is enabled) ---
            if self.target_lang and self.target_lang != "no-translate":
                translated_text = self.translator.translate(user_transcript)

                logger.info(
                    f"[{self.participant_identity}] "
                    f"({self.from_lang}->{self.target_lang}) "
                    f"{user_transcript!r} -> {translated_text!r}"
                )

                # Publish audio track on first translation (lazy publish to avoid empty tracks)
                if not self._track_published:
                    await self.room.local_participant.publish_track(self.track)
                    self._track_published = True

                # Synthesize translated text into audio frames and send to the room
                async for synthesized in self.tts_plugin.synthesize(translated_text):
                    await self.audio_source.capture_frame(synthesized.frame)

        except Exception as e:
            logger.error(
                f"Translation/Speech error for {self.participant_identity}: {e}"
            )

        # StopResponse tells the agent framework to skip any default LLM response
        # for this turn — we handled everything manually above
        raise StopResponse()


# ================================================================
#  MULTI-USER TRANSCRIBER MANAGER
#  Manages one Transcriber + AgentSession per remote participant.
#  Creates a new agent when someone joins, cleans up when they leave.
# ================================================================

class MultiUserTranscriber:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._sessions: dict[str, AgentSession] = {}   # identity -> AgentSession
        self._tasks: set[asyncio.Task] = set()          # background async tasks
        self._agents: dict[str, Transcriber] = {}       # identity -> Transcriber
        self._user_from_lang: dict[str, str] = {}       # identity -> source language
        self._user_target_lang: dict[str, str] = {}     # identity -> target language
        self._user_voice: dict[str, str] = {}           # identity -> voice ID

    def start(self):
        """Register room event listeners to react to participant changes."""
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.on("participant_metadata_changed", self.on_metadata_changed)

    async def aclose(self):
        """Clean shutdown — remove all listeners, cancel tasks, close all sessions."""
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.off("participant_metadata_changed", self.on_metadata_changed)

        # Cancel all pending background tasks
        await utils.aio.cancel_and_wait(*self._tasks)

        # Close all active agent sessions gracefully
        await asyncio.gather(
            *[self._close_session(session) for session in self._sessions.values()]
        )

    def _update_settings(self, participant: rtc.RemoteParticipant):
        """
        Parse participant metadata and update language/voice settings.
        Expected metadata format (JSON string):
            {
                "user_lang": "bn",                       // source language (Deepgram code)
                "target_lang": "en",                     // translation target language
                "voice_id": "JBFqnCBsd6RMkjVDRZzb"      // ElevenLabs voice ID
            }
        """
        # Safe defaults — English input, no translation
        from_lang = "en"
        target_lang = "no-translate"
        voice_id = DEFAULT_VOICE_ID

        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                from_lang = metadata.get("user_lang", "en")
                target_lang = metadata.get("target_lang", "no-translate")
                voice_id = metadata.get("voice_id", DEFAULT_VOICE_ID)
            except Exception as e:
                logger.warning(
                    f"Metadata parse error for {participant.identity}: {e}"
                )

        # Build a fresh Translator instance for the new language pair
        new_translator = Translator(to_lang=target_lang, from_lang=from_lang)

        # Store per-user settings for later use when starting sessions
        self._user_from_lang[participant.identity] = from_lang
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id

        # If this participant already has a running agent, update it live
        if participant.identity in self._agents:
            agent = self._agents[participant.identity]
            asyncio.create_task(
                agent.update_settings(from_lang, target_lang, voice_id, new_translator)
            )

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        """Triggered when a participant updates their metadata (e.g., language change)."""
        self._update_settings(participant)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """
        Triggered when a new participant joins the room.
        Skips agent participants (identity prefix "agent-") to avoid infinite loops.
        """
        if (
            participant.identity in self._sessions
            or participant.identity.startswith("agent-")
        ):
            return  # already handled or is a LiveKit agent participant — skip

        self._update_settings(participant)

        # Start the session in background so we don't block the event callback
        task = asyncio.create_task(self._start_session(participant))
        self._tasks.add(task)

        def on_task_done(t: asyncio.Task):
            try:
                # Store the completed session keyed by participant identity
                self._sessions[participant.identity] = t.result()
            except Exception as e:
                logger.error(f"Session failed for {participant.identity}: {e}")
            finally:
                self._tasks.discard(t)

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """
        Triggered when a participant leaves the room.
        Cleans up all stored state and closes the agent session.
        """
        # Remove stored per-user settings
        self._user_from_lang.pop(participant.identity, None)
        self._user_target_lang.pop(participant.identity, None)
        self._user_voice.pop(participant.identity, None)
        self._agents.pop(participant.identity, None)

        # Pop the session — returns None if it was already removed
        session = self._sessions.pop(participant.identity, None)
        if session is None:
            return

        # Close the session asynchronously without blocking the event callback
        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:
        """
        Create and start an AgentSession + Transcriber for a single participant.
        RoomIO wires up the room's audio tracks to this agent's STT/TTS pipeline.
        """
        session = AgentSession()

        agent = Transcriber(
            participant_identity=participant.identity,
            translator=Translator(
                to_lang=self._user_target_lang[participant.identity],
                from_lang=self._user_from_lang[participant.identity],
            ),
            room=self.ctx.room,
            from_lang=self._user_from_lang[participant.identity],
            target_lang=self._user_target_lang[participant.identity],
            voice_id=self._user_voice.get(participant.identity, DEFAULT_VOICE_ID),
        )

        self._agents[participant.identity] = agent

        # RoomIO connects this participant's audio stream to the agent's STT/TTS pipeline
        room_io = RoomIO(
            agent_session=session,
            room=self.ctx.room,
            participant=participant,
            input_options=RoomInputOptions(
                text_enabled=False,                         # audio only, no text input
                noise_cancellation=noise_cancellation.BVC() # filter background noise
            ),
            output_options=RoomOutputOptions(
                transcription_enabled=True,   # publish transcript events to the room
                audio_enabled=False           # disable default agent audio output —
                                              # we manually publish via audio_source instead
            ),
        )

        await room_io.start()
        await session.start(agent=agent)
        await agent.update_agent_metadata()  # broadcast agent info to the room

        return session

    async def _close_session(self, sess: AgentSession) -> None:
        """Gracefully close an agent session and release its resources."""
        await sess.aclose()


# ================================================================
#  ENTRYPOINT
#  Called by the LiveKit Worker for each new job/room assignment.
# ================================================================

async def entrypoint(ctx: JobContext):
    # Automatically close the worker process when the room disconnects
    ctx.room.close_on_disconnect = True

    transcriber = MultiUserTranscriber(ctx)
    transcriber.start()  # register room event listeners

    # Connect to the LiveKit room, subscribe to audio tracks only (no video)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Handle participants who joined before this agent connected
    for p in ctx.room.remote_participants.values():
        transcriber.on_participant_connected(p)

    # Keep the worker alive — the real work happens in event callbacks
    while True:
        await asyncio.sleep(1)


# ================================================================
#  MAIN ENTRY POINT
#  Run with: python agent.py start
#  (This is the Start Command you set on Render)
# ================================================================

if __name__ == "__main__":
    # Start health check HTTP server in a background daemon thread BEFORE
    # launching the LiveKit worker. daemon=True ensures it exits with the main process.
    threading.Thread(target=run_health_server, daemon=True).start()

    # Start the LiveKit agent worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))