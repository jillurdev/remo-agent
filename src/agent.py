# ============================================================
#  agent.py — LiveKit Multi-User Personalized Translator
#  Uses our own Voice API (STT + translation + TTS in one
#  websocket pipeline) instead of Deepgram/GoogleTranslator/edge_tts.
#
#  Architecture (personalized model):
#  - Every participant can simultaneously be a SPEAKER and a LISTENER.
#  - A listener picks their own `target_lang` (independent of what
#    anyone else picked, independent of what language the speaker
#    is speaking in). "no-translate" means "AI mode off for me".
#  - For each speaker, we keep one Voice API connection PER DISTINCT
#    language currently wanted by *other* participants in the room
#    (not one connection per listener — listeners sharing a language
#    share one connection's output). Connections are created/closed
#    on the fly as people join/leave/change their language.
#  - Translated audio (mp3) is delivered ONLY to the listeners who
#    asked for that language, via a data-channel message addressed
#    with destination_identities — never broadcast to everyone.
#  - The original-language transcript (captions in the speaker's own
#    language) is still broadcast to everyone, but only while at
#    least one Voice API connection is open for that speaker (since
#    STT only happens inside those connections). We elect one
#    connection as "primary" per speaker purely for caption purposes;
#    every connection still delivers its own translated audio.
# ============================================================

import asyncio
import logging
import json
import os
import time
import re
import threading
import base64
import websocket
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    utils,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transcriber")

VOICE_API_URL = os.getenv("VOICE_API_URL")
if not VOICE_API_URL:
    raise RuntimeError("VOICE_API_URL is not set in environment (.env)")

logger.info(f"Voice API URL loaded from environment: {VOICE_API_URL}")

LANG_CODE_TO_NAME = {
    "en": "English",
    "bn": "Bengali",
    "fr": "French",
    "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "fa": "Persian",
    "he": "Hebrew",
    "cs": "Czech",
    "da": "Danish",
    "el": "Greek",
    "fi": "Finnish",
    "hu": "Hungarian",
    "no": "Norwegian",
    "ro": "Romanian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "af": "Afrikaans",
    "ha": "Hausa",
    "ig": "Igbo",
    "yo": "Yoruba",
}

API_SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
CHUNK_SIZE = int(API_SAMPLE_RATE * (CHUNK_DURATION_MS / 1000) * 2)  # 16-bit = 2 bytes


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
#  VOICE API CLIENT (one persistent connection = one speaker+language pair)
# ================================================================


class VoiceAPIClient:
    """Persistent WebSocket connection to the Voice Translator API.

    One instance handles ONE (speaker, target_language) pair: raw PCM
    audio goes in, translated MP3 audio + transcript messages come out.
    """

    def __init__(
        self,
        target_lang: str,
        on_mp3: callable,
        on_transcript: callable,
        owner_label: str = "unknown",
    ):
        self._target_lang = target_lang
        self._on_mp3 = on_mp3
        self._on_transcript = on_transcript
        self._owner_label = owner_label
        self._ws: websocket.WebSocket | None = None
        self._ready = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        """Blocking: connects and waits (up to 10s) for readiness.
        Callers should invoke this via loop.run_in_executor so the
        asyncio event loop is never frozen waiting on this.
        """
        logger.info(f"[{self._owner_label}] Starting VoiceAPI connection thread")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        ok = self._ready.wait(timeout=10)
        if not ok:
            logger.warning(
                f"[{self._owner_label}] VoiceAPI did not become ready within timeout"
            )

    def _run(self):
        try:
            logger.info(
                f"[{self._owner_label}] Connecting to VoiceAPI: {VOICE_API_URL}"
            )
            self._ws = websocket.create_connection(VOICE_API_URL)
            logger.info(f"[{self._owner_label}] VoiceAPI WS connected")

            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "ready":
                logger.info(
                    f"[{self._owner_label}] VoiceAPI ready, client_id={msg.get('client_id')}"
                )

            lang_name = LANG_CODE_TO_NAME.get(self._target_lang, self._target_lang)
            self._ws.send(json.dumps({"type": "set_language", "language": lang_name}))

            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "language_set":
                logger.info(
                    f"[{self._owner_label}] VoiceAPI language set to: {msg.get('language')}"
                )

            self._ready.set()

            threading.Thread(target=self._keepalive, daemon=True).start()

            while not self._closed:
                try:
                    data = self._ws.recv()
                    if isinstance(data, bytes):
                        logger.debug(
                            f"[{self._owner_label}] Received MP3 chunk ({len(data)} bytes)"
                        )
                        self._on_mp3(data)
                    else:
                        msg = json.loads(data)
                        msg_type = msg.get("type")
                        if msg_type == "transcript":
                            logger.info(
                                f"[{self._owner_label}] Transcript: "
                                f"original='{msg.get('original', '')[:60]}' "
                                f"translated='{msg.get('translated', '')[:60]}'"
                            )
                            self._on_transcript(msg)
                        elif msg_type == "pong":
                            logger.debug(f"[{self._owner_label}] pong")
                        elif msg_type == "processing":
                            logger.debug(f"[{self._owner_label}] processing...")
                        elif msg_type == "error":
                            logger.error(
                                f"[{self._owner_label}] VoiceAPI error: {msg.get('message')}"
                            )
                        else:
                            logger.debug(f"[{self._owner_label}] VoiceAPI msg: {msg}")
                except Exception as e:
                    if not self._closed:
                        logger.error(f"[{self._owner_label}] VoiceAPI recv error: {e}")
                    break

            logger.info(f"[{self._owner_label}] VoiceAPI receive loop ended")

        except Exception as e:
            logger.error(f"[{self._owner_label}] VoiceAPI connection failed: {e}")
            self._ready.set()

    def _keepalive(self):
        while not self._closed:
            time.sleep(15)
            try:
                with self._lock:
                    if self._ws and not self._closed:
                        self._ws.send(json.dumps({"type": "ping"}))
            except Exception as e:
                logger.warning(f"[{self._owner_label}] Keepalive ping failed: {e}")
                break

    def send_pcm(self, pcm_bytes: bytes):
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send_binary(pcm_bytes)
        except Exception as e:
            logger.error(f"[{self._owner_label}] send_pcm error: {e}")

    def close(self):
        logger.info(f"[{self._owner_label}] Closing VoiceAPI connection")
        self._closed = True
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass


# ================================================================
#  SPEAKER PIPELINE — one per connected participant
# ================================================================


class SpeakerPipeline:
    """Forwards one participant's mic audio into N parallel VoiceAPI
    connections (one per distinct language currently wanted by OTHER
    participants), and routes each connection's translated audio back
    out, personalized, to only the listeners who asked for that language.
    """

    def __init__(
        self,
        *,
        participant: rtc.RemoteParticipant,
        room: rtc.Room,
        manager: "MultiUserTranslationManager",
        loop: asyncio.AbstractEventLoop,
    ):
        self.participant = participant
        self.room = room
        self.manager = manager
        self._loop = loop

        self._pcm_buffer = bytearray()
        self._lang_clients: dict[str, VoiceAPIClient] = {}
        self._primary_lang: str | None = None
        self._audio_tasks: set[asyncio.Task] = set()
        self._subscribed_track_sids: set[str] = set()
        self._closed = False

        logger.info(f"[{participant.identity}] Speaker pipeline created")

    async def start(self):
        """Pick up any audio tracks already subscribed at setup time."""
        for pub in self.participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(
                    f"[{self.participant.identity}] Existing audio track found, subscribing"
                )
                self.handle_track_subscribed(pub.track)

    def handle_track_subscribed(self, track: rtc.Track):
        """Called for newly-subscribed audio tracks.

        NOTE: `track_subscribed` only fires on the Room object in the
        LiveKit Python SDK (RemoteParticipant has no `.on()` method),
        so MultiUserTranslationManager listens at the room level and
        routes the event here.
        """
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if track.sid in self._subscribed_track_sids:
            return
        self._subscribed_track_sids.add(track.sid)

        logger.info(f"[{self.participant.identity}] Audio track subscribed")
        task = asyncio.create_task(self._forward_audio(track))
        self._audio_tasks.add(task)
        task.add_done_callback(self._audio_tasks.discard)

    async def _forward_audio(self, track: rtc.Track):
        logger.info(f"[{self.participant.identity}] Audio forwarding started")
        audio_stream = rtc.AudioStream(
            track, sample_rate=API_SAMPLE_RATE, num_channels=1
        )
        try:
            async for event in audio_stream:
                self._on_audio_frame(event.frame)
        finally:
            logger.info(f"[{self.participant.identity}] Audio forwarding stopped")

    def _on_audio_frame(self, frame: rtc.AudioFrame):
        if not self._lang_clients:
            return  # nobody is listening to this speaker right now
        self._pcm_buffer.extend(bytes(frame.data))
        while len(self._pcm_buffer) >= CHUNK_SIZE:
            chunk = bytes(self._pcm_buffer[:CHUNK_SIZE])
            self._pcm_buffer = self._pcm_buffer[CHUNK_SIZE:]
            for client in list(self._lang_clients.values()):
                client.send_pcm(chunk)

    def reconcile_langs(self, needed_langs: set[str]):
        """Open VoiceAPI connections for newly-needed languages, close
        the ones nobody wants anymore, and re-elect a primary language
        (used only to decide whose transcript becomes the broadcast
        original-language caption).
        """
        removed = set(self._lang_clients.keys()) - needed_langs
        added = needed_langs - set(self._lang_clients.keys())

        for lang in removed:
            logger.info(
                f"[{self.participant.identity}] AI translation OFF for language: {lang} (no listeners left)"
            )
            self._lang_clients.pop(lang).close()

        for lang in added:
            logger.info(
                f"[{self.participant.identity}] AI translation ON for language: {lang} (listener(s) requested it)"
            )
            owner_label = f"{self.participant.identity}->{lang}"
            client = VoiceAPIClient(
                target_lang=lang,
                on_mp3=lambda data, lang=lang: self._on_mp3(lang, data),
                on_transcript=lambda msg, lang=lang: self._on_transcript(lang, msg),
                owner_label=owner_label,
            )
            self._lang_clients[lang] = client
            # Don't block the event loop with the up-to-10s readiness wait.
            self._loop.run_in_executor(None, client.start)

        if self._primary_lang not in self._lang_clients:
            self._primary_lang = next(iter(self._lang_clients), None)
            if self._primary_lang:
                logger.info(
                    f"[{self.participant.identity}] Primary caption language is now: {self._primary_lang}"
                )

    def _on_mp3(self, lang: str, mp3_bytes: bytes):
        asyncio.run_coroutine_threadsafe(
            self._deliver_audio(lang, mp3_bytes), self._loop
        )

    def _on_transcript(self, lang: str, msg: dict):
        asyncio.run_coroutine_threadsafe(self._handle_transcript(lang, msg), self._loop)

    async def _deliver_audio(self, lang: str, mp3_bytes: bytes):
        listener_identities = self.manager.listeners_for_lang(
            lang, exclude=self.participant.identity
        )
        if not listener_identities:
            return  # everyone who wanted this language already left/changed their mind

        audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
        payload = json.dumps(
            {
                "type": "translation_audio",
                "audio_b64": audio_b64,
                "lang": lang,
                "from": self.participant.identity,
            }
        ).encode("utf-8")

        try:
            await self.room.local_participant.publish_data(
                payload=payload,
                reliable=True,
                topic="translation_audio",
                destination_identities=listener_identities,
            )
            logger.info(
                f"[{self.participant.identity}->{lang}] Sent {len(mp3_bytes)} bytes "
                f"to {len(listener_identities)} listener(s): {listener_identities}"
            )
        except Exception as e:
            logger.error(
                f"[{self.participant.identity}->{lang}] Failed to send audio: {e}"
            )

    async def _handle_transcript(self, lang: str, msg: dict):
        original = msg.get("original", "")
        if lang == self._primary_lang:
            await self._publish_original_caption(original)

    async def _publish_original_caption(self, text: str):
        if not text.strip():
            return
        raw_identity = self.participant.identity
        clean_name = re.sub(r"_{2,}[a-zA-Z0-9]+$", "", raw_identity)
        payload = {
            "message": text,
            "timestamp": int(time.time() * 1000),
            "id": f"transcript-{raw_identity}-{time.time()}",
            "from": {"identity": raw_identity, "name": clean_name, "isLocal": False},
        }
        try:
            await self.room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="transcription_data",
            )
            logger.info(f"[{raw_identity}] Original-language caption published")
        except Exception as e:
            logger.error(f"[{raw_identity}] Caption publish error: {e}")

    def close(self):
        logger.info(f"[{self.participant.identity}] Closing speaker pipeline")
        self._closed = True
        for task in self._audio_tasks:
            task.cancel()
        for client in self._lang_clients.values():
            client.close()
        self._lang_clients.clear()


# ================================================================
#  MULTI-USER TRANSLATION MANAGER
# ================================================================


class MultiUserTranslationManager:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._pipelines: dict[str, SpeakerPipeline] = {}
        self._listener_targets: dict[
            str, str
        ] = {}  # identity -> target_lang (only non "no-translate")
        self._tasks: set[asyncio.Task] = set()
        self._loop = asyncio.get_event_loop()

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.on("participant_metadata_changed", self.on_metadata_changed)
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        logger.info("MultiUserTranslationManager started, listening for room events")

    async def aclose(self):
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.off("participant_metadata_changed", self.on_metadata_changed)
        self.ctx.room.off("track_subscribed", self.on_track_subscribed)
        await utils.aio.cancel_and_wait(*self._tasks)
        for pipeline in self._pipelines.values():
            pipeline.close()
        logger.info("MultiUserTranslationManager closed")

    def _parse_settings(self, participant: rtc.RemoteParticipant):
        from_lang = "en"
        target_lang = "no-translate"
        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                from_lang = metadata.get("user_lang", "en")
                target_lang = metadata.get("target_lang", "no-translate")
            except Exception as e:
                logger.warning(f"[{participant.identity}] Metadata parse error: {e}")
        return from_lang, target_lang

    def listeners_for_lang(self, lang: str, exclude: str) -> list[str]:
        return [
            ident
            for ident, t in self._listener_targets.items()
            if t == lang and ident != exclude
        ]

    def _recompute_listener_targets(self):
        """Rebuild the identity -> target_lang map from every participant's
        current metadata (only those with AI mode ON are included)."""
        old = dict(self._listener_targets)
        self._listener_targets.clear()
        for ident, p in self.ctx.room.remote_participants.items():
            _, target_lang = self._parse_settings(p)
            if target_lang and target_lang != "no-translate":
                self._listener_targets[ident] = target_lang

        if old != self._listener_targets:
            logger.info(f"Active listener languages in room: {self._listener_targets}")

    def _reconcile_all(self):
        self._recompute_listener_targets()
        for ident, pipeline in self._pipelines.items():
            needed = {
                lang
                for l_ident, lang in self._listener_targets.items()
                if l_ident != ident
            }
            pipeline.reconcile_langs(needed)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        pipeline = self._pipelines.get(participant.identity)
        if not pipeline:
            logger.warning(
                f"[{participant.identity}] track_subscribed fired but no pipeline exists yet"
            )
            return
        pipeline.handle_track_subscribed(track)

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        from_lang, target_lang = self._parse_settings(participant)
        ai_status = "ON" if target_lang and target_lang != "no-translate" else "OFF"
        logger.info(
            f"[{participant.identity}] Metadata changed -> target_lang={target_lang} (AI listening {ai_status})"
        )
        self._reconcile_all()

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant joined: {participant.identity}")

        if participant.identity.startswith("agent-"):
            logger.info(f"[{participant.identity}] Skipping (agent identity)")
            return

        if participant.identity in self._pipelines:
            logger.info(
                f"[{participant.identity}] Pipeline already exists, skipping setup"
            )
            return

        pipeline = SpeakerPipeline(
            participant=participant, room=self.ctx.room, manager=self, loop=self._loop
        )
        self._pipelines[participant.identity] = pipeline

        task = asyncio.create_task(pipeline.start())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        logger.info(
            f"[{participant.identity}] Pipeline registered (total active: {len(self._pipelines)})"
        )
        self._reconcile_all()

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🚪 Participant left: {participant.identity}")
        pipeline = self._pipelines.pop(participant.identity, None)
        if pipeline:
            pipeline.close()
            logger.info(
                f"[{participant.identity}] Pipeline removed (remaining active: {len(self._pipelines)})"
            )
        self._reconcile_all()


# ================================================================
#  ENTRYPOINT
# ================================================================


async def entrypoint(ctx: JobContext):
    logger.info(f"=== Agent job starting | room={ctx.room.name} ===")
    ctx.room.close_on_disconnect = True

    manager = MultiUserTranslationManager(ctx)
    manager.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to LiveKit room: {ctx.room.name}")

    existing = list(ctx.room.remote_participants.values())
    logger.info(f"Found {len(existing)} participant(s) already in the room")
    for p in existing:
        manager.on_participant_connected(p)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        logger.info(f"=== Agent job ending | room={ctx.room.name} ===")
        await manager.aclose()


if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
