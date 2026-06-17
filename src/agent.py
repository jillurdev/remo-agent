# ============================================================
#  agent.py — LiveKit Multi-User Personalized Translator (FIXED)
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
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, utils

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
    raise RuntimeError("VOICE_API_URL is not set")

LANG_CODE_TO_NAME = {
    "en": "English",
    "bn": "Bengali",
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "es": "Spanish",
}

API_SAMPLE_RATE = 16000
CHUNK_SIZE = int(API_SAMPLE_RATE * 0.2) * 2  # 200ms PCM


# ================================================================
# Voice API CLIENT
# ================================================================
class VoiceAPIClient:
    def __init__(self, target_lang, on_mp3, on_transcript, owner_label="unknown"):
        self._target_lang = target_lang
        self._on_mp3 = on_mp3
        self._on_transcript = on_transcript
        self._owner_label = owner_label
        self._ws = None
        self._ready = threading.Event()
        self._closed = False
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run(self):
        try:
            self._ws = websocket.create_connection(VOICE_API_URL)

            raw = self._ws.recv()
            msg = json.loads(raw)

            lang_name = LANG_CODE_TO_NAME.get(self._target_lang, self._target_lang)
            self._ws.send(json.dumps({"type": "set_language", "language": lang_name}))

            self._ready.set()

            while not self._closed:
                data = self._ws.recv()

                if isinstance(data, bytes):
                    self._on_mp3(data)
                else:
                    msg = json.loads(data)

                    if msg.get("type") == "transcript":
                        self._on_transcript(msg)

        except Exception as e:
            logger.error(f"[{self._owner_label}] VoiceAPI error: {e}")

    def send_pcm(self, pcm_bytes: bytes):
        try:
            with self._lock:
                if self._ws:
                    self._ws.send_binary(pcm_bytes)
        except Exception as e:
            logger.error(f"[send_pcm error] {e}")

    def close(self):
        self._closed = True
        try:
            if self._ws:
                self._ws.close()
        except:
            pass


# ================================================================
# SPEAKER PIPELINE
# ================================================================
class SpeakerPipeline:
    def __init__(self, participant, room, manager, loop):
        self.participant = participant
        self.room = room
        self.manager = manager
        self.loop = loop

        self._pcm_buffer = bytearray()
        self._lang_clients = {}
        self._primary_lang = None
        self._audio_tasks = set()
        self._subscribed = set()

    # ---------------- FIX 1: STABLE AUDIO BUFFER ----------------
    def _on_audio_frame(self, frame: rtc.AudioFrame):
        if not self._lang_clients:
            return

        self._pcm_buffer.extend(bytes(frame.data))

        if len(self._pcm_buffer) >= CHUNK_SIZE:
            chunk = bytes(self._pcm_buffer)
            self._pcm_buffer.clear()

            for client in list(self._lang_clients.values()):
                client.send_pcm(chunk)

    # ---------------- TRACK ----------------
    def handle_track_subscribed(self, track: rtc.Track):
        if track.sid in self._subscribed:
            return

        self._subscribed.add(track.sid)
        asyncio.create_task(self._forward(track))

    async def _forward(self, track):
        stream = rtc.AudioStream(track, sample_rate=API_SAMPLE_RATE)

        async for event in stream:
            self._on_audio_frame(event.frame)

    # ---------------- FIX 2: SAFE HANDLERS ----------------
    def make_mp3_handler(self, lang):
        return lambda data: self._on_mp3(lang, data)

    def make_transcript_handler(self, lang):
        return lambda msg: self._on_transcript(lang, msg)

    def _on_mp3(self, lang, data):
        asyncio.run_coroutine_threadsafe(self._deliver(lang, data), self.loop)

    def _on_transcript(self, lang, msg):
        asyncio.run_coroutine_threadsafe(self._caption(lang, msg), self.loop)

    async def _deliver(self, lang, data):
        listeners = self.manager.listeners_for_lang(lang, self.participant.identity)
        if not listeners:
            return

        payload = json.dumps(
            {
                "type": "audio",
                "lang": lang,
                "audio_b64": base64.b64encode(data).decode(),
            }
        ).encode()

        await self.room.local_participant.publish_data(
            payload, reliable=True, destination_identities=listeners
        )

    async def _caption(self, lang, msg):
        if lang != self._primary_lang:
            return

        await self.room.local_participant.publish_data(
            json.dumps(
                {
                    "type": "caption",
                    "text": msg.get("original", ""),
                }
            ).encode(),
            reliable=True,
        )

    # ---------------- LANG RECONCILE ----------------
    def reconcile_langs(self, needed):
        removed = set(self._lang_clients) - needed
        added = needed - set(self._lang_clients)

        for lang in removed:
            self._lang_clients.pop(lang).close()

        for lang in added:
            client = VoiceAPIClient(
                target_lang=lang,
                on_mp3=self.make_mp3_handler(lang),
                on_transcript=self.make_transcript_handler(lang),
                owner_label=f"{self.participant.identity}->{lang}",
            )
            self._lang_clients[lang] = client
            self.loop.run_in_executor(None, client.start)

        if not self._primary_lang and self._lang_clients:
            self._primary_lang = next(iter(self._lang_clients))


# ================================================================
# MANAGER
# ================================================================
class MultiUserTranslationManager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pipelines = {}
        self.listener_targets = {}
        self.loop = asyncio.get_event_loop()

    def listeners_for_lang(self, lang, exclude):
        return [
            i for i, l in self.listener_targets.items() if l == lang and i != exclude
        ]

    def _recompute(self):
        self.listener_targets.clear()

        for ident, p in self.ctx.room.remote_participants.items():
            try:
                meta = json.loads(p.metadata or "{}")
                lang = meta.get("target_lang")
                if lang and lang != "no-translate":
                    self.listener_targets[ident] = lang
            except:
                pass

    def _reconcile(self):
        self._recompute()

        for ident, pipeline in self.pipelines.items():
            needed = {lang for i, lang in self.listener_targets.items() if i != ident}
            pipeline.reconcile_langs(needed)

    def on_participant_connected(self, p):
        if p.identity.startswith("agent-"):
            return

        pipeline = SpeakerPipeline(p, self.ctx.room, self, self.loop)
        self.pipelines[p.identity] = pipeline
        asyncio.create_task(pipeline.handle_track_subscribed)

        self._reconcile()

    def on_metadata_changed(self, p, _):
        self._reconcile()


# ================================================================
# ENTRY
# ================================================================
async def entrypoint(ctx: JobContext):
    manager = MultiUserTranslationManager(ctx)

    ctx.room.on("participant_connected", manager.on_participant_connected)
    ctx.room.on("participant_metadata_changed", manager.on_metadata_changed)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
