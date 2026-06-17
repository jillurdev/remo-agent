# ============================================================
#  PRODUCTION FIXED LiveKit Translator (VoiceAPI)
# ============================================================

import asyncio
import logging
import json
import os
import time
import base64
import threading
import websocket

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

VOICE_API_URL = os.getenv("VOICE_API_URL")

API_SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_SIZE = int(API_SAMPLE_RATE * (CHUNK_MS / 1000)) * 2


# ============================================================
# Voice API CLIENT
# ============================================================
class VoiceAPIClient:
    def __init__(self, lang, on_mp3, on_transcript, label=""):
        self.lang = lang
        self.on_mp3 = on_mp3
        self.on_transcript = on_transcript
        self.label = label

        self.ws = None
        self.closed = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.ws = websocket.create_connection(VOICE_API_URL)

            self.ws.send(json.dumps({"type": "set_language", "language": self.lang}))

            while not self.closed:
                msg = self.ws.recv()

                if isinstance(msg, bytes):
                    self.on_mp3(msg)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "transcript":
                        self.on_transcript(data)

        except Exception as e:
            logger.error(f"[VoiceAPI] {self.label} error: {e}")

    def send_pcm(self, pcm: bytes):
        try:
            with self.lock:
                if self.ws:
                    self.ws.send_binary(pcm)
        except:
            pass

    def close(self):
        self.closed = True
        try:
            if self.ws:
                self.ws.close()
        except:
            pass


# ============================================================
# SPEAKER PIPELINE
# ============================================================
class SpeakerPipeline:
    def __init__(self, participant, room, manager, loop):
        self.participant = participant
        self.room = room
        self.manager = manager
        self.loop = loop

        self.buffer = bytearray()
        self.clients = {}
        self.primary_lang = None
        self.last_flush = time.time()

    # AUDIO STREAM
    def on_audio_frame(self, frame: rtc.AudioFrame):
        if not self.clients:
            return

        self.buffer.extend(bytes(frame.data))

        now = time.time()

        if len(self.buffer) >= CHUNK_SIZE or (now - self.last_flush) > 0.1:
            chunk = bytes(self.buffer)
            self.buffer.clear()
            self.last_flush = now

            for c in self.clients.values():
                c.send_pcm(chunk)

    # TRACK HANDLER (FIXED)
    def handle_track(self, track: rtc.Track):
        asyncio.create_task(self._forward(track))

    async def _forward(self, track):
        stream = rtc.AudioStream(track, sample_rate=API_SAMPLE_RATE)

        async for ev in stream:
            self.on_audio_frame(ev.frame)

    # CALLBACK WRAPPERS
    def mp3_handler(self, lang):
        return lambda data: self._mp3(lang, data)

    def transcript_handler(self, lang):
        return lambda msg: self._transcript(lang, msg)

    def _mp3(self, lang, data):
        asyncio.run_coroutine_threadsafe(self._deliver(lang, data), self.loop)

    def _transcript(self, lang, msg):
        asyncio.run_coroutine_threadsafe(self._caption(lang, msg), self.loop)

    async def _deliver(self, lang, data):
        listeners = self.manager.listeners(lang, self.participant.identity)
        if not listeners:
            return

        await self.room.local_participant.publish_data(
            json.dumps(
                {
                    "type": "audio",
                    "lang": lang,
                    "audio": base64.b64encode(data).decode(),
                }
            ).encode(),
            reliable=True,
            destination_identities=listeners,
        )

    async def _caption(self, lang, msg):
        if lang != self.primary_lang:
            return

        await self.room.local_participant.publish_data(
            json.dumps({"type": "caption", "text": msg.get("original", "")}).encode(),
            reliable=True,
        )

    # LANGUAGE CONTROL
    def reconcile(self, needed):
        remove = set(self.clients) - needed
        add = needed - set(self.clients)

        for l in remove:
            self.clients.pop(l).close()

        for l in add:
            client = VoiceAPIClient(
                l,
                self.mp3_handler(l),
                self.transcript_handler(l),
                label=f"{self.participant.identity}->{l}",
            )
            self.clients[l] = client
            self.loop.call_soon_threadsafe(client.start)

        if not self.primary_lang and self.clients:
            self.primary_lang = next(iter(self.clients))


# ============================================================
# MANAGER (FIXED)
# ============================================================
class Manager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pipelines = {}
        self.listeners_map = {}
        self.loop = asyncio.get_event_loop()

    def listeners(self, lang, exclude):
        return [i for i, l in self.listeners_map.items() if l == lang and i != exclude]

    def recompute(self):
        self.listeners_map.clear()

        for i, p in self.ctx.room.remote_participants.items():
            try:
                m = json.loads(p.metadata or "{}")
                lang = m.get("target_lang")
                if lang and lang != "no-translate":
                    self.listeners_map[i] = lang
            except:
                pass

    def reconcile_all(self):
        self.recompute()

        for i, pipe in self.pipelines.items():
            needed = {lang for x, lang in self.listeners_map.items() if x != i}
            pipe.reconcile(needed)

    # FIXED: NO create_task here
    def on_join(self, p):
        if p.identity.startswith("agent-"):
            return

        pipe = SpeakerPipeline(p, self.ctx.room, self, self.loop)
        self.pipelines[p.identity] = pipe

        self.reconcile_all()

    def on_meta(self, p, _):
        self.reconcile_all()

    # FIXED: THIS WAS MISSING BEFORE
    def on_track(self, track, pub, participant):
        pipe = self.pipelines.get(participant.identity)
        if pipe:
            pipe.handle_track(track)


# ============================================================
# ENTRYPOINT
# ============================================================
async def entrypoint(ctx: JobContext):
    m = Manager(ctx)

    ctx.room.on("participant_connected", m.on_join)
    ctx.room.on("participant_metadata_changed", m.on_meta)

    # 🔥 IMPORTANT FIX
    ctx.room.on("track_subscribed", m.on_track)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
