import asyncio
import logging
import json
import time
import re
import threading
import websocket
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    utils,
)
from livekit.plugins import noise_cancellation
import av
import io

load_dotenv()
logger = logging.getLogger("transcriber")

VOICE_API_URL = "wss://voice-agent-437894783947.us-central1.run.app/ws/stream"

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

LIVEKIT_SAMPLE_RATE = 48000
LIVEKIT_CHANNELS = 1
API_SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
CHUNK_SIZE = int(API_SAMPLE_RATE * (CHUNK_DURATION_MS / 1000) * 2)  # 16-bit = 2 bytes


def mp3_bytes_to_pcm(
    mp3_bytes: bytes, target_sample_rate: int = LIVEKIT_SAMPLE_RATE
) -> bytes:
    """Convert MP3 bytes to raw 16-bit PCM bytes at target sample rate using av."""
    container = av.open(io.BytesIO(mp3_bytes), format="mp3")
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sample_rate,
    )
    pcm_chunks = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            pcm_chunks.append(bytes(resampled.planes[0]))
    for resampled in resampler.resample(None):
        pcm_chunks.append(bytes(resampled.planes[0]))
    return b"".join(pcm_chunks)


class VoiceAPIClient:
    """Persistent WebSocket connection to the Voice Translator API for one participant."""

    def __init__(self, target_lang: str, on_mp3: callable, on_transcript: callable):
        self._target_lang = target_lang
        self._on_mp3 = on_mp3
        self._on_transcript = on_transcript
        self._ws: websocket.WebSocket | None = None
        self._ready = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run(self):
        try:
            self._ws = websocket.create_connection(VOICE_API_URL)
            logger.info("VoiceAPI WS connected")

            # Wait for ready signal
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "ready":
                logger.info(f"VoiceAPI ready, client_id={msg.get('client_id')}")

            # Set output language
            lang_name = LANG_CODE_TO_NAME.get(self._target_lang, self._target_lang)
            self._ws.send(json.dumps({"type": "set_language", "language": lang_name}))

            # Wait for language_set confirmation
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "language_set":
                logger.info(f"VoiceAPI language set to: {msg.get('language')}")

            self._ready.set()

            # Start keepalive
            threading.Thread(target=self._keepalive, daemon=True).start()

            # Receive loop
            while not self._closed:
                try:
                    data = self._ws.recv()
                    if isinstance(data, bytes):
                        self._on_mp3(data)
                    else:
                        msg = json.loads(data)
                        msg_type = msg.get("type")
                        if msg_type == "transcript":
                            self._on_transcript(msg)
                        elif msg_type == "pong":
                            pass
                        elif msg_type == "processing":
                            logger.debug("VoiceAPI processing...")
                        elif msg_type == "error":
                            logger.error(f"VoiceAPI error: {msg.get('message')}")
                        else:
                            logger.debug(f"VoiceAPI msg: {msg}")
                except Exception as e:
                    if not self._closed:
                        logger.error(f"VoiceAPI recv error: {e}")
                    break

        except Exception as e:
            logger.error(f"VoiceAPI connection failed: {e}")
            self._ready.set()  # unblock even on failure

    def _keepalive(self):
        while not self._closed:
            time.sleep(15)
            try:
                with self._lock:
                    if self._ws and not self._closed:
                        self._ws.send(json.dumps({"type": "ping"}))
            except Exception as e:
                logger.warning(f"Keepalive ping failed: {e}")
                break

    def send_pcm(self, pcm_bytes: bytes):
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send_binary(pcm_bytes)
        except Exception as e:
            logger.error(f"VoiceAPI send_pcm error: {e}")

    def flush(self):
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error(f"VoiceAPI flush error: {e}")

    def update_language(self, new_target_lang: str):
        self._target_lang = new_target_lang
        lang_name = LANG_CODE_TO_NAME.get(new_target_lang, new_target_lang)
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send(
                        json.dumps({"type": "set_language", "language": lang_name})
                    )
                    logger.info(f"VoiceAPI language updated to: {lang_name}")
        except Exception as e:
            logger.error(f"VoiceAPI update_language error: {e}")

    def close(self):
        self._closed = True
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass


class ParticipantTranscriber:
    """Handles audio forwarding and translation for a single participant."""

    def __init__(
        self,
        *,
        participant: rtc.RemoteParticipant,
        room: rtc.Room,
        from_lang: str,
        target_lang: str,
        loop: asyncio.AbstractEventLoop,
    ):
        self.participant = participant
        self.room = room
        self.from_lang = from_lang
        self.target_lang = target_lang
        self._loop = loop

        self._pcm_buffer = bytearray()
        self._voice_client: VoiceAPIClient | None = None
        self._audio_tasks: set[asyncio.Task] = set()

        self._audio_source = rtc.AudioSource(LIVEKIT_SAMPLE_RATE, LIVEKIT_CHANNELS)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant.identity}", self._audio_source
        )
        self._track_published = False

        if target_lang and target_lang != "no-translate":
            self._init_voice_client()

    def _init_voice_client(self):
        self._voice_client = VoiceAPIClient(
            target_lang=self.target_lang,
            on_mp3=self._on_mp3_received,
            on_transcript=self._on_transcript_received,
        )
        self._voice_client.start()

    def _on_mp3_received(self, mp3_bytes: bytes):
        asyncio.run_coroutine_threadsafe(self._play_mp3(mp3_bytes), self._loop)

    def _on_transcript_received(self, msg: dict):
        asyncio.run_coroutine_threadsafe(self._publish_transcript(msg), self._loop)

    async def _play_mp3(self, mp3_bytes: bytes):
        try:
            if not self._track_published:
                await self.room.local_participant.publish_track(self._track)
                self._track_published = True

            pcm_data = await asyncio.get_event_loop().run_in_executor(
                None, mp3_bytes_to_pcm, mp3_bytes
            )

            FRAME_SIZE = LIVEKIT_SAMPLE_RATE // 10  # 100ms frames
            frame_bytes = FRAME_SIZE * 2  # 16-bit

            for i in range(0, len(pcm_data), frame_bytes):
                chunk = pcm_data[i : i + frame_bytes]
                if len(chunk) < frame_bytes:
                    chunk = chunk + b"\x00" * (frame_bytes - len(chunk))

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=LIVEKIT_SAMPLE_RATE,
                    num_channels=LIVEKIT_CHANNELS,
                    samples_per_channel=FRAME_SIZE,
                )
                await self._audio_source.capture_frame(frame)

        except Exception as e:
            logger.error(f"MP3 playback error for {self.participant.identity}: {e}")

    async def _publish_transcript(self, msg: dict):
        try:
            raw_identity = self.participant.identity
            clean_name = re.sub(r"_{2,}[a-zA-Z0-9]+$", "", raw_identity)

            payload = {
                "message": msg.get("original", ""),
                "translated": msg.get("translated", ""),
                "target_language": msg.get("target_language", ""),
                "timestamp": int(time.time() * 1000),
                "id": f"transcript-{raw_identity}-{time.time()}",
                "from": {
                    "identity": raw_identity,
                    "name": clean_name,
                    "isLocal": False,
                },
            }

            await self.room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="transcription_data",
            )
        except Exception as e:
            logger.error(
                f"Transcript publish error for {self.participant.identity}: {e}"
            )

    def on_audio_frame(self, frame: rtc.AudioFrame):
        """Feed raw PCM audio frames into the Voice API in 200ms chunks."""
        if not self._voice_client:
            return

        self._pcm_buffer.extend(bytes(frame.data))

        while len(self._pcm_buffer) >= CHUNK_SIZE:
            chunk = bytes(self._pcm_buffer[:CHUNK_SIZE])
            self._pcm_buffer = self._pcm_buffer[CHUNK_SIZE:]
            self._voice_client.send_pcm(chunk)

    async def update_settings(self, new_from_lang: str, new_target_lang: str):
        logger.info(
            f"Updating settings for {self.participant.identity}: {new_from_lang} -> {new_target_lang}"
        )
        self.from_lang = new_from_lang
        self.target_lang = new_target_lang

        if new_target_lang and new_target_lang != "no-translate":
            if self._voice_client:
                self._voice_client.update_language(new_target_lang)
            else:
                self._init_voice_client()
        else:
            if self._voice_client:
                self._voice_client.close()
                self._voice_client = None

        await self._update_agent_metadata()

    async def _update_agent_metadata(self):
        agent_metadata = {
            "is_agent": True,
            "input_lang": self.from_lang,
            "output_lang": self.target_lang,
            "translated_user": self.participant.identity,
        }
        await self.room.local_participant.set_metadata(json.dumps(agent_metadata))
        logger.info(f"Agent metadata updated for {self.participant.identity}")

    async def start(self):
        """Subscribe to participant's audio tracks."""

        async def on_track_subscribed(track: rtc.Track, *_):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                task = asyncio.create_task(self._forward_audio(track))
                self._audio_tasks.add(task)
                task.add_done_callback(self._audio_tasks.discard)

        self.participant.on("track_subscribed", on_track_subscribed)

        # Handle already-subscribed tracks
        for pub in self.participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                task = asyncio.create_task(self._forward_audio(pub.track))
                self._audio_tasks.add(task)
                task.add_done_callback(self._audio_tasks.discard)

        await self._update_agent_metadata()

    async def _forward_audio(self, track: rtc.Track):
        """Stream audio from LiveKit track → Voice API."""
        logger.info(f"Audio forwarding started for {self.participant.identity}")
        audio_stream = rtc.AudioStream(
            track, sample_rate=API_SAMPLE_RATE, num_channels=1
        )
        async for event in audio_stream:
            self.on_audio_frame(event.frame)

    def close(self):
        for task in self._audio_tasks:
            task.cancel()
        if self._voice_client:
            self._voice_client.close()


class MultiUserTranscriber:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._transcribers: dict[str, ParticipantTranscriber] = {}
        self._tasks: set[asyncio.Task] = set()
        self._loop = asyncio.get_event_loop()

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.on("participant_metadata_changed", self.on_metadata_changed)

    async def aclose(self):
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.off("participant_metadata_changed", self.on_metadata_changed)
        await utils.aio.cancel_and_wait(*self._tasks)
        for t in self._transcribers.values():
            t.close()

    def _parse_settings(self, participant: rtc.RemoteParticipant):
        from_lang = "en"
        target_lang = "no-translate"

        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                from_lang = metadata.get("user_lang", "en")
                target_lang = metadata.get("target_lang", "no-translate")
            except Exception as e:
                logger.warning(f"Metadata parse error for {participant.identity}: {e}")

        return from_lang, target_lang

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        from_lang, target_lang = self._parse_settings(participant)
        if participant.identity in self._transcribers:
            t = self._transcribers[participant.identity]
            asyncio.create_task(t.update_settings(from_lang, target_lang))

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        if (
            participant.identity in self._transcribers
            or participant.identity.startswith("agent-")
        ):
            return

        from_lang, target_lang = self._parse_settings(participant)

        transcriber = ParticipantTranscriber(
            participant=participant,
            room=self.ctx.room,
            from_lang=from_lang,
            target_lang=target_lang,
            loop=self._loop,
        )
        self._transcribers[participant.identity] = transcriber

        task = asyncio.create_task(transcriber.start())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        transcriber = self._transcribers.pop(participant.identity, None)
        if transcriber:
            transcriber.close()


async def entrypoint(ctx: JobContext):
    ctx.room.close_on_disconnect = True

    transcriber = MultiUserTranscriber(ctx)
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    for p in ctx.room.remote_participants.values():
        transcriber.on_participant_connected(p)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
