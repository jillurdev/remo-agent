import asyncio
import logging
import json
import os
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

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Without a configured handler, plain `logger.info(...)` calls may not show
# up in the console at all (root logger defaults to WARNING). This makes
# sure every join/leave/AI-mode/error event is actually visible.
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

    def __init__(
        self,
        target_lang: str,
        on_mp3: callable,
        on_transcript: callable,
        owner_identity: str = "unknown",
    ):
        self._target_lang = target_lang
        self._on_mp3 = on_mp3
        self._on_transcript = on_transcript
        self._owner_identity = owner_identity  # only used for clearer logs
        self._ws: websocket.WebSocket | None = None
        self._ready = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        logger.info(
            f"[{self._owner_identity}] Starting VoiceAPI connection thread "
            f"(target_lang={self._target_lang})"
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        ok = self._ready.wait(timeout=10)
        if not ok:
            logger.warning(
                f"[{self._owner_identity}] VoiceAPI did not become ready within timeout"
            )

    def _run(self):
        try:
            logger.info(
                f"[{self._owner_identity}] Connecting to VoiceAPI: {VOICE_API_URL}"
            )
            self._ws = websocket.create_connection(VOICE_API_URL)
            logger.info(f"[{self._owner_identity}] VoiceAPI WS connected")

            # Wait for ready signal
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "ready":
                logger.info(
                    f"[{self._owner_identity}] VoiceAPI ready, client_id={msg.get('client_id')}"
                )

            # Set output language
            lang_name = LANG_CODE_TO_NAME.get(self._target_lang, self._target_lang)
            self._ws.send(json.dumps({"type": "set_language", "language": lang_name}))

            # Wait for language_set confirmation
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "language_set":
                logger.info(
                    f"[{self._owner_identity}] VoiceAPI language set to: {msg.get('language')}"
                )

            self._ready.set()

            # Start keepalive
            threading.Thread(target=self._keepalive, daemon=True).start()

            # Receive loop
            while not self._closed:
                try:
                    data = self._ws.recv()
                    if isinstance(data, bytes):
                        logger.debug(
                            f"[{self._owner_identity}] Received MP3 audio chunk "
                            f"({len(data)} bytes)"
                        )
                        self._on_mp3(data)
                    else:
                        msg = json.loads(data)
                        msg_type = msg.get("type")
                        if msg_type == "transcript":
                            logger.info(
                                f"[{self._owner_identity}] Transcript received: "
                                f"original='{msg.get('original', '')[:60]}' "
                                f"translated='{msg.get('translated', '')[:60]}'"
                            )
                            self._on_transcript(msg)
                        elif msg_type == "pong":
                            logger.debug(
                                f"[{self._owner_identity}] Keepalive pong received"
                            )
                        elif msg_type == "processing":
                            logger.debug(
                                f"[{self._owner_identity}] VoiceAPI processing..."
                            )
                        elif msg_type == "error":
                            logger.error(
                                f"[{self._owner_identity}] VoiceAPI error: {msg.get('message')}"
                            )
                        else:
                            logger.debug(
                                f"[{self._owner_identity}] VoiceAPI msg: {msg}"
                            )
                except Exception as e:
                    if not self._closed:
                        logger.error(
                            f"[{self._owner_identity}] VoiceAPI recv error: {e}"
                        )
                    break

            logger.info(f"[{self._owner_identity}] VoiceAPI receive loop ended")

        except Exception as e:
            logger.error(f"[{self._owner_identity}] VoiceAPI connection failed: {e}")
            self._ready.set()  # unblock even on failure

    def _keepalive(self):
        while not self._closed:
            time.sleep(15)
            try:
                with self._lock:
                    if self._ws and not self._closed:
                        self._ws.send(json.dumps({"type": "ping"}))
                        logger.debug(f"[{self._owner_identity}] Keepalive ping sent")
            except Exception as e:
                logger.warning(f"[{self._owner_identity}] Keepalive ping failed: {e}")
                break

    def send_pcm(self, pcm_bytes: bytes):
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send_binary(pcm_bytes)
        except Exception as e:
            logger.error(f"[{self._owner_identity}] VoiceAPI send_pcm error: {e}")

    def flush(self):
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send(json.dumps({"type": "flush"}))
                    logger.info(f"[{self._owner_identity}] VoiceAPI flush sent")
        except Exception as e:
            logger.error(f"[{self._owner_identity}] VoiceAPI flush error: {e}")

    def update_language(self, new_target_lang: str):
        old_lang = self._target_lang
        self._target_lang = new_target_lang
        lang_name = LANG_CODE_TO_NAME.get(new_target_lang, new_target_lang)
        try:
            with self._lock:
                if self._ws and not self._closed:
                    self._ws.send(
                        json.dumps({"type": "set_language", "language": lang_name})
                    )
                    logger.info(
                        f"[{self._owner_identity}] VoiceAPI language updated: "
                        f"{old_lang} -> {new_target_lang} ({lang_name})"
                    )
        except Exception as e:
            logger.error(
                f"[{self._owner_identity}] VoiceAPI update_language error: {e}"
            )

    def close(self):
        logger.info(f"[{self._owner_identity}] Closing VoiceAPI connection")
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
        self._subscribed_track_sids: set[str] = set()

        self._audio_source = rtc.AudioSource(LIVEKIT_SAMPLE_RATE, LIVEKIT_CHANNELS)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant.identity}", self._audio_source
        )
        self._track_published = False

        logger.info(
            f"[{participant.identity}] Transcriber created "
            f"(from_lang={from_lang}, target_lang={target_lang})"
        )

        if target_lang and target_lang != "no-translate":
            logger.info(f"[{participant.identity}] AI translation is ON at startup")
            # Run in executor: VoiceAPIClient.start() blocks the calling thread
            # for up to 10s waiting for the connection to become ready. Doing
            # this directly here would freeze the whole asyncio event loop
            # (no audio forwarding, no room events) for everyone in the room.
            self._loop.run_in_executor(None, self._init_voice_client)
        else:
            logger.info(f"[{participant.identity}] AI translation is OFF at startup")

    def _init_voice_client(self):
        self._voice_client = VoiceAPIClient(
            target_lang=self.target_lang,
            on_mp3=self._on_mp3_received,
            on_transcript=self._on_transcript_received,
            owner_identity=self.participant.identity,
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
                logger.info(
                    f"[{self.participant.identity}] Translation audio track published"
                )

            pcm_data = await asyncio.get_event_loop().run_in_executor(
                None, mp3_bytes_to_pcm, mp3_bytes
            )
            logger.debug(
                f"[{self.participant.identity}] Decoded MP3 -> PCM "
                f"({len(pcm_data)} bytes), playing back"
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
            logger.error(f"[{self.participant.identity}] MP3 playback error: {e}")

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
            logger.info(
                f"[{raw_identity}] Transcript published to room "
                f"(lang={payload['target_language']})"
            )
        except Exception as e:
            logger.error(f"[{self.participant.identity}] Transcript publish error: {e}")

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
        old_from_lang = self.from_lang
        old_target_lang = self.target_lang

        was_on = bool(old_target_lang and old_target_lang != "no-translate")
        will_be_on = bool(new_target_lang and new_target_lang != "no-translate")

        logger.info(
            f"[{self.participant.identity}] Settings update requested: "
            f"from_lang {old_from_lang} -> {new_from_lang}, "
            f"target_lang {old_target_lang} -> {new_target_lang}"
        )

        self.from_lang = new_from_lang
        self.target_lang = new_target_lang

        if will_be_on:
            if self._voice_client:
                if not was_on:
                    logger.info(
                        f"[{self.participant.identity}] AI translation turned ON "
                        f"(target={new_target_lang})"
                    )
                self._voice_client.update_language(new_target_lang)
            else:
                logger.info(
                    f"[{self.participant.identity}] AI translation turned ON "
                    f"(target={new_target_lang})"
                )
                # Same reasoning as in __init__: keep the blocking 10s wait
                # off the event loop so other participants aren't affected.
                self._loop.run_in_executor(None, self._init_voice_client)
        else:
            if self._voice_client:
                logger.info(f"[{self.participant.identity}] AI translation turned OFF")
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
        logger.info(
            f"[{self.participant.identity}] Agent metadata updated "
            f"(input_lang={self.from_lang}, output_lang={self.target_lang})"
        )

    def handle_track_subscribed(self, track: rtc.Track):
        """Called for any audio track that gets subscribed for this participant.

        NOTE: `track_subscribed` is a Room-level event in the LiveKit Python SDK
        (RemoteParticipant has no `.on()` method), so MultiUserTranscriber listens
        on the room and routes the event here. This also handles tracks that were
        already subscribed before `start()` ran.
        """
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if track.sid in self._subscribed_track_sids:
            return  # avoid double-forwarding the same track
        self._subscribed_track_sids.add(track.sid)

        logger.info(f"[{self.participant.identity}] Audio track subscribed")
        task = asyncio.create_task(self._forward_audio(track))
        self._audio_tasks.add(task)
        task.add_done_callback(self._audio_tasks.discard)

    async def start(self):
        """Handle any audio tracks already subscribed at setup time."""
        for pub in self.participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(
                    f"[{self.participant.identity}] Existing audio track found, subscribing"
                )
                self.handle_track_subscribed(pub.track)

        await self._update_agent_metadata()

    async def _forward_audio(self, track: rtc.Track):
        """Stream audio from LiveKit track → Voice API."""
        logger.info(f"[{self.participant.identity}] Audio forwarding started")
        audio_stream = rtc.AudioStream(
            track, sample_rate=API_SAMPLE_RATE, num_channels=1
        )
        try:
            async for event in audio_stream:
                self.on_audio_frame(event.frame)
        finally:
            logger.info(f"[{self.participant.identity}] Audio forwarding stopped")

    def close(self):
        logger.info(f"[{self.participant.identity}] Closing transcriber")
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
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        logger.info("MultiUserTranscriber started, listening for room events")

    async def aclose(self):
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)
        self.ctx.room.off("participant_metadata_changed", self.on_metadata_changed)
        self.ctx.room.off("track_subscribed", self.on_track_subscribed)
        await utils.aio.cancel_and_wait(*self._tasks)
        for t in self._transcribers.values():
            t.close()
        logger.info("MultiUserTranscriber closed")

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

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Room-level event: routes to the right participant's transcriber.

        (RemoteParticipant itself has no `.on()` method, so this must be
        registered on the Room, not on individual participants.)
        """
        t = self._transcribers.get(participant.identity)
        if not t:
            logger.warning(
                f"[{participant.identity}] track_subscribed fired but no "
                f"transcriber is registered for them yet"
            )
            return
        t.handle_track_subscribed(track)

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        from_lang, target_lang = self._parse_settings(participant)
        logger.info(
            f"[{participant.identity}] Metadata changed -> "
            f"from_lang={from_lang}, target_lang={target_lang}"
        )
        if participant.identity in self._transcribers:
            t = self._transcribers[participant.identity]
            asyncio.create_task(t.update_settings(from_lang, target_lang))
        else:
            logger.warning(
                f"[{participant.identity}] Metadata changed but no transcriber exists for them"
            )

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant joined: {participant.identity}")

        if participant.identity.startswith("agent-"):
            logger.info(
                f"[{participant.identity}] Skipping (this is an agent identity)"
            )
            return

        if participant.identity in self._transcribers:
            logger.info(
                f"[{participant.identity}] Transcriber already exists, skipping setup"
            )
            return

        from_lang, target_lang = self._parse_settings(participant)
        ai_status = "ON" if target_lang and target_lang != "no-translate" else "OFF"
        logger.info(
            f"[{participant.identity}] Initial settings: from_lang={from_lang}, "
            f"target_lang={target_lang} (AI translation {ai_status})"
        )

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

        logger.info(
            f"[{participant.identity}] Transcriber registered "
            f"(total active: {len(self._transcribers)})"
        )

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"🚪 Participant left: {participant.identity}")
        transcriber = self._transcribers.pop(participant.identity, None)
        if transcriber:
            transcriber.close()
            logger.info(
                f"[{participant.identity}] Transcriber removed "
                f"(remaining active: {len(self._transcribers)})"
            )
        else:
            logger.warning(
                f"[{participant.identity}] Disconnected but no transcriber was tracked for them"
            )


async def entrypoint(ctx: JobContext):
    logger.info(f"=== Agent job starting | room={ctx.room.name} ===")
    ctx.room.close_on_disconnect = True

    transcriber = MultiUserTranscriber(ctx)
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to LiveKit room: {ctx.room.name}")

    existing = list(ctx.room.remote_participants.values())
    logger.info(f"Found {len(existing)} participant(s) already in the room")
    for p in existing:
        transcriber.on_participant_connected(p)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        logger.info(f"=== Agent job ending | room={ctx.room.name} ===")
        await transcriber.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
