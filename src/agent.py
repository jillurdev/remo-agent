import asyncio
import logging
import json
import time
import re
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

class Transcriber(Agent):
    def __init__(self, *, participant_identity: str, translator: Translator, room: rtc.Room, 
                 from_lang: str, target_lang: str, voice_id: str):
        self.participant_identity = participant_identity
        self.translator = translator
        self.room = room
        self.from_lang = from_lang
        self.target_lang = target_lang
        self.voice_id = voice_id

        self.stt_plugin = deepgram.STT(
            model="nova-2",
            language=self.from_lang,
            smart_format=True
        )

        self.tts_plugin = elevenlabs.TTS(
            model="eleven_multilingual_v2",
            voice_id=self.voice_id
        )

        super().__init__(
            instructions="not-needed",
            stt=self.stt_plugin,
            tts=self.tts_plugin
        )
        
        self.audio_source = rtc.AudioSource(
            self.tts_plugin.sample_rate, 
            self.tts_plugin.num_channels
        )
        
        self.track = rtc.LocalAudioTrack.create_audio_track(
            f"translation_{participant_identity}", 
            self.audio_source
        )
        self._track_published = False

    async def update_agent_metadata(self):
        logger.info(f"Agent settings updated for  {self.participant_identity}: {self.from_lang} -> {self.target_lang}")

        agent_metadata = {
            "is_agent": True,
            "input_lang": self.from_lang,
            "output_lang": self.target_lang,
            "translated_user": self.participant_identity
        }
        
        await self.room.local_participant.set_metadata(json.dumps(agent_metadata))

    async def update_settings(self, new_from_lang: str, new_target_lang: str, new_voice_id: str, new_translator: Translator):
        logger.info(f"Updating settings for {self.participant_identity}: {new_from_lang} -> {new_target_lang}")
        
        self.translator = new_translator
        self.from_lang = new_from_lang
        self.target_lang = new_target_lang

        self.stt_plugin = deepgram.STT(
            model="nova-3-general",
            language=self.from_lang,
            smart_format=True
        )
        self._stt = self.stt_plugin 

        if self.voice_id != new_voice_id:
            self.voice_id = new_voice_id
            self.tts_plugin = elevenlabs.TTS(
                model="eleven_multilingual_v2",
                voice_id=self.voice_id
            )
            self._tts = self.tts_plugin
        
        await self.update_agent_metadata()

    async def on_user_turn_completed(self, _, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        if not user_transcript.strip():
            return
            
        try:
            raw_identity = self.participant_identity
            clean_name = re.sub(r'_{2,}[a-zA-Z0-9]+$', '', raw_identity)
            payload = {
                "message": user_transcript,
                "timestamp": int(time.time() * 1000),
                "id": f"transcript-{raw_identity}-{time.time()}",
                "from": {
                    "identity": raw_identity,
                    "name": clean_name,
                    "isLocal": False
                }
            }

            data = json.dumps(payload).encode("utf-8")

            await self.room.local_participant.publish_data(
                payload=data,
                reliable=True,
                topic="transcription_data"
            )

            if self.target_lang and self.target_lang != "no-translate":
                translated_text = self.translator.translate(user_transcript)
                
                logger.info(f"[{self.participant_identity}] ({self.from_lang}->{self.target_lang}) {user_transcript} -> {translated_text}")

                if not self._track_published:
                    await self.room.local_participant.publish_track(self.track)
                    self._track_published = True

                async for synthesized in self.tts_plugin.synthesize(translated_text):
                    await self.audio_source.capture_frame(synthesized.frame)
            
        except Exception as e:
            logger.error(f"Translation/Speech error for {self.participant_identity}: {e}")

        raise StopResponse()


class MultiUserTranscriber:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: set[asyncio.Task] = set()
        self._agents: dict[str, Transcriber] = {}
        self._user_from_lang: dict[str, str] = {}
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
        await asyncio.gather(*[self._close_session(session) for session in self._sessions.values()])

    def _update_settings(self, participant: rtc.RemoteParticipant):
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
                logger.warning(f"Metadata parse error for {participant.identity}: {e}")

        new_translator = Translator(to_lang=target_lang, from_lang=from_lang)
        self._user_from_lang[participant.identity] = from_lang
        self._user_target_lang[participant.identity] = target_lang
        self._user_voice[participant.identity] = voice_id

        if participant.identity in self._agents:
            agent = self._agents[participant.identity]
            asyncio.create_task(agent.update_settings(from_lang, target_lang, voice_id, new_translator))

    def on_metadata_changed(self, participant: rtc.RemoteParticipant, _):
        self._update_settings(participant)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        if participant.identity in self._sessions or participant.identity.startswith("agent-"):
            return

        self._update_settings(participant)
        task = asyncio.create_task(self._start_session(participant))
        self._tasks.add(task)

        def on_task_done(t: asyncio.Task):
            try:
                self._sessions[participant.identity] = t.result()
            except Exception as e:
                logger.error(f"Session failed for {participant.identity}: {e}")
            finally:
                self._tasks.discard(t)

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        self._user_from_lang.pop(participant.identity, None)
        self._user_target_lang.pop(participant.identity, None)
        self._user_voice.pop(participant.identity, None)
        self._agents.pop(participant.identity, None)
        
        if (session := self._sessions.pop(participant.identity, None)) is None:
            return
        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:

        session = AgentSession()
        
        agent = Transcriber(
            participant_identity=participant.identity,
            translator=Translator(
                to_lang=self._user_target_lang[participant.identity], 
                from_lang=self._user_from_lang[participant.identity]
            ),
            room=self.ctx.room,
            from_lang=self._user_from_lang[participant.identity],
            target_lang=self._user_target_lang[participant.identity],
            voice_id=self._user_voice.get(participant.identity, DEFAULT_VOICE_ID)
        )

        self._agents[participant.identity] = agent

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
        await room_io.start()
        await session.start(agent=agent)
        await agent.update_agent_metadata()
        return session

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.aclose()

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