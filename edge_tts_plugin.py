from livekit import rtc
from livekit.agents import tts, utils
import asyncio
import numpy as np

class EdgeTTS(tts.TTS):
    def __init__(self, *, voice: str = "en-US-AriaNeural"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice

    def synthesize(self, text: str, *, conn_options=None) -> "tts.ChunkedStream":
        return ChunkedStream(self, text, self._voice, conn_options=conn_options or utils.http_context.http_session())


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, tts_instance: EdgeTTS, text: str, voice: str, *, conn_options):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._voice = voice
        self._text = text

    async def _run(self, output_emitter):
        duration = max(0.5, len(self._text) * 0.05)
        num_samples = int(24000 * duration)
        
        audio_data = np.zeros(num_samples, dtype=np.int16)
        
        frame = rtc.AudioFrame(
            data=audio_data.tobytes(),
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=num_samples,
        )
        
        output_emitter.emit(tts.SynthesizedAudio(frame=frame))
        await asyncio.sleep(0.01)
