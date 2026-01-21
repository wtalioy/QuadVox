import numpy as np
from TTS.api import TTS
from .base import BaseTTS

class XTTSv2(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "XTTSv2"
        self.require_vc = False
        self.model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.model = TTS(model_name=self.model_id).cuda()
        assert self.model.synthesizer is not None, "TTS synthesizer is not initialized."
        self.sample_rate = int(self.model.synthesizer.output_sample_rate)

    def infer(self, text: str, prompt_wav: str, language="en", **kwargs):
        wav = self.model.tts(text=text, speaker_wav=prompt_wav, language=language)
        return np.array(wav), self.sample_rate