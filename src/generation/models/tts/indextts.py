from .src_indextts.infer_v2 import IndexTTS2 as IndexTTS2Model
from .src_indextts.infer import IndexTTS as IndexTTSModel
from .base import BaseTTS

class IndexTTS2(BaseTTS):
    def __init__(self, model_dir="cache/IndexTTS-2", *args, **kwargs):
        self.model_name = "IndexTTS2"
        self.require_vc = False
        self.model = IndexTTS2Model(model_dir=model_dir)
        self.sample_rate = 22050
        self.emotion_ids = {
            "happy": 0,
            "angry": 1,
            "sad": 2,
            "afraid": 3,
            "disgusted": 4,
            "melancholic": 5,
            "surprised": 6,
            "calm": 7
        }

    def infer(self, text: str, prompt_wav: str, language="en", **kwargs):
        audio = self.model.infer(text=text, spk_audio_prompt=prompt_wav)
        if audio is None:
            raise RuntimeError("No audio generated")
        return audio, self.sample_rate

    def infer_emotion(self, text: str, emotion: str, prompt_wav: str, **kwargs):
        if emotion in self.emotion_ids:
            emotion_id = self.emotion_ids[emotion]
            emo_vector = [0.0] * len(self.emotion_ids)
            emo_vector[emotion_id] = 0.45
        elif emotion == "neutral":
            emo_vector = None
        else:
            raise ValueError(f"Emotion {emotion} not supported")
        audio = self.model.infer(text=text, spk_audio_prompt=prompt_wav, emo_vector=emo_vector)
        if audio is None:
            raise RuntimeError("No audio generated")
        return audio, self.sample_rate


class IndexTTS1_5(BaseTTS):
    def __init__(self, model_dir="cache/IndexTTS-1.5", *args, **kwargs):
        self.model_name = "IndexTTS1.5"
        self.require_vc = False
        self.model = IndexTTSModel(model_dir=model_dir)
        self.sample_rate = 24000

    def infer(self, text: str, prompt_wav: str, language="en", **kwargs):
        audio, sr = self.model.infer(text=text, spk_audio_prompt=prompt_wav)
        if audio is None:
            raise RuntimeError("No audio generated")
        return audio, sr