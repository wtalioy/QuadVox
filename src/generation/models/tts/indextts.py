from .src_indextts.infer import IndexTTS2
from .base import BaseTTS

class IndexTTS(BaseTTS):
    def __init__(self, model_dir="cache/IndexTTS-2", *args, **kwargs):
        self.model_name = "IndexTTS2"
        self.require_vc = False
        self.model = IndexTTS2(model_dir=model_dir)
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
        return self.model.infer(text=text, spk_audio_prompt=prompt_wav)

    def infer_emotion(self, text: str, emotion: str, prompt_wav: str, **kwargs):
        if emotion in self.emotion_ids:
            emotion_id = self.emotion_ids[emotion]
        else:
            raise ValueError(f"Emotion {emotion} not supported")
        emo_vector = [0.0] * len(self.emotion_ids)
        emo_vector[emotion_id] = 0.45
        return self.model.infer(text=text, spk_audio_prompt=prompt_wav, emo_vector=emo_vector)