from typing import List
from nemo.collections.asr.models import ASRModel

class Parakeet:
    def __init__(self, device: str = "cuda"):
        self.model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)

    def transcribe(self, audio_paths: List[str], language: str = "en") -> List[str]:
        return self.model.transcribe(audio_paths)[0]

    def get_word_timestamps(self, audio_paths: List[str]) -> List[str]:
        outputs = self.model.transcribe(audio_paths, timestamps=True)
        return [output[0].timestep['word'] for output in outputs]