from typing import List
import nemo.collections.asr as nemo_asr

class Parakeet:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

    def transcribe(self, audio_paths: List[str], language: str = "en") -> List[str]:
        return [output.text for output in self.model.transcribe(audio_paths, timestamps=True)]

    def get_word_timestamps(self, audio_paths: List[str]) -> List[str]:
        outputs = self.model.transcribe(audio_paths, timestamps=True)
        return [output.timestamp['word'] for output in outputs]