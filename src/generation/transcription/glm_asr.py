from typing import List
from transformers import GlmAsrForConditionalGeneration, AutoProcessor

class GLMASR:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
        
        self.model = GlmAsrForConditionalGeneration.from_pretrained(
            "zai-org/GLM-ASR-Nano-2512", 
            dtype="auto", 
            device_map=device
        )

    def transcribe(self, audio_paths: List[str], language: str = "en") -> List[str]:
        inputs = self.processor.apply_transcription_request(audio_paths)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return decoded_outputs