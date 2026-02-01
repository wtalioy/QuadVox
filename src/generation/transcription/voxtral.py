from typing import List
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import json

class Voxtral:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = VoxtralForConditionalGeneration.from_pretrained("mistralai/Voxtral-Mini-3B-2507", torch_dtype=torch.bfloat16, device_map=device)
        self.processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

    def transcribe(self, audio_paths: List[str], language: str = "en") -> List[str]:
        inputs = self.processor.apply_transcription_request(language=language, audio=audio_paths, model_id="mistralai/Voxtral-Mini-3B-2507")
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded_outputs

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tqdm import tqdm
    import os
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/QuadVoxBench/Podcast")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    model = Voxtral()
    with open(os.path.join(args.data_dir, "meta.json"), "r") as f:
        meta_data = json.load(f)
    for i in tqdm(range(0, len(meta_data), args.batch_size)):
        batch = meta_data[i:i+args.batch_size]
        audio_paths = [os.path.join(args.data_dir, item["audio"]["real"]) for item in batch]
        texts = model.transcribe(audio_paths, args.language)
        for j, item in enumerate(batch):
            item["text"] = texts[j]
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)