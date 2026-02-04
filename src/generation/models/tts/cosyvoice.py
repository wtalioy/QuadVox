import sys
sys.path.append('src/generation/models/tts/third_party/Matcha-TTS')
import numpy as np
from typing import Tuple

# Create module alias for cosyvoice imports BEFORE importing anything
import importlib
cosyvoice_module = importlib.import_module('.src_cosyvoice', package=__package__)
sys.modules['cosyvoice'] = cosyvoice_module
for submodule_name in ['llm', 'flow', 'hifigan', 'tokenizer', 'transformer', 'utils', 'dataset']:
    try:
        submodule = importlib.import_module(f'.src_cosyvoice.{submodule_name}', package=__package__)
        sys.modules[f'cosyvoice.{submodule_name}'] = submodule
    except:
        pass

from .src_cosyvoice.cli.cosyvoice import AutoModel
from .base import BaseTTS


class CosyVoice(BaseTTS):
    def __init__(self, model_name, repo_id, *args, **kwargs):
        self.model_name = model_name
        self.require_vc = False
        
        self.model = AutoModel(
            repo_id=repo_id,
            cache_dir="cache/" + repo_id.split("/")[-1],
            fp16=False,
            load_trt=False,
            load_vllm=True,
        )
        
        self.sampling_rate = self.model.sample_rate
        self.emotion_prompts = {
            "happy": "请非常开心地说一句话。",
            "sad": "请非常伤心地说一句话。",
            "angry": "请非常生气地说一句话。"
        }

    def infer(
        self,
        text: str,
        prompt_text: str,
        prompt_wav: str,
        stream: bool = False,
        speed: float = 1.0,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        if self.model_name == "CosyVoice3":
            prompt_text = f"You are a helpful assistant.<|endofprompt|>" + prompt_text
        audio_generator = self.model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed
        )
        
        audio_segments = []
        for output in audio_generator:
            audio_segments.append(output['tts_speech'].numpy())
        
        if len(audio_segments) > 0:
            audio = np.concatenate(audio_segments, axis=1)
            audio = audio.squeeze()
            
            if audio.ndim > 1:
                audio = audio.flatten()
            
            return audio, self.sampling_rate
        else:
            raise RuntimeError("No audio generated")

    def infer_emotion(
        self,
        text: str,
        emotion: str,
        prompt_wav: str,
        stream: bool = False,
        speed: float = 1.0,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        if emotion in self.emotion_prompts:
            prompt_text = self.emotion_prompts[emotion]
        elif emotion == "neutral":
            prompt_text = ""
        else:
            prompt_text = "Speak this sentence with a " + emotion + " tone."
        audio_generator = self.model.inference_instruct2(
            tts_text=text,
            instruct_text=f"You are a helpful assistant. {prompt_text}<|endofprompt|>",
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed
        )
        
        audio_segments = []
        for output in audio_generator:
            audio_segments.append(output['tts_speech'].numpy())
        
        if len(audio_segments) > 0:
            audio = np.concatenate(audio_segments, axis=1)
            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.flatten()
            return audio, self.sampling_rate
        else:
            raise RuntimeError("No audio generated")


class CosyVoice3(CosyVoice):
    def __init__(self, *args, **kwargs):
        super().__init__(model_name="CosyVoice3", repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", *args, **kwargs)

class CosyVoice2(CosyVoice):
    def __init__(self, *args, **kwargs):
        super().__init__(model_name="CosyVoice2", repo_id="FunAudioLLM/CosyVoice2-0.5B", *args, **kwargs)