from typing import Optional

from .src_voxcpm.core import VoxCPM as VoxCPMModel
from .base import BaseTTS


class VoxCPMTTS(BaseTTS):
    def __init__(
        self,
        model="openbmb/VoxCPM1.5",
        *args,
        **kwargs
    ):
        self.model_name = "VoxCPM"
        self.require_vc = False
        
        self.model = VoxCPMModel.from_pretrained(
            hf_model_id=model,
            cache_dir="cache",
            optimize=True,
        )

        self.sample_rate = self.model.tts_model.sample_rate
    
    def infer(
        self,
        text: str,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        **kwargs
    ):
        if prompt_wav is not None and prompt_text is None:
            raise ValueError("prompt_text is required when prompt_wav is provided")
        
        cfg_value = kwargs.get("cfg_value", 2.0)
        inference_timesteps = kwargs.get("inference_timesteps", 10)
        normalize = kwargs.get("normalize", False)
        retry_badcase = kwargs.get("retry_badcase", True)
        
        audio = self.model.generate(
            text=text,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            retry_badcase=retry_badcase,
        )
        
        if audio is None or len(audio) == 0:
            raise RuntimeError("No audio generated")
        
        return audio, self.sample_rate
