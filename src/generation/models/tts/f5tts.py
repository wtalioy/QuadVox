import os
from omegaconf import OmegaConf
from cached_path import cached_path

from .src_f5tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    device as default_device,
)
from .base import BaseTTS


class F5TTS(BaseTTS):
    def __init__(
        self,
        model="F5TTS_v1_Base",
        device=None,
        vocoder_name="vocos",
        *args,
        **kwargs
    ):
        self.model_name = "F5TTS"
        self.require_vc = False
        
        if device is None:
            self.device = default_device
        else:
            self.device = device
        
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(_current_dir, "src_f5tts", "configs")
        model_cfg = os.path.join(configs_dir, f"{model}.yaml")
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Model config not found: {model_cfg}")
        model_cfg_obj = OmegaConf.load(model_cfg)
        backbone_name = model_cfg_obj.model.backbone
        if backbone_name == "Dit" or backbone_name == "DiT":
            from .src_f5tts.model.backbones.dit import DiT as model_cls
        else:
            print(f"Warning: Backbone {backbone_name} not explicitly supported, using DiT")
            from .src_f5tts.model.backbones.dit import DiT as model_cls
        model_arc = model_cfg_obj.model.arch
        
        self.mel_spec_type = model_cfg_obj.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg_obj.model.mel_spec.target_sample_rate
        
        vocab_file = os.path.join(configs_dir, "vocab.txt")

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
        if model == "F5TTS_Base":
            if vocoder_name == "vocos":
                ckpt_step = 1200000
            elif vocoder_name == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000
        
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
        
        self.vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            device=self.device,
            hf_cache_dir="cache"
        )
        
        self.model = load_model(
            model_cls=model_cls,
            model_cfg=model_arc,
            ckpt_path=ckpt_file,
            mel_spec_type=vocoder_name,
            vocab_file=vocab_file,
            device=self.device,
        )
    
    def infer(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str = "",
        **kwargs
    ):
        ref_audio_processed, ref_text = preprocess_ref_audio_text(
            prompt_wav,
            prompt_text,
            show_info=lambda x: None
        )
        
        nfe_step = kwargs.get("nfe_step", 32)
        cfg_strength = kwargs.get("cfg_strength", 2.0)
        speed = kwargs.get("speed", 1.0)
        target_rms = kwargs.get("target_rms", 0.1)
        cross_fade_duration = kwargs.get("cross_fade_duration", 0.15)
        
        audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text,
            gen_text=text,
            model_obj=self.model,
            vocoder=self.vocoder,
            mel_spec_type=self.mel_spec_type,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            speed=speed,
            device=self.device,
            show_info=lambda x: None,
            progress=None,
        )
        
        if audio is None:
            raise RuntimeError("No audio generated")
        
        return audio, sample_rate
