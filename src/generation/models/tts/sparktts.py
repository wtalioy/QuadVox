import re
import torch
from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from .src_sparktts.utils.file import load_config
from .src_sparktts.models.audio_tokenizer import BiCodecTokenizer
from .src_sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
from .base import BaseTTS


class SparkTTS(BaseTTS):
    def __init__(
        self,
        device=None,
        *args,
        **kwargs
    ):
        self.model_name = "SparkTTS"
        self.require_vc = False
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_dir = "cache/Spark-TTS-0.5B"
        snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=self.model_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(Path(self.model_dir) / "LLM")
        self.model = AutoModelForCausalLM.from_pretrained(Path(self.model_dir) / "LLM")
        self.audio_tokenizer = BiCodecTokenizer(Path(self.model_dir), device=self.device)
        self.model.to(self.device)
        
        self.configs = load_config(Path(self.model_dir) / "config.yaml")
        self.sample_rate = self.configs["sample_rate"]

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ):
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def infer(
        self,
        text: str,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        **kwargs
    ):
        """
        - gender: female | male.
        - pitch: very_low | low | moderate | high | very_high
        - speed: very_low | low | moderate | high | very_high
        - temperature: Sampling temperature for controlling randomness. Default is 0.8.
        - top_k: Top-k sampling parameter. Default is 50.
        - top_p: Top-p (nucleus) sampling parameter. Default is 0.95.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
            global_token_ids = None
        else:
            if prompt_wav is None:
                raise ValueError("prompt_wav is required when gender is not specified")
            prompt, global_token_ids = self.process_prompt(
                text, Path(prompt_wav), prompt_text
            )
            
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        
        return wav, self.sample_rate
