from typing import List
import os
import json
from ..models import BaseTTS, BaseVC
from .base import BaseRawSubset
from loguru import logger
from tqdm import tqdm
import soundfile as sf


class Emotional(BaseRawSubset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "Emotional"), *args, **kwargs)

    def _try_generate_audio(self, item, tts_model, vc_model=None, language: str = "en", **kwargs):
        text = item['text']
        audio_rel_path = item['audio']['real']
        audio_path = os.path.join(self.data_dir, audio_rel_path)
        
        emotion = item['emotion'].lower()
        model_name = tts_model.model_name
        output_rel_path = audio_rel_path.replace("real", model_name)
        output_path = os.path.join(self.data_dir, output_rel_path)

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fake_audio, sample_rate = tts_model.infer_emotion(
                text=text,
                emotion=emotion,
                prompt_wav=audio_path,
                **kwargs
            )
            
            sf.write(output_path, fake_audio, sample_rate)
            logger.info(f"Generated audio at {output_path}")
            item['audio']['fake'][model_name] = output_rel_path
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate audio for text: {text[:50]}... with {model_name}")
            logger.error(f"Emotion: {emotion}, Error: {e}")
            return False

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], language: str = "en", *args, **kwargs):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        for tts_model in tts_models:
            model_name = tts_model.model_name
            logger.info(f"Starting generation with {model_name}...")
            os.makedirs(os.path.join(self.data_dir, "audio", model_name), exist_ok=True)
            
            failed_items = []
            for idx, item in enumerate(tqdm(meta_data, desc=f"Generating with {model_name}")):
                if not self._try_generate_audio(item, tts_model, **kwargs):
                    failed_items.append(item)
                    logger.warning(f"Failed to generate audio for item {idx+1} with {model_name}: {item['text'][:50]}...")
            
            logger.info(f"{model_name} completed: {len(meta_data) - len(failed_items)}/{len(meta_data)} items successful")
            if failed_items:
                logger.warning(f"{model_name} failed items: {len(failed_items)}")
            
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"All models completed. Meta data saved to {self.meta_path}")
