from .base import BaseRawSubset
import os
import json
import random
import librosa
import soundfile as sf
from loguru import logger
import numpy as np
from tqdm import tqdm
from typing import List
from ..models import BaseTTS, BaseVC
from ..transcription.parakeet import Parakeet

class PartialFake(BaseRawSubset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "PartialFake"), *args, **kwargs)
        self.data_sources = ["Interview", "Podcast", "PublicSpeech"]
        self.sample_rate = 16000
        self.ratio = 0.3
        if not os.path.exists(self.meta_path):
            meta_data = self._create_meta(base_dir=data_dir)

    def _create_meta(self, base_dir) -> dict:
        os.makedirs(self.data_dir, exist_ok=True)
        model = Parakeet()
        meta_data = {}
        for data_source in self.data_sources:
            with open(os.path.join(base_dir, data_source, "meta_test.json"), "r") as f:
                domain_meta_data = json.load(f)

            random.shuffle(domain_meta_data)
            domain_meta_data = domain_meta_data[:int(len(domain_meta_data) * self.ratio)]

            for i in tqdm(range(0, len(domain_meta_data), 32), desc=f"Processing {data_source} source"):
                batch = domain_meta_data[i:i+32]
                audio_paths = [os.path.join("data", data_source, item["audio"]["real"]) for item in batch]
                word_timestamps = model.get_word_timestamps(audio_paths)
                for j, item in enumerate(batch):
                    for key in list(item.keys()):
                        if key not in ["text", "audio"]:
                            del item[key]
                    item["word_timestamps"] = word_timestamps[j]
                    item["audio"]["fake"] = {}
            
            meta_data[data_source] = domain_meta_data

        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        return meta_data

    def _select_partial_fake(self, item, source_name: str, min_word_num=2, max_word_num=4) -> tuple[str, np.ndarray, np.ndarray]:
        word_timestamps = item["word_timestamps"]
        num_words = random.randint(min_word_num, max_word_num)
        
        # Choose a random starting index that allows for words consecutive words
        max_start_index = len(word_timestamps) - num_words
        start_index = random.randint(0, max_start_index)

        text = ' '.join([word_item['word'] for word_item in word_timestamps[start_index:start_index + num_words]])    

        array, sample_rate = librosa.load(os.path.join("data", source_name, item['audio']['real']), sr=None)
        start_time = word_timestamps[start_index]['start'] * sample_rate
        end_time = word_timestamps[start_index + num_words - 1]['end'] * sample_rate
        org_head = array[:int(start_time)]
        org_tail = array[int(end_time):]
        org_head = librosa.resample(org_head, orig_sr=sample_rate, target_sr=self.sample_rate)
        org_tail = librosa.resample(org_tail, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        return text, org_head, org_tail

    def _try_generate_audio(self, item, tts_model, vc_model=None, idx: int = 0, source_name: str = "Podcast", language: str = "en", **kwargs):
        """Try to generate audio with given models. Returns True if successful, False otherwise."""
        text = item['text']
        audio_rel_path = item['audio']['real']
        output_rel_path = f"audio/{source_name.lower()}/{idx}.wav"
        audio_path = os.path.join("data", source_name, audio_rel_path)
        output_path = os.path.join(self.data_dir, output_rel_path)
        item['audio']['fake'] = {}
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            text, org_head, org_tail = self._select_partial_fake(item, source_name=source_name)
            
            if vc_model is None:
                # TTS only
                fake_audio, sample_rate = tts_model.infer(text, prompt_wav=audio_path, language=language, **kwargs)
                fake_audio = librosa.resample(fake_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                model_name = tts_model.model_name
            else:
                # TTS + VC
                fake_audio, sample_rate = tts_model.infer(text, prompt_wav=audio_path, language=language, **kwargs)

                # Convert the voice of the corresponding sample to that of the real audio
                sample_path = f"src/generation/samples/{language}.wav"
                temp_vc_path = output_path.replace(".wav", "_temp_vc.wav")
                vc_model.convert(sample_path, audio_path, temp_vc_path)
                
                # Save TTS audio as temporary file
                temp_tts_path = output_path.replace(".wav", "_temp_tts.wav")
                sf.write(temp_tts_path, fake_audio, sample_rate)
                
                # Then perform voice conversion with VC
                vc_model.convert(temp_tts_path, temp_vc_path, output_path)
                fake_audio = librosa.resample(fake_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                
                # Clean up temporary file
                if os.path.exists(temp_tts_path):
                    os.remove(temp_tts_path)
                if os.path.exists(temp_vc_path):
                    os.remove(temp_vc_path)
                    
                model_name = f"{tts_model.model_name} + {vc_model.model_name}"

            fake_audio = np.concatenate([org_head, fake_audio, org_tail])
            sf.write(output_path, fake_audio, self.sample_rate)
            
            logger.info(f"Generated audio at {output_path}")
            item['audio']['fake'][model_name] = output_rel_path
            item['partial_text'] = text
            del item['word_timestamps']
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate audio for text: {text} with {tts_model.model_name}" + 
                        (f" + {vc_model.model_name}" if vc_model else ""))
            logger.error(e)
            return False

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], language: str = "en", alternate: bool = True, *args, **kwargs):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        # Create all possible model combinations
        all_combinations = []
        
        # TTS-only combinations
        tts_only_models = [model for model in tts_models if not model.require_vc]
        for tts_model in tts_only_models:
            all_combinations.append((tts_model, None))
        
        # TTS+VC combinations
        tts_vc_models = [model for model in tts_models if model.require_vc]
        if vc_models and len(vc_models) > 0:
            for tts_model in tts_vc_models:
                for vc_model in vc_models:
                    all_combinations.append((tts_model, vc_model))
        else:
            # If no VC models provided, treat TTS+VC models as TTS-only
            for tts_model in tts_vc_models:
                all_combinations.append((tts_model, None))
        
        if not all_combinations:
            logger.warning("No valid model combinations found")
            return
        
        logger.info(f"Available combinations: {len(all_combinations)}")
        for i, (tts, vc) in enumerate(all_combinations):
            combo_name = tts.model_name + (f" + {vc.model_name}" if vc else "")
            logger.info(f"  {i+1}. {combo_name}")
        
        # Process each item with round-robin distribution and fallback logic
        failed_items = []
        for source_name, items in meta_data.items():
            for idx, item in enumerate(tqdm(items, desc=f"Generating partial fake audio for {source_name}")):
                if alternate:
                    success = False
                    start_combo_idx = idx % len(all_combinations)
                    for i in range(len(all_combinations)):
                        combo_idx = (start_combo_idx + i) % len(all_combinations)
                        tts_model, vc_model = all_combinations[combo_idx]
                        combo_name = tts_model.model_name + (f" + {vc_model.model_name}" if vc_model else "")
                        if i == 0:
                            logger.info(f"Processing item {idx+1} with assigned combination: {combo_name}")
                        else:
                            logger.info(f"Trying fallback combination {i+1}: {combo_name}")
                        
                        if self._try_generate_audio(item, tts_model, vc_model, idx=idx, source_name=source_name, language=language, **kwargs):
                            success = True
                            break
                    
                    if not success:
                        logger.warning(f"Failed to generate audio for item {idx+1} with all combinations: {item['text'][:50]}...")
                        failed_items.append(item)
                else:
                    for tts_model, vc_model in all_combinations:
                        if not self._try_generate_audio(item, tts_model, vc_model, idx=idx, source_name=source_name, language=language, **kwargs):
                            raise RuntimeError(f"Failed to generate audio for item {idx+1} with combination: {tts_model.model_name} + {vc_model.model_name}")

        # Save updated metadata
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Meta data updated and saved to {self.meta_path}")
        logger.info(f"Successfully processed: {len(meta_data) - len(failed_items)}/{len(meta_data)} items")
        if failed_items:
            logger.warning(f"Failed items: {len(failed_items)}")