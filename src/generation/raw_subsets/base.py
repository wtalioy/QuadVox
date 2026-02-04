from typing import List
from ..models import BaseTTS, BaseVC
import os
import json
from loguru import logger
from tqdm import tqdm
import soundfile as sf

class BaseRawSubset:
    def __init__(self, data_dir: str, *args, **kwargs):
        self.data_dir = data_dir
        self.meta_path = os.path.join(self.data_dir, "meta.json")

        from datetime import datetime
        logger.add(f"logs/generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", rotation="100 MB", retention="60 days")

    def _try_generate_audio(self, item, tts_model, vc_model=None, language: str = "en", **kwargs):
        """Try to generate audio with given models. Returns True if successful, False otherwise."""
        text = item['text']
        audio_rel_path = item['audio']['real']
        output_rel_path = audio_rel_path.replace("audio/real", f"audio/{tts_model.model_name}" + (f"+{vc_model.model_name}" if vc_model else ""))
        audio_path = os.path.join(self.data_dir, audio_rel_path)
        output_path = os.path.join(self.data_dir, output_rel_path)
        if "fake" not in item["audio"] or not isinstance(item["audio"]["fake"], dict):
            item["audio"]["fake"] = {}
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if vc_model is None:
                # TTS only
                fake_audio, sample_rate = tts_model.infer(text, prompt_wav=audio_path, prompt_text=text, language=language, **kwargs)
                sf.write(output_path, fake_audio, sample_rate)
                model_name = tts_model.model_name
            else:
                # TTS + VC
                fake_audio, sample_rate = tts_model.infer(text, language=language, **kwargs)

                # Convert the voice of the corresponding sample to that of the real audio
                sample_path = kwargs.get("vc_source_sample") or f"src/generation/samples/{language}.wav"
                if not os.path.exists(sample_path):
                    fallback_sample = "src/generation/samples/en.wav"
                    logger.warning(
                        f"VC source sample not found at {sample_path}; "
                        f"falling back to {fallback_sample}"
                    )
                    sample_path = fallback_sample
                temp_vc_path = output_path.replace(".wav", "_temp_vc.wav")
                vc_model.convert(sample_path, audio_path, temp_vc_path)
                
                # Save TTS audio as temporary file
                temp_tts_path = output_path.replace(".wav", "_temp_tts.wav")
                sf.write(temp_tts_path, fake_audio, sample_rate)
                
                # Then perform voice conversion with VC
                vc_model.convert(temp_tts_path, temp_vc_path, output_path)
                
                # Clean up temporary file
                if os.path.exists(temp_tts_path):
                    os.remove(temp_tts_path)
                if os.path.exists(temp_vc_path):
                    os.remove(temp_vc_path)
                    
                model_name = f"{tts_model.model_name} + {vc_model.model_name}"
            
            logger.info(f"Generated audio at {output_path}")
            item['audio']['fake'][model_name] = output_rel_path
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate audio for text: {text} with {tts_model.model_name}" + 
                        (f" + {vc_model.model_name}" if vc_model else ""))
            logger.error(e)
            return False

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], language: str = "en", alternate: bool = True, *args, **kwargs):
        output_dir = os.path.join(self.data_dir, "audio/fake")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        for idx, item in enumerate(tqdm(meta_data, desc="Generating audio", leave=False)):
            # if "fake" in item["audio"] and len(item["audio"]["fake"]) > 0:
            #     continue

            if alternate:
                success = False
                start_combo_idx = idx % len(all_combinations)
                
                # Try combinations starting from the assigned one, then cycle through others
                for i in range(len(all_combinations)):
                    combo_idx = (start_combo_idx + i) % len(all_combinations)
                    tts_model, vc_model = all_combinations[combo_idx]     
                    combo_name = tts_model.model_name + (f" + {vc_model.model_name}" if vc_model else "")
                    if i == 0:
                        logger.info(f"Processing item {idx+1} with assigned combination: {combo_name}")
                    else:
                        logger.info(f"Trying fallback combination {i+1}: {combo_name}")
                    
                    if self._try_generate_audio(item, tts_model, vc_model, language=language, **kwargs):
                        success = True
                        break
                
                if not success:
                    logger.warning(f"Failed to generate audio for item {idx+1} with all combinations: {item['text'][:50]}...")
                    failed_items.append(item)
            else:
                for tts_model, vc_model in all_combinations:
                    if not self._try_generate_audio(item, tts_model, vc_model, language=language, **kwargs):
                        raise RuntimeError(f"Failed to generate audio for item {idx+1} with combination: {tts_model.model_name} + {vc_model.model_name}")

        # Save updated metadata
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Meta data updated and saved to {self.meta_path}")
        logger.info(f"Successfully processed: {len(meta_data) - len(failed_items)}/{len(meta_data)} items")
        if failed_items:
            logger.warning(f"Failed items: {len(failed_items)}")