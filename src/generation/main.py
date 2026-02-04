import os
import argparse
import warnings
from loguru import logger
from .models import get_tts_model, get_vc_model, list_tts_models, list_vc_models
from .raw_subsets import get_raw_subset, list_raw_subsets
from datetime import datetime
from .models.tts.src_cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
from vllm import ModelRegistry
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


def main():
    parser = argparse.ArgumentParser(description="Generate audio data")
    parser.add_argument("-s", "--subset", type=str, nargs="+", default=["podcast"], help="Name of the subset", choices=list_raw_subsets())
    parser.add_argument("-t", "--tts_model", type=str, nargs="+", default=["xttsv2"], help="Name of the TTS model", choices=list_tts_models())
    parser.add_argument("-d", "--tts_device", type=int, nargs="+", default=None, help="Device for the TTS model (for CUDA: 0, 1, 2, ...)")
    parser.add_argument("-v", "--vc_model", type=str, nargs="+", default=[], help="Name of the VC model", choices=list_vc_models())
    parser.add_argument("-p", "--partition", type=str, default="en", help="Partition of the subset")
    parser.add_argument("--data_dir", type=str, default="data/QuadVoxBench", help="Directory for dataset")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if len(args.tts_model) != len(args.tts_device):
        raise ValueError("Number of TTS models and devices must be the same")
    
    os.makedirs("logs", exist_ok=True)
    log_id = logger.add("logs/generation.log", rotation="20 MB", retention="60 days")
    start_time = datetime.now()
    logger.info(f"Generating fake audio data for subsets: {args.subset} with TTS models: {args.tts_model} and VC models: {args.vc_model}")
    logger.remove(log_id)

    tts_models = [get_tts_model(model)(device=f"cuda:{device}") for model, device in zip(args.tts_model, args.tts_device)]
    vc_models = [get_vc_model(model)() for model in args.vc_model]
    for subset in args.subset:
        logger.info(f"Generating fake audio data for {subset} ...")
        raw_subset = get_raw_subset(subset)(**vars(args))
        raw_subset.generate(tts_models=tts_models, vc_models=vc_models)
        logger.info(f"Generation for {subset} completed")

    end_time = datetime.now()
    logger.add("logs/generation.log", rotation="20 MB", retention="60 days")
    logger.info(f"Generation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} completed in {end_time - start_time}")

if __name__ == "__main__":
    main()