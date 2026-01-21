import os
import argparse
from loguru import logger
from .models import get_tts_model, get_vc_model, list_tts_models, list_vc_models
from .raw_subsets import get_raw_subset, list_raw_subsets
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Generate audio data")
    parser.add_argument("-s", "--subset", type=str, nargs="+", default=["partialfake"], help="Name of the subset", choices=list_raw_subsets())
    parser.add_argument("-t", "--tts_model", type=str, nargs="+", default=["gpt4omini", "xttsv2", "melotts", "bark", "yourtts"], help="Name of the TTS model", choices=list_tts_models())
    parser.add_argument("-v", "--vc_model", type=str, nargs="+", default=["openvoice"], help="Name of the VC model", choices=list_vc_models())
    parser.add_argument("-p", "--partition", type=str, default="zh-cn", help="Partition of the subset")
    parser.add_argument("--data_dir", type=str, default="data/QuadVoxBench", help="Directory for dataset")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    log_id = logger.add("logs/generation.log", rotation="20 MB", retention="60 days")
    start_time = datetime.now()
    logger.info(f"Generating fake audio data for subsets: {args.subset} with TTS models: {args.tts_model} and VC models: {args.vc_model}")
    logger.remove(log_id)

    tts_models = [get_tts_model(model)() for model in args.tts_model]
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