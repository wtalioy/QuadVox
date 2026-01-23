import os
import argparse
import warnings
import torch
from loguru import logger
from .baselines import get_baseline_model, list_baseline_models
from .subsets import get_subset_model, list_subset_models

def display_results(results: dict, baseline: str, dataset: str):
    if isinstance(list(results.values())[0], dict): # PartialFake and NoisySpeech specific
        for source_name, source_results in results.items():
            logger.info("Evaluation results:")
            for metric, value in source_results.items():
                logger.info(f"({baseline} on {dataset} from {source_name}) {metric}: {value:.4f}")
    else:
        logger.info("Evaluation results:")
        for metric, value in results.items():
            logger.info(f"({baseline} on {dataset}) {metric}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline on dataset")
    parser.add_argument("-b", "--baseline", type=str, nargs="+", default=["aasist", "aasist-l", "rawnet2", "res-tssdnet", "inc-tssdnet", "rawgat-st", "rapt"], help="Name of the baseline", choices=list_baseline_models())
    parser.add_argument("-s", "--subset", type=str, nargs="+", default=["audiobook", "news", "podcast", "interview", "phonecall", "publicspeech", "publicfigure", "movie", "emotional"], help="Name of the subset", choices=list_subset_models())
    parser.add_argument("-m", "--mode", type=str, default="in", help="Mode of the evaluation", choices=["cross", "in"])
    parser.add_argument("--train_only", action="store_true", help="Train the baseline only")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate the baseline only")
    parser.add_argument("--metric", type=str, nargs="+", default=["eer"], help="Metrics to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/QuadVoxBench", help="Path to the data directory")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/eval.log", rotation="20 MB", retention="60 days")
    
    for subset in args.subset:
        logger.info(f"Preparing {subset} ...")
        subset = get_subset_model(subset)(**vars(args))
        for baseline in args.baseline:
            baseline = get_baseline_model(baseline)(**vars(args))
            logger.info(f"Evaluating {baseline.name} on {subset.name} with metric: {args.metric} in {args.mode}-domain mode")

            results = None
            if args.mode == "cross":
                if not args.train_only:
                    results = subset.evaluate(baseline, args.metric)
            elif args.mode == "in":
                if not args.eval_only:
                    logger.info("Training baseline ...")
                    subset.train(baseline)
                if not args.train_only:
                    logger.info("Evaluating baseline ...")
                    results = subset.evaluate(baseline, args.metric, in_domain=True)
                else:
                    continue

            logger.info(f"Evaluation of {baseline.name} on {subset.name} completed")
            if results is not None:
                display_results(results, baseline.name, subset.name)

    logger.info(f"Evaluation completed")

if __name__ == "__main__":
    main()