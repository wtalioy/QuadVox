from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import random
from loguru import logger

from ..baselines import get_baseline_model
from ..subsets import get_subset_model
from . import CrossLanguageTestConfig, TestConfig

__all__ = ["TestRunner"]

class TestRunner:
    def __init__(self, data_dir: str, baselines: List[str], device: str = "cuda"):
        self.data_dir = data_dir
        self.baseline_names = baselines if isinstance(baselines, list) else [baselines]
        self.device = device
        self.results: dict[str, Any] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._subset_cache: Dict[Tuple[str, str | None], Any] = {}

        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        log_file = f"logs/test_{self.timestamp}.log"
        logger.add(log_file, rotation="20 MB", retention="30 days")

    def _create_combined_subset(
        self, subset_names: List[str], split: str, subset: str | None = None, shuffle: bool = True, limit: int | None = None
    ) -> Tuple[List[Any], List[Any]]:
        """Combine multiple subsets for a specific split, using a cache."""

        def _get_subset(name: str, subset: str | None):
            key = (name, subset)
            if key not in self._subset_cache:
                kwargs = {"data_dir": self.data_dir}
                if name == "phonecall" and subset is not None:
                    kwargs["subset"] = subset
                self._subset_cache[key] = get_subset_model(name)(**kwargs)
            return self._subset_cache[key]

        combined_data: List[Any] = []
        combined_labels: List[Any] = []

        for subset_name in subset_names:
            subset = _get_subset(subset_name, subset if subset_name == "phonecall" else None)
            split_data = subset.data.get(split, [])
            if not split_data:
                continue

            combined_data.extend(split_data)
            combined_labels.extend(subset.labels[split])
            logger.info(f"Added {len(split_data)} samples from {subset_name} {split}")

        if shuffle:
            random.seed(34)
            indices = list(range(len(combined_data)))
            random.shuffle(indices)
            combined_data = [combined_data[i] for i in indices]
            combined_labels = [combined_labels[i] for i in indices]
        if limit is not None and len(combined_data) > limit:
            combined_data = combined_data[:limit]
            combined_labels = combined_labels[:limit]

        logger.info(f"Combined {split}: {len(combined_data)} total samples from {len(subset_names)} subsets")
        return combined_data, combined_labels

    def _train_model(
        self, train_subsets: List[str], val_subsets: List[str], expr_name: str, subset: str | None, baseline_name: str
    ) -> Any:
        """Train a model on specified subsets."""
        logger.info(f"Starting training for test {expr_name}")
        logger.info(f"Training {baseline_name} on {train_subsets}")

        train_data, train_labels = self._create_combined_subset(train_subsets, "train", subset)
        val_data, val_labels = self._create_combined_subset(val_subsets, "dev", subset, shuffle=False)

        if not train_data:
            logger.warning(f"No training data found for {train_subsets}")
            return None

        baseline = get_baseline_model(baseline_name)(device=self.device)
        ref_data, ref_labels = None, None
        if baseline.name == "RAPT":
            ref_num = baseline.ref_num * 2
            if len(train_data) >= ref_num:
                ref_data, ref_labels = train_data[:ref_num], train_labels[:ref_num]
                train_data, train_labels = train_data[ref_num:], train_labels[ref_num:]
                logger.info(f"Extracted {ref_num} reference samples for RAPT")
            else:
                logger.warning(f"Not enough training samples for RAPT reference (need {ref_num}, got {len(train_data)})")

        baseline.train(
            train_data=train_data,
            train_labels=train_labels,
            eval_data=val_data,
            eval_labels=val_labels,
            ref_data=ref_data,
            ref_labels=ref_labels,
            dataset_name=expr_name,
            sr=16000,
        )
        return baseline

    def _evaluate_model(
        self, baseline: Any, test_subsets: List[str], expr_name: str, subset: str | None, baseline_name: str
    ) -> Dict[str, float]:
        """Evaluate a trained model on test datasets."""
        if baseline is None:
            return {"eer": float("inf")}

        logger.info(f"Start evaluation for test {expr_name}")
        logger.info(f"Evaluating {baseline_name} on {test_subsets}")

        if len(test_subsets) == 1 and test_subsets[0] in ["partialfake", "noisyspeech"]:
            subset = get_subset_model(test_subsets[0])(data_dir=self.data_dir)
            return subset.evaluate(baseline, ["eer"], in_domain=True, expr_name=expr_name)

        test_data, test_labels = self._create_combined_subset(test_subsets, "test", subset, shuffle=False)
        if not test_data:
            logger.warning(f"No test data found for {test_subsets}")
            return {"eer": float("inf")}

        return baseline.evaluate(
            data=test_data, labels=test_labels, metrics=["eer"], sr=16000, in_domain=True, dataset_name=expr_name
        )

    def _run_test(self, name: str, config: TestConfig) -> dict[str, Any]:
        """Run a single test described by *config* for all baselines."""
        logger.info(f"Running {name} with baselines: {', '.join(self.baseline_names)}")
        test_results: dict[str, Any] = {}

        for baseline_name in self.baseline_names:
            logger.info(f"  -> Baseline: {baseline_name}")
            model = self._train_model(
                train_subsets=config.train_subsets,
                val_subsets=config.val_subsets,
                expr_name=name,
                subset=config.subset,
                baseline_name=baseline_name,
            )

            baseline_results: dict[str, Any] = {
                test_name: self._evaluate_model(
                    baseline=model,
                    test_subsets=subsets,
                    expr_name=name,
                    subset=config.subset,
                    baseline_name=baseline_name,
                )
                for test_name, subsets in config.test_subsets.items()
            }
            test_results[baseline_name] = baseline_results

        self.results[name] = test_results
        self.save_results()
        return test_results

    def _run_cross_language_test(self, name: str, config: CrossLanguageTestConfig) -> Dict[str, Any]:
        """Run a test that trains separate models per language."""
        logger.info(f"Running cross-language test {name}")
        exp_results: dict[str, Any] = {}

        for baseline_name in self.baseline_names:
            logger.info(f"  -> Baseline: {baseline_name}")

            language_models = {
                lang_cfg.name: self._train_model(
                    train_subsets=lang_cfg.train_subsets,
                    val_subsets=lang_cfg.val_subsets,
                    expr_name=f"{name}_{lang_cfg.name}",
                    subset=lang_cfg.subset,
                    baseline_name=baseline_name,
                )
                for lang_cfg in config.languages
            }

            baseline_results: dict[str, Any] = {}
            for model_lang, model in language_models.items():
                results_for_model = {
                    test_lang: self._evaluate_model(
                        baseline=model,
                        test_subsets=subsets,
                        expr_name=f"{name}_{model_lang}",
                        subset=next((lc.subset for lc in config.languages if lc.name == test_lang), None),
                        baseline_name=baseline_name,
                    )
                    for test_lang, subsets in config.test_subsets.items()
                }
                baseline_results[f"model_{model_lang}"] = results_for_model
            exp_results[baseline_name] = baseline_results

        self.results[name] = exp_results
        self.save_results()
        return exp_results

    def save_results(self):
        """Save results to a JSON file."""
        results_file = f"results/test_{self.timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
