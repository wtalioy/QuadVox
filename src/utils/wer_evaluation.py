"""
WER (English) / CER (Chinese) evaluation: transcribe real and fake audio, compare to ground truth.

Usage:
  python -m utils.wer_evaluation --subset Interview -d data/QuadVoxBench -o results/wer.json
"""

import os
import json
import argparse
import sys

import numpy as np
from tqdm import tqdm
from loguru import logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from jiwer import wer as jiwer_wer, cer as jiwer_cer

from generation.transcription.parakeet import Parakeet
from generation.transcription.voxtral import Voxtral
from generation.transcription.glm_asr import GLMASR


def get_asr(model_name: str, device: str):
    if model_name == "parakeet":
        return Parakeet(device=device)
    if model_name == "voxtral":
        return Voxtral(device=device)
    if model_name == "glm_asr":
        return GLMASR(device=device)
    raise ValueError(f"Unknown or unavailable ASR model: {model_name}")


def _ensure_text_in_meta(meta_path: str, data_dir: str, asr, language: str, batch_size: int) -> None:
    """If meta has no 'text', transcribe real audio and add it."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if any(item.get("text") for item in meta):
        return
    logger.info("No text in metadata; transcribing real audio...")
    paths = []
    indices_by_path = {}
    for idx, item in enumerate(meta):
        if "audio" not in item or "real" not in item.get("audio", {}):
            continue
        p = os.path.join(data_dir, item["audio"]["real"])
        if os.path.exists(p):
            if p not in indices_by_path:
                paths.append(p)
                indices_by_path[p] = []
            indices_by_path[p].append(idx)
    if not paths:
        return
    transcriptions = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        try:
            transcriptions.extend(asr.transcribe(batch, language))
        except Exception as e:
            logger.warning(f"Transcribe batch failed: {e}")
            transcriptions.extend([""] * len(batch))
    for path, trans in zip(paths, transcriptions):
        if not trans.strip():
            continue
        for idx in indices_by_path[path]:
            meta[idx]["text"] = trans
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def compute_wer(
    meta_path: str,
    data_dir: str,
    asr,
    language: str,
    batch_size: int = 16,
) -> dict:
    _ensure_text_in_meta(meta_path, data_dir, asr, language, batch_size)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    use_cer = language == "zh"
    metric_fn = jiwer_cer if use_cer else jiwer_wer
    metric_name = "CER" if use_cer else "WER"

    real_paths, real_gt = [], []
    fake_paths, fake_gt = [], []
    for item in meta:
        if "audio" not in item:
            continue
        gt = item.get("text") or ""
        if not gt.strip():
            continue
        if "real" in item.get("audio", {}):
            p = os.path.join(data_dir, item["audio"]["real"])
            if os.path.exists(p):
                real_paths.append(p)
                real_gt.append(gt)
        if "fake" in item.get("audio", {}):
            f = item["audio"]["fake"]
            for path in (f.values() if isinstance(f, dict) else [f]):
                p = os.path.join(data_dir, path)
                if os.path.exists(p):
                    fake_paths.append(p)
                    fake_gt.append(gt)

    all_paths = real_paths + fake_paths
    if not all_paths:
        return {"metric": metric_name, "language": language, "error": "no_audio"}

    transcriptions = []
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Transcribing"):
        batch = all_paths[i : i + batch_size]
        try:
            transcriptions.extend(asr.transcribe(batch, language))
        except Exception as e:
            logger.warning(f"Transcribe batch failed: {e}")
            transcriptions.extend([""] * len(batch))
    real_trans = transcriptions[: len(real_paths)]
    fake_trans = transcriptions[len(real_paths) :]

    scores_real = []
    for ref, hyp in zip(real_gt, real_trans):
        if ref.strip() and hyp.strip():
            try:
                scores_real.append(metric_fn(ref, hyp))
            except Exception:
                pass
    scores_fake = []
    for ref, hyp in zip(fake_gt, fake_trans):
        if ref.strip() and hyp.strip():
            try:
                scores_fake.append(metric_fn(ref, hyp))
            except Exception:
                pass

    key = metric_name.lower()
    out = {
        "metric": metric_name,
        "language": language,
        "n_real": len(scores_real),
        "n_fake": len(scores_fake),
    }
    if scores_real:
        out[f"{key}_real_mean"] = float(np.mean(scores_real))
        out[f"{key}_real_std"] = float(np.std(scores_real))
    if scores_fake:
        out[f"{key}_fake_mean"] = float(np.mean(scores_fake))
        out[f"{key}_fake_std"] = float(np.std(scores_fake))
    return out


def main():
    parser = argparse.ArgumentParser(description="WER/CER evaluation for one subset")
    parser.add_argument("-d", "--subset_dir", type=str, required=True, help="Path to the subset directory")
    parser.add_argument("--meta_file", type=str, default="meta.json", help="Metadata filename (e.g. meta_test.json)")
    parser.add_argument("--asr_model", type=str, default="parakeet", choices=["parakeet", "voxtral", "glm_asr"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-l", "--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-o", "--output", type=str, default="results/wer_evaluation.json")
    args = parser.parse_args()

    meta_path = os.path.join(args.subset_dir, args.meta_file)
    if not os.path.exists(meta_path):
        logger.error(f"Metadata not found: {meta_path}")
        return

    asr = get_asr(args.asr_model, args.device)
    results = compute_wer(meta_path, args.subset_dir, asr, args.language, args.batch_size)
    if "error" in results:
        logger.error(results["error"])
        return

    metric = results["metric"]
    k = metric.lower()
    logger.info(f"{metric} (real): {results.get(f'{k}_real_mean', 0):.4f} ± {results.get(f'{k}_real_std', 0):.4f} (n={results['n_real']})")
    logger.info(f"{metric} (fake): {results.get(f'{k}_fake_mean', 0):.4f} ± {results.get(f'{k}_fake_std', 0):.4f} (n={results['n_fake']})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({args.subset_dir: results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {args.output}")


if __name__ == "__main__":
    main()
