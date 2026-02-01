"""
Distribution analysis: real vs fake t-SNE for a single baseline on test3.

Usage:
  python -m utils.distribution_analysis --baseline aasist -d data/QuadVoxBench -o results/tsne
"""

import os
import json
import argparse
import sys
import warnings
from typing import List, Tuple

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from loguru import logger

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval.baselines import get_baseline_model, list_baseline_models
from eval.subsets import get_subset_model
from eval.test_protocol.test3 import _CONFIG as TEST3_CONFIG


def load_test3_paths(data_dir: str, max_samples: int = 2000) -> Tuple[List[str], List[str]]:
    """Load (audio_path, type) for test3. type is 'real' or 'fake'."""
    paths: List[str] = []
    types: List[str] = []

    for test_name, subset_names in TEST3_CONFIG.test_subsets.items():
        for subset_name in subset_names:
            try:
                subset = get_subset_model(subset_name)(data_dir=data_dir)
                base_dir = subset.data_dir
                meta_path = os.path.join(base_dir, "meta_test.json")
                if not os.path.exists(meta_path):
                    meta_path = os.path.join(base_dir, "meta.json")
                if not os.path.exists(meta_path):
                    logger.warning(f"No meta at {meta_path}, skipping {subset_name}")
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if isinstance(meta, list):
                    for item in meta:
                        if "audio" not in item:
                            continue
                        if "real" in item.get("audio", {}):
                            p = os.path.join(base_dir, item["audio"]["real"])
                            if os.path.exists(p):
                                paths.append(p)
                                types.append("real")
                        if "fake" in item.get("audio", {}):
                            fake = item["audio"]["fake"]
                            for path in (fake.values() if isinstance(fake, dict) else [fake]):
                                p = os.path.join(base_dir, path)
                                if os.path.exists(p):
                                    paths.append(p)
                                    types.append("fake")
                else:
                    for source_name, items in meta.items():
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if "audio" not in item:
                                continue
                            if "real" in item.get("audio", {}):
                                p = os.path.join(os.path.dirname(base_dir), source_name, item["audio"]["real"])
                                if os.path.exists(p):
                                    paths.append(p)
                                    types.append("real")
                            if "fake" in item.get("audio", {}):
                                fake = item["audio"]["fake"]
                                if isinstance(fake, dict):
                                    for path in fake.values():
                                        p = os.path.join(base_dir, path)
                                        if os.path.exists(p):
                                            paths.append(p)
                                            types.append("fake")
                                else:
                                    p = os.path.join(base_dir, fake)
                                    if os.path.exists(p):
                                        paths.append(p)
                                        types.append("fake")
            except Exception as e:
                logger.warning(f"Error loading {subset_name}: {e}")
                continue

    if len(paths) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), max_samples, replace=False)
        paths = [paths[i] for i in sorted(idx)]
        types = [types[i] for i in sorted(idx)]
    logger.info(f"Loaded {len(paths)} paths ({sum(1 for t in types if t == 'real')} real, {sum(1 for t in types if t == 'fake')} fake)")
    return paths, types


AUDIO_LEN = 64600
SR = 16000


def _pad_audio(audio: np.ndarray, length: int = AUDIO_LEN) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    n = int(length / len(audio)) + 1
    return np.tile(audio, (1, n))[:, :length][0]


def make_extractor(baseline: str, device: str = "cuda"):
    """Return a function path -> feature vector. Loads the baseline model once."""
    if not get_baseline_model or baseline not in list_baseline_models():
        raise ValueError(f"Unknown baseline '{baseline}'. Choose from: {list_baseline_models()}")

    bl = get_baseline_model(baseline)(device=device)
    ckpt_path = getattr(bl, "default_ckpt", None)
    if ckpt_path:
        ckpt_dir = os.path.dirname(ckpt_path)
        test3 = os.path.join(ckpt_dir, "test3_best.pt")
        if os.path.exists(test3):
            ckpt_path = test3
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        bl.model.load_state_dict(state, strict=False)
    bl.model.eval()

    def extract(path: str) -> np.ndarray | None:
        try:
            audio, _ = librosa.load(path, sr=SR)
        except Exception as e:
            logger.warning(f"Load failed {path}: {e}")
            return None
        x = torch.from_numpy(_pad_audio(audio)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = bl.model(x)
        feat = out[0] if isinstance(out, tuple) else out
        return feat.squeeze(0).cpu().numpy().flatten()

    return extract


def compute_tsne(features: np.ndarray, perplexity: int = 30, max_iter: int = 1000, seed: int = 42) -> np.ndarray:
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=seed)
        features = pca.fit_transform(features)
    perplexity = min(perplexity, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=max_iter, verbose=1)
    return tsne.fit_transform(features)


def plot_real_vs_fake(tsne_coords: np.ndarray, types: List[str], output_path: str) -> None:
    colors = ["#2ecc71" if t == "real" else "#e74c3c" for t in types]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=colors, alpha=0.6, s=40)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="Real"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="Fake"),
        ],
        loc="best",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real vs fake t-SNE for one baseline on test3")
    parser.add_argument("--baseline", "-b", type=str, required=True, help="Baseline name (e.g. aasist, rawnet2, rapt)")
    parser.add_argument("-d", "--data_dir", type=str, default="data/QuadVoxBench", help="QuadVoxBench data directory")
    parser.add_argument("-o", "--output", type=str, default="results/distribution_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=2000, help="Cap samples for speed")
    args = parser.parse_args()

    paths, types = load_test3_paths(args.data_dir, max_samples=args.max_samples)
    if not paths:
        logger.error("No audio paths found")
        return

    extract = make_extractor(args.baseline, args.device)
    features_list = []
    types_ok = []
    for path, t in tqdm(zip(paths, types), total=len(paths), desc="Extracting"):
        f = extract(path)
        if f is not None:
            features_list.append(f)
            types_ok.append(t)
    if not features_list:
        logger.error("No features extracted")
        return

    X = np.array(features_list)
    logger.info(f"Features shape {X.shape}")
    tsne_coords = compute_tsne(X, perplexity=args.perplexity, max_iter=args.max_iter)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"tsne_{args.baseline}_real_vs_fake.png")
    plot_real_vs_fake(tsne_coords, types_ok, out_path)
    np.save(os.path.join(args.output, f"tsne_{args.baseline}.npy"), tsne_coords)
    logger.info("Done.")


if __name__ == "__main__":
    main()
