import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from ..base import Baseline
from ...config import Label

from .df_whisper import commons as df_commons
from .df_whisper import metrics as df_metrics
from .df_whisper import base_dataset as df_base_dataset
from .df_whisper.models.whisper_specrnet import WhisperMultiFrontSpecRNet

_MODEL_NAME = "whisper_specrnet"
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / f"{_MODEL_NAME}.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _ArrayDataset(Dataset):
    _warned_sox = False

    def __init__(self, data: List[np.ndarray], labels: List[Label], sr: int):
        self.data = data
        self.labels = labels
        self.sr = sr
        self.apply_preprocessing = df_base_dataset.apply_preprocessing
        self.frames_number = getattr(df_base_dataset, "FRAMES_NUMBER", 480_000)

    def _fallback_preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() > 1:
            waveform = waveform[:1, ...]
        waveform = waveform.squeeze(0)
        if waveform.numel() >= self.frames_number:
            return waveform[: self.frames_number]
        repeats = int(self.frames_number / waveform.numel()) + 1
        return waveform.repeat(repeats)[: self.frames_number]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data[idx]
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        try:
            waveform, _ = self.apply_preprocessing(waveform, self.sr)
            if waveform.dim() > 1:
                waveform = waveform.squeeze(0)
        except OSError as exc:
            if "sox" in str(exc).lower():
                if not _ArrayDataset._warned_sox:
                    logger.warning(
                        "torchaudio sox effects unavailable; using simple pad/trim."
                    )
                    _ArrayDataset._warned_sox = True
                waveform = self._fallback_preprocess(waveform)
            else:
                raise
        label = int(self.labels[idx])
        return waveform, label


class WhisperSpecRNet(Baseline):
    """Whisper + LFCC frontend SpecRNet baseline (whisper_frontend_specrnet only)."""

    def __init__(
        self,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.name = _MODEL_NAME
        self.model_name = _MODEL_NAME
        self.supported_metrics = ["eer", "auroc", "accuracy", "precision", "recall", "f1"]

        self._config = _load_config()
        seed = self._config.get("data", {}).get("seed", 42)
        df_commons.set_seed(seed)

        self.model = WhisperMultiFrontSpecRNet(
            input_channels=self._config.get("input_channels", 2),
            freeze_encoder=self._config.get("freeze_encoder", False),
            frontend_algorithm=self._config.get("frontend_algorithm", "lfcc"),
            device=device,
        )

        self.checkpoint_path = self._resolve_checkpoint_path(None)

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> Optional[str]:
        if checkpoint_path:
            return str(Path(checkpoint_path))
        config_ckpt = self._config.get("checkpoint", {}).get("path", "")
        if not config_ckpt:
            return None
        config_ckpt = Path(config_ckpt)
        if not config_ckpt.is_absolute():
            config_ckpt = Path(__file__).resolve().parent / "ckpts" / config_ckpt.name
        return str(config_ckpt)

    def _build_loader(
        self,
        data,
        labels,
        sr,
        shuffle=False,
        drop_last=False,
        batch_size=16,
        num_workers=0,
    ):
        dataset = _ArrayDataset(data, labels, sr)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def _load_cross_dataset(
        self,
        split: str = "train",
        limit: Optional[int] = 512,
        shuffle: bool = False,
        seed: int = 34,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        logger.info(f"Loading cross-domain data from ASVspoof2019 LA {split} ...")
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        real_count = 0
        fake_count = 0
        for item in dataset:
            if item["key"] == 0 and (limit is None or real_count < limit):
                real_data.append(item["audio"]["array"])
                real_count += 1
            elif item["key"] == 1 and (limit is None or fake_count < limit):
                fake_data.append(item["audio"]["array"])
                fake_count += 1
            if limit is not None and real_count >= limit and fake_count >= limit:
                break
        return real_data, fake_data

    def _aggregate_data(
        self, real_data: List[np.ndarray], fake_data: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Label]]:
        data = real_data + fake_data
        labels = [Label.real] * len(real_data) + [Label.fake] * len(fake_data)
        return data, labels

    def _predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scores = []
        labels = []
        with torch.inference_mode():
            for batch, label in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                logits = logits.view(-1)
                probs = torch.sigmoid(logits)
                scores.extend(probs.detach().cpu().numpy().tolist())
                labels.extend(label.numpy().tolist())
        return np.array(scores), np.array(labels)

    def _load_weights(self, checkpoint_path: Optional[str]):
        if not checkpoint_path:
            logger.warning(
                f"{self.name}: no checkpoint provided; using randomly initialized weights"
            )
            return
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)

    def _evaluate_eer(self, scores: np.ndarray, labels: np.ndarray) -> float:
        _, eer, _, _ = df_metrics.calculate_eer(
            y=1 - labels,
            y_score=scores,
        )
        return float(eer)

    def _evaluate_auroc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        return float(roc_auc_score(labels, scores))

    def _evaluate_prf(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
        pred_label = (scores >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_label, average="binary", beta=1.0
        )
        return float(precision), float(recall), float(f1)

    def evaluate(
        self,
        data: List[np.ndarray],
        labels: List[Label],
        metrics: List[str],
        sr: int,
        in_domain: bool = False,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if in_domain and dataset_name is not None:
            ckpt_path = self._get_ckpt_path(dataset_name)
            self._load_weights(ckpt_path)
        else:
            dataset_name = "default"
            ckpt_path = (
                kwargs.get("ckpt_path")
                or self.checkpoint_path
                or self._get_ckpt_path(dataset_name)
            )
            if not ckpt_path or not os.path.exists(ckpt_path):
                logger.info(
                    f"No default checkpoint for {self.name}; training on ASVspoof2019 LA"
                )
                train_real, train_fake = self._load_cross_dataset(
                    split="train", limit=8192, shuffle=True
                )
                train_data, train_labels = self._aggregate_data(train_real, train_fake)
                eval_real, eval_fake = self._load_cross_dataset(
                    split="validation", limit=768, shuffle=False
                )
                eval_data, eval_labels = self._aggregate_data(eval_real, eval_fake)
                self.train(
                    train_data=train_data,
                    train_labels=train_labels,
                    eval_data=eval_data,
                    eval_labels=eval_labels,
                    dataset_name=dataset_name,
                    sr=sr,
                )
            self._load_weights(ckpt_path)

        loader = self._build_loader(
            data, labels, sr, shuffle=False, drop_last=False, batch_size=16
        )
        scores, labels_np = self._predict(loader)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(scores, labels_np)
            results[metric] = metric_rst
        return results

    def _get_ckpt_path(self, dataset_name: str) -> str:
        ckpt_dir = Path(__file__).resolve().parent / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return str(ckpt_dir / f"{dataset_name}_{self.model_name}.pt")

    def train(
        self,
        train_data: List[np.ndarray],
        train_labels: List[Label],
        eval_data: List[np.ndarray],
        eval_labels: List[Label],
        dataset_name: str,
        sr: int = 16000,
        **kwargs,
    ):
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)

        train_loader = self._build_loader(
            train_data,
            train_labels,
            sr,
            shuffle=True,
            drop_last=True,
            batch_size=args["bs"],
            num_workers=args["nb_worker"],
        )
        eval_loader = self._build_loader(
            eval_data,
            eval_labels,
            sr,
            shuffle=False,
            drop_last=False,
            batch_size=args["eval_bs"],
            num_workers=args["nb_worker"],
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args["lr"], weight_decay=args["wd"]
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        best_eer = 1.0
        best_epoch = 0
        best_state = None
        worse_epochs = 0

        os.makedirs("logs", exist_ok=True)
        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(
            f"Training {self.name} on {dataset_name} for {args['epoch']} epochs"
        )

        for epoch in range(args["epoch"]):
            self.model.train()
            running_loss = 0.0
            num_total = 0
            with tqdm(total=len(train_loader), desc="Training") as pbar:
                for batch, label in train_loader:
                    batch = batch.to(self.device)
                    label = label.to(self.device).float().unsqueeze(1)
                    logits = self.model(batch)
                    loss = criterion(logits, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * batch.size(0)
                    num_total += batch.size(0)
                    pbar.set_description(f"epoch: {epoch}, loss: {loss.item():.3f}")
                    pbar.update(1)
            if num_total > 0:
                running_loss /= num_total
            scores, labels_np = self._predict(eval_loader)
            eer = self._evaluate_eer(scores, labels_np)
            logger.info(
                f"Epoch {epoch + 1}/{args['epoch']} - loss: {running_loss:.4f}, EER: {eer:.4f}"
            )
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                worse_epochs = 0
                best_state = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                worse_epochs += 1
            if worse_epochs >= args["patience"]:
                logger.info(
                    f"Early stopping at epoch {epoch} due to no improvement in EER for {args['patience']} epochs."
                )
                break

        if best_state is not None:
            ckpt_path = self._get_ckpt_path(dataset_name)
            torch.save(best_state, ckpt_path)
            self.model.load_state_dict(best_state)
            logger.info(f"Saved best checkpoint to {ckpt_path}")
            logger.info(
                f"Training complete! Best EER: {best_eer:.4f} at epoch {best_epoch}"
            )
        logger.remove(log_id)
