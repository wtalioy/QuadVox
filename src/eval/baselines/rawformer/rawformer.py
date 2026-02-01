import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base import Baseline
from ...config import Label
from .models import Rawformer_SE


class Rawformer(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.name = "Rawformer"
        self.variant = "rawformer_se"
        self.model_config = self._load_model_config(os.path.dirname(__file__), model_name=self.variant)

        self.sample_rate = int(self.model_config.get("sample_rate", 16000))
        self.max_len = int(self.model_config.get("max_len", 64600))
        self.transformer_hidden = int(self.model_config.get("transformer_hidden", 660))
        self.pre_emphasis = float(self.model_config.get("pre_emphasis", 0.0))
        self.eval_batch_size = int(self.model_config.get("eval_bs", 16))
        self.cross_train_limit = int(self.model_config.get("cross_train_limit", 2048))
        self.cross_eval_limit = int(self.model_config.get("cross_eval_limit", 512))
        self.cross_seed = int(self.model_config.get("cross_seed", 34))
        self.use_amp = self.device.startswith("cuda")
        self.grad_accum = 1
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        default_ckpt_name = self.model_config.get("default_ckpt", "rawformer_se.pth")
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", default_ckpt_name)

        self.model = Rawformer_SE(
            device=self.device,
            transformer_hidden=self.transformer_hidden,
            sample_rate=self.sample_rate,
        ).to(self.device)

        self.supported_metrics = ["eer", "auroc"]

    def _apply_pre_emphasis(self, batch: torch.Tensor) -> torch.Tensor:
        if self.pre_emphasis <= 0:
            return batch
        emphasized = batch[:, 1:] - self.pre_emphasis * batch[:, :-1]
        return torch.cat([batch[:, :1], emphasized], dim=1)

    def _init_train(self, args: dict):
        self.criterion = nn.BCELoss()
        lr = float(args.get("lr", 1e-4))
        wd = float(args.get("wd", 0.0))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.grad_accum = max(1, int(args.get("grad_accum", 1)))
        self.use_amp = bool(args.get("use_amp", True)) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _train_epoch(self, epoch: int, train_loader: DataLoader):
        self.model.train()
        with tqdm(total=len(train_loader), desc="Training") as pbar:
            self.optimizer.zero_grad(set_to_none=True)
            for step, (batch, label) in enumerate(train_loader, start=1):
                batch = batch.to(self.device)
                label = label.to(self.device).float().view(-1)
                batch = self._apply_pre_emphasis(batch)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    scores = self.model(batch).view(-1)
                loss = self.criterion(scores.float(), label.float())
                loss = loss / self.grad_accum
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % self.grad_accum == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                pbar.set_description(f"epoch: {epoch}, bce:{loss.item() * self.grad_accum:.3f}")
                pbar.update(1)
            if len(train_loader) % self.grad_accum != 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _evaluate_eer(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        scores = []
        labels = []
        with tqdm(total=len(eval_loader), desc="Evaluating EER") as pbar:
            for batch, label in eval_loader:
                batch = batch.to(self.device)
                batch = self._apply_pre_emphasis(batch)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    output = self.model(batch).view(-1)
                scores.extend(output.cpu().numpy().ravel().tolist())
                labels.extend(label)
                pbar.update(1)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return float(eer)

    @torch.no_grad()
    def _evaluate_auroc(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        scores = []
        labels = []
        with tqdm(total=len(eval_loader), desc="Evaluating AUROC") as pbar:
            for batch, label in eval_loader:
                batch = batch.to(self.device)
                batch = self._apply_pre_emphasis(batch)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    output = self.model(batch).view(-1)
                scores.extend(output.cpu().numpy().ravel().tolist())
                labels.extend(label)
                pbar.update(1)
        return float(roc_auc_score(labels, np.array(scores)))

    def _resolve_ckpt(self, in_domain: bool, dataset_name: Optional[str], ckpt_path: Optional[str]) -> str:
        if ckpt_path is not None:
            return ckpt_path
        if in_domain:
            if not dataset_name:
                raise ValueError("dataset_name is required for in-domain evaluation.")
            return os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        return self.default_ckpt

    def _load_cross_dataset(
        self,
        split: str = "train",
        limit: Optional[int] = 2048,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[Label]]:
        data = []
        labels = []
        logger.info(f"Loading cross-domain data from ASVspoof2019 LA {split} ...")
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
        if shuffle:
            dataset = dataset.shuffle(seed=seed if seed is not None else self.cross_seed)
        real_count = 0
        fake_count = 0
        for item in dataset:
            if item["key"] == 0 and (limit is None or real_count < limit):
                data.append(item["audio"]["array"])
                labels.append(Label.real)
                real_count += 1
            elif item["key"] == 1 and (limit is None or fake_count < limit):
                data.append(item["audio"]["array"])
                labels.append(Label.fake)
                fake_count += 1
            if limit is not None and real_count >= limit and fake_count >= limit:
                break
        return data, labels

    def train(self, train_data: List[np.ndarray], train_labels: np.ndarray, eval_data: List[np.ndarray], eval_labels: np.ndarray, dataset_name: str, **kwargs):
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        batch_size = int(args.get("bs", 64))
        eval_batch_size = int(args.get("eval_bs", batch_size))
        num_workers = int(args.get("nb_worker", 8))
        train_loader = self._prepare_loader(
            train_data,
            train_labels,
            max_len=self.max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        eval_loader = self._prepare_loader(
            eval_data,
            eval_labels,
            max_len=self.max_len,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training {self.name} on {dataset_name}")

        self._init_train(args)

        best_eer = 100.0
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        worse_epochs = 0
        patience = int(args.get("patience", 4))

        for epoch in range(int(args.get("epoch", 15))):
            self._train_epoch(epoch, train_loader)
            eer = self._evaluate_eer(eval_loader)
            logger.info(f"Epoch {epoch} EER: {100 * eer:.2f}%")
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                worse_epochs = 0
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best EER: {100 * best_eer:.2f}% at epoch {epoch}")
            else:
                worse_epochs += 1

            if worse_epochs >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} due to no improvement in EER for {patience} epochs."
                )
                break

        logger.info(f"Training complete! Best EER: {100 * best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)

    def evaluate(
        self,
        data: List[np.ndarray],
        labels: List[Label],
        metrics: List[str],
        in_domain: bool = False,
        dataset_name: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if not in_domain and ckpt_path is None:
            dataset_name = "default"
            ckpt = os.path.join(os.path.dirname(__file__), "ckpts", "default_best.pt")
            if not os.path.exists(ckpt):
                logger.info(f"Default model not found at {ckpt}, training from scratch")
                train_data, train_labels = self._load_cross_dataset(
                    split="train",
                    limit=self.cross_train_limit,
                    shuffle=True,
                    seed=self.cross_seed,
                )
                eval_data, eval_labels = self._load_cross_dataset(
                    split="validation",
                    limit=self.cross_eval_limit,
                    shuffle=False,
                )
                self.train(train_data, train_labels, eval_data, eval_labels, dataset_name="default")
        else:
            ckpt = self._resolve_ckpt(in_domain, dataset_name, ckpt_path)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found for {self.name}: {ckpt}. Train in-domain or pass ckpt_path."
            )
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        eval_loader = self._prepare_loader(
            data,
            labels,
            max_len=self.max_len,
            shuffle=False,
            drop_last=False,
            batch_size=self.eval_batch_size,
        )

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            results[metric] = func(eval_loader)
        return results
