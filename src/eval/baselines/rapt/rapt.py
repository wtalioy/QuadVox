from typing import Any, List, Tuple, Optional
import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import combine_pvalues

from ..base import Baseline
from .mmd_utils import MMD_3_Sample_Test, MMDu
from .model import RAPTModel
from ...config import Label

class RAPT(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device=device, **kwargs)
        self.name = "RAPT"
        self.device = device
        self.sample_rate = 16000
        self.ref_num = 200
        self.seed = 34
        self.supported_metrics = ['eer', 'auroc']

        self.model = RAPTModel(config=self._load_model_config(os.path.dirname(__file__)), device=device)

    def _init_train(self, args: dict):
        self.finetune_extractor = args['finetune_extractor']
        if self.finetune_extractor:
            for param in self.model.extractor.parameters():
                param.requires_grad = True
        else:
            for param in self.model.extractor.parameters():
                param.requires_grad = False

        param_groups = []
        if self.finetune_extractor:
            param_groups.append(
                {
                    'params': self.model.extractor.parameters(),
                    'lr': args['extractor_lr'],
                    'weight_decay': args['extractor_wd'],
                }
            )
        param_groups.extend([
            {
                'params': self.model.mmd_model.parameters(),
                'lr': args['basemodel_lr'],
                'weight_decay': args['basemodel_wd'],
            },
            {
                'params': [self.model.sigma, self.model.sigma0_u],
                'lr': args['mmd_lr'],
                'weight_decay': 0,
            },
            {
                'params': [self.model.raw_ep],
                'lr': args['mmd_lr'] * 0.1,
                'weight_decay': 0,
            },
        ])
        self.optimizer = torch.optim.Adam(param_groups)
        
        total_steps = args['num_epoch'] * args['steps_per_epoch']
        warmup_steps = max(1, int(0.01 * total_steps))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    def _split_data(self, data: List[np.ndarray], labels: List[Label]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        labels_np = np.array(labels)
        real_data = [d for i, d in enumerate(data) if labels_np[i] == Label.real]
        fake_data = [d for i, d in enumerate(data) if labels_np[i] == Label.fake]
        return real_data, fake_data

    def _train_epoch(self, epoch: int, real_loader: DataLoader, fake_loader: DataLoader):
        self.model.train()

        with tqdm(zip(real_loader, fake_loader), total=min(len(real_loader), len(fake_loader))) as pbar:
            for (real_audio, _), (fake_audio, _) in pbar:
                real_audio = real_audio.to(self.device)
                fake_audio = fake_audio.to(self.device)
                if real_audio.size(0) == 1 or fake_audio.size(0) == 1:
                    # only happens when the size of the last batch is 1
                    continue

                inputs = torch.cat([real_audio, fake_audio], dim=0)
                features, outputs = self.model(inputs)

                temp = MMDu(
                    outputs,
                    features.view(features.shape[0], -1),
                    real_audio.shape[0],
                    self.model.sigma,
                    self.model.sigma0_u,
                    self.model.ep,
                    coeff_xy=self.model.coeff_xy,
                    is_yy_zero=self.model.is_yy_zero,
                    is_xx_zero=self.model.is_xx_zero,
                )
                mmd_value_temp = -1 * (temp[0])
                if temp[1] is not None:
                    mmd_std_temp = torch.sqrt(temp[1] + 10 ** (-8))
                else:
                    mmd_std_temp = torch.sqrt(torch.tensor(10 ** (-8), device=mmd_value_temp.device))

                loss = torch.div(mmd_value_temp, mmd_std_temp)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.extractor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.mmd_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_([self.model.sigma, self.model.sigma0_u, self.model.raw_ep], max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_description('epoch:%d, sigma:%.3f, sigma0_u:%.3f, ep:%.3f, loss:%.3f'%(epoch, self.model.sigma.item(), self.model.sigma0_u.item(), self.model.ep.item(), loss.item()))
                pbar.update(1)

    def train(
        self,
        train_data: List[np.ndarray],
        train_labels: List[Label],
        eval_data: List[np.ndarray],
        eval_labels: List[Label],
        dataset_name: str,
        ref_data: Optional[List[np.ndarray]] = None,
        ref_labels: Optional[List[Label]] = None,
        sr: int = 16000,
        finetune_extractor: bool = True,
    ):
        if ref_data is None or ref_labels is None:
            raise ValueError(f"{self.name} requires ref_data and ref_labels to be provided")
        
        # cache ref_data & ref_labels
        self.ref_data = ref_data
        self.ref_labels = ref_labels
        
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        args['finetune_extractor'] = finetune_extractor
        
        ref_real_data, ref_fake_data = self._split_data(ref_data, ref_labels)
        train_real_data, train_fake_data = self._split_data(train_data, train_labels)

        batch_size = args['batch_size']
        real_loader = self._prepare_loader(train_real_data, [Label.real] * len(train_real_data), max_len=64600, batch_size=batch_size//2)
        fake_loader = self._prepare_loader(train_fake_data, [Label.fake] * len(train_fake_data), max_len=64600, batch_size=batch_size//2)
        eval_loader = self._prepare_loader(eval_data, eval_labels, max_len=64600, shuffle=False, drop_last=False, batch_size=64)
        args['steps_per_epoch'] = min(len(real_loader), len(fake_loader))

        self._auto_tune_bandwidths(train_real_data, train_fake_data)

        log_id = logger.add("logs/train.log", rotation="10 MB", retention="60 days")
        logger.info(f"Training RAPT on {dataset_name}")
        
        self._init_train(args)
        
        best_eer = 1.0
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        worse_epochs = 0
        patience = args.get("patience", 5) 

        for epoch in range(args['num_epoch']):
            self._train_epoch(epoch, real_loader, fake_loader)
            if epoch % args['eval_interval'] == 0 and epoch > 0:
                self._precompute_ref_cache(ref_real_data, ref_fake_data)
                eer = self._evaluate_eer(eval_loader, sr=sr)
                logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
                if eer < best_eer:
                    best_eer = eer
                    best_epoch = epoch
                    worse_epochs = 0
                    self.model.save_to_checkpoint(save_path)
                    logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")
                else:
                    worse_epochs += 1
                
                if worse_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch} due to no improvement in EER for {patience * args['eval_interval']} epochs.")
                    break
        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)
    
    def _load_cross_dataset(self, split: str = "train", limit: Optional[int] = 512, shuffle: bool = False, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        logger.info(f"Loading cross-domain data from ASVspoof2019 LA {split} ...")
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
        if shuffle:
            dataset = dataset.shuffle(seed=seed if seed is not None else self.seed)
        real_count = 0
        fake_count = 0
        for item in dataset:
            if item["key"] == 0 and real_count < limit:
                real_data.append(item["audio"]["array"])
                real_count += 1
            elif item["key"] == 1 and fake_count < limit:
                fake_data.append(item["audio"]["array"])
                fake_count += 1
            if real_count >= limit and fake_count >= limit:
                break

        return real_data, fake_data

    def _aggregate_data(self, real_data: List[np.ndarray], fake_data: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Label]]:
        data = real_data + fake_data
        labels = [Label.real] * len(real_data) + [Label.fake] * len(fake_data)
        return data, labels

    @torch.inference_mode()
    def _precompute_ref_cache(self, real_data: List[np.ndarray], fake_data: List[np.ndarray]):
        self.model.eval()
        
        # for real
        real_loader = self._prepare_loader(real_data, [Label.real]*len(real_data), max_len=64600, batch_size=32, shuffle=False, drop_last=False)
        all_real_feas = []
        all_real_net = []
        for audio, _ in real_loader:
            audio = audio.to(self.device)
            feas, net = self.model(audio)
            all_real_feas.append(feas)
            all_real_net.append(net)

        # for fake
        fake_loader = self._prepare_loader(fake_data, [Label.fake]*len(fake_data), max_len=64600, batch_size=32, shuffle=False, drop_last=False)
        all_fake_feas = []
        all_fake_net = []
        for audio, _ in fake_loader:
            audio = audio.to(self.device)
            feas, net = self.model(audio)
            all_fake_feas.append(feas)
            all_fake_net.append(net)

        self._ref_cache = {
            "fea_real": torch.cat(all_real_feas),
            "net_real": torch.cat(all_real_net),
            "fea_fake": torch.cat(all_fake_feas),
            "net_fake": torch.cat(all_fake_net),
        }

    def evaluate(
        self,
        data: List[np.ndarray],
        labels: List[Label],
        metrics: List[str],
        sr: int,
        in_domain: bool = False,
        dataset_name: Optional[str] = None,
        ref_data: Optional[List[np.ndarray]] = None,
        ref_labels: Optional[List[Label]] = None
    ) -> dict:
        torch.manual_seed(self.seed)
        if not in_domain:
            dataset_name = "default"
            ref_data, ref_labels = self._aggregate_data(*self._load_cross_dataset(split="train", limit=self.ref_num))
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"default_best.pt")
            if not os.path.exists(default_ckpt):
                logger.info(f"Default model not found at {default_ckpt}, training from scratch")
                train_real_data, train_fake_data = self._load_cross_dataset(split="train", limit=8192+self.ref_num)
                train_real_data, train_fake_data = train_real_data[self.ref_num:], train_fake_data[self.ref_num:]
                train_data, train_labels = self._aggregate_data(train_real_data, train_fake_data)
                eval_data, eval_labels = self._aggregate_data(*self._load_cross_dataset(split="validation", limit=768))
                self.train(train_data, train_labels, eval_data, eval_labels, dataset_name="default", ref_data=ref_data, ref_labels=ref_labels, sr=sr, finetune_extractor=False)
        else:
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")

        # cached during training
        if ref_data is None or ref_labels is None:
            ref_data, ref_labels = self.ref_data, self.ref_labels
        
        self.model.load_from_checkpoint(default_ckpt)
        ref_real_data, ref_fake_data = self._split_data(ref_data, ref_labels)
        self._precompute_ref_cache(ref_real_data, ref_fake_data)

        if Label.real != 1:
            labels = [1 - label for label in labels]
        eval_loader = self._prepare_loader(data, labels, max_len=64600, shuffle=False, drop_last=False, batch_size=64)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(eval_loader, sr=sr)
            results[metric] = metric_rst

        return results

    @torch.inference_mode()
    def _evaluate_eer(self, eval_loader: DataLoader, sr: int) -> float:
        logger.info("Evaluating EER")
        self.model.eval()

        if sr != self.sample_rate:
            data = [librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate) for audio in data]

        scores, labels = self._three_sample_test(eval_loader, round=8)
        scores = np.array(scores)
        labels = np.array(labels)
        eer, _ = self._compute_eer(scores, labels)
        
        return float(eer)

    @torch.inference_mode()
    def _evaluate_auroc(self, eval_loader: DataLoader, sr: int) -> float:
        logger.info("Evaluating AUROC")
        self.model.eval()
        if sr != self.sample_rate:
            data = [librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate) for audio in data]
        
        scores, labels = self._three_sample_test(eval_loader, round=8)
        scores = np.array(scores)
        
        return float(roc_auc_score(labels, scores))

    def _three_sample_test(
        self,
        eval_loader: DataLoader,
        round: int = 1
    ) -> Tuple[List[float], List[Label]]:
        scores = []
        labels = []
        for audio_batch, label_batch in tqdm(eval_loader, desc="Running three sample test"):
            audio_batch = audio_batch.to(self.device)
            fea_tests_batch, net_tests_batch = self.model(audio_batch)
            labels.extend(label_batch)
            for fea_test, net_test in zip(fea_tests_batch, net_tests_batch):
                fea_real = self._ref_cache["fea_real"]
                net_real = self._ref_cache["net_real"]
                fea_fake = self._ref_cache["fea_fake"]
                net_fake = self._ref_cache["net_fake"]

                idx_real = torch.randperm(len(fea_real), device=fea_real.device)
                idx_fake = torch.randperm(len(fea_fake), device=fea_fake.device)
                fea_real = fea_real[idx_real]
                fea_fake = fea_fake[idx_fake]
                net_real = net_real[idx_real]
                net_fake = net_fake[idx_fake]
                fea_test = fea_test.unsqueeze(0)
                net_test = net_test.unsqueeze(0)

                p_value_list = []
                for j in range(max(round, 1)):
                    p_value = MMD_3_Sample_Test(
                        net_test,
                        net_real[[j]],
                        net_fake[[j]],
                        fea_test.reshape(fea_test.shape[0], -1),
                        fea_real[[j]].reshape(fea_real[[j]].shape[0], -1),
                        fea_fake[[j]].reshape(fea_fake[[j]].shape[0], -1),
                        self.model.sigma,
                        self.model.sigma0_u,
                        self.model.ep,
                    )
                    p_value_list.append(p_value)        
                _, p_value = combine_pvalues(p_value_list, method='stouffer')
                if np.isnan(p_value):
                    logger.warning("NaN score encountered")
                    p_value = 1.0
                scores.append(float(p_value))

        return scores, labels

    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold

    def _auto_tune_bandwidths(self, real_data: List[np.ndarray], fake_data: List[np.ndarray], sample_size: int = 512):
        """Estimate reasonable initial values for sigma (original space) and sigma0_u (hidden space).

        The method samples up to ``sample_size`` utterances from the reference real & fake pools,
        computes pair-wise squared Euclidean distances in both the original feature space and the
        network hidden space, and sets ``self.model.sigma`` to the median original-space distance and
        ``self.model.sigma0_u`` to the median hidden-space distance.
        """
        with torch.inference_mode():
            self.model.eval()
            # Sample from real and fake features and concatenate
            num_real = len(real_data)
            num_fake = len(fake_data)
            total = num_real + num_fake
            
            all_data = real_data + fake_data
            if total > sample_size:
                # Proportional sampling
                real_sample_size = int(sample_size * num_real / total)
                fake_sample_size = sample_size - real_sample_size
                
                real_indices = np.random.permutation(num_real)[:real_sample_size]
                fake_indices = np.random.permutation(num_fake)[:fake_sample_size]

                sampled_data = [real_data[i] for i in real_indices] + [fake_data[i] for i in fake_indices]
            else:
                sampled_data = all_data

            loader = self._prepare_loader(sampled_data, [Label.real]*len(sampled_data), max_len=64600, batch_size=32, shuffle=False, drop_last=False)
            
            all_feas = []
            all_hidden = []
            for audio, _ in loader:
                audio = audio.to(self.device)
                feas, hidden = self.model(audio)
                all_feas.append(feas)
                all_hidden.append(hidden)
            
            feas = torch.cat(all_feas)
            hidden = torch.cat(all_hidden)

            # Original-space distances
            feas_flat = feas.view(feas.size(0), -1)
            D_org = torch.cdist(feas_flat, feas_flat, p=2).pow(2)
            sigma_org = torch.median(D_org)  # 0.5 quantile

            # Hidden-space distances (one forward pass)
            D_hid = torch.cdist(hidden, hidden, p=2).pow(2)
            sigma_hid = torch.quantile(D_hid, 0.9)

            eps = 1e-6
            sigma_org = torch.clamp(sigma_org, min=eps)
            sigma_hid = torch.clamp(sigma_hid, min=eps)

            with torch.no_grad():
                self.model.sigma.data.fill_(sigma_org.item())
                self.model.sigma0_u.data.fill_(sigma_hid.item())

            logger.info(
                f"Auto-tuned sigma={sigma_org.item():.3f}, "
                f"sigma0_u={sigma_hid.item():.3f}"
            )