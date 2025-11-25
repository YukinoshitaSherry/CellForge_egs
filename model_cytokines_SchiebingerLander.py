import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import pandas as pd
from scipy.stats import pearsonr
import time
from datetime import datetime
import argparse
import random
import copy


def print_log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


train_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None


class CytokineTrajectoryDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', age_key='age',
                 scaler=None, pca_model=None, pca_dim=128, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.age_key = age_key
        self.augment = augment
        self.is_train = is_train
        self.pca_dim = pca_dim
        self.common_genes_info = common_genes_info

        # ===== Gene subset & preprocessing =====
        if common_genes_info is not None:
            if is_train:
                gene_idx = common_genes_info['train_idx']
            else:
                gene_idx = common_genes_info['test_idx']
            if scipy.sparse.issparse(adata.X):
                data = adata.X[:, gene_idx].toarray()
            else:
                data = adata.X[:, gene_idx]
        else:
            if scipy.sparse.issparse(adata.X):
                data = adata.X.toarray()
            else:
                data = adata.X

        data = np.maximum(data, 0)
        data = np.maximum(data, 1e-10)

        data_mean_before = np.mean(data)
        data_std_before = np.std(data)
        data_max_before = np.max(data)
        data_min_before = np.min(data)

        is_log1p_already = (data_max_before < 20.0 and data_min_before >= 0.0)
        is_standardized_already = (
            abs(data_mean_before) < 0.5 and 0.5 < data_std_before < 2.0)

        if not is_log1p_already:
            data = np.log1p(data)

        if scaler is None:
            if is_standardized_already:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.zeros(data.shape[1])
                self.scaler.scale_ = np.ones(data.shape[1])
                self.scaler.var_ = np.ones(data.shape[1])
                self.scaler.n_features_in_ = data.shape[1]
            else:
                self.scaler = StandardScaler()
                data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            if not is_standardized_already:
                data = self.scaler.transform(data)

        data = np.clip(data, -10, 10)
        data = data / 10.0

        self.original_expression_data = data.copy()
        self.expression_data = data
        self.n_genes = data.shape[1]
        self.pca = pca_model if pca_model is not None else None
        self.pca_dim = pca_dim if pca_model is not None else None

        # ===== Perturbation encoding =====
        self.perturbation_names = [
            x for x in adata.obs[perturbation_key].unique() if pd.notnull(x)]
        self.perturbations = pd.get_dummies(
            adata.obs[perturbation_key]).values.astype(np.float32)
        print_log(
            f"{'Training' if is_train else 'Test/Val'} set perturbation dimension: {self.perturbations.shape[1]}")

        perturbation_labels = adata.obs[perturbation_key].astype(str).values
        perturbation_labels = np.array(perturbation_labels, dtype='U')
        lower_labels = np.char.lower(perturbation_labels)
        negctrl_mask = np.char.find(lower_labels, 'negctrl') >= 0
        control_mask = (lower_labels == 'control') | (
            lower_labels == 'ctrl') | negctrl_mask
        self._control_indices = np.where(control_mask)[0]
        self._non_control_indices = np.where(~control_mask)[0]
        self._control_set = set(self._control_indices.tolist())

        self._non_control_pert_names = [
            name for name in self.perturbation_names
            if name not in ['control', 'Control', 'ctrl', 'Ctrl']
            and 'negctrl' not in name.lower()
        ]

        if len(self._control_indices) == 0:
            if self.is_train:
                raise ValueError(
                    "CRITICAL ERROR: Training set has NO control samples! "
                    "Control samples are REQUIRED for perturbation prediction. "
                    "This model performs control -> perturbed prediction, NOT autoencoder. "
                    "Please ensure your training data includes control samples "
                    "(labeled as 'control' or containing 'NegCtrl')."
                )
            else:
                raise ValueError(
                    "CRITICAL ERROR: Evaluation set has NO control samples! "
                    "For unseen perturbation prediction, evaluation set should use training set control baseline. "
                    "Please ensure training set has control samples."
                )

        self.rng = np.random.RandomState(42)

        # ===== Time embeddings =====
        self.time_embeddings = self._encode_timepoints(adata.obs[age_key])
        print_log(
            f"{'Training' if is_train else 'Test/Val'} set time dimension: {self.time_embeddings.shape[1]}")

        # 对 is_train=False 的 dataset（val/test）预先固定 baseline-target pair
        if not self.is_train:
            self._create_fixed_pairs_for_test()

    def _encode_timepoints(self, timepoints):
        """
        Encode timepoints like 'D0', 'D2', 'iPSC' into a 3D embedding: [sin, cos, normalized].
        """
        time_mapping = {}
        unique_times = sorted(timepoints.unique())
        for raw_tp in unique_times:
            time_str = str(raw_tp)
            if time_str in ('iPSC', 'iPSCs'):
                time_mapping[time_str] = 20.0
            elif time_str.startswith('D'):
                try:
                    time_mapping[time_str] = float(time_str[1:])
                except Exception:
                    time_mapping[time_str] = 0.0
            else:
                time_mapping[time_str] = 0.0

        time_features = []
        for tp in timepoints:
            time_val = time_mapping[str(tp)]
            sin_time = np.sin(2 * np.pi * time_val / 20.0)
            cos_time = np.cos(2 * np.pi * time_val / 20.0)
            norm_time = time_val / 20.0
            time_features.append([sin_time, cos_time, norm_time])
        return np.array(time_features, dtype=np.float32)

    def _create_fixed_pairs_for_test(self):
        """
        For evaluation datasets, create deterministic (baseline_idx, target_idx) pairs.
        """
        n = len(self.adata)
        self.pairs = [None] * n
        for idx in range(n):
            if idx in self._control_set:
                baseline_idx = idx
                if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                    target_idx = int(
                        self._non_control_indices[idx % len(self._non_control_indices)])
                else:
                    alternative_controls = self._control_indices[self._control_indices != idx]
                    if len(alternative_controls) > 0:
                        target_idx = int(
                            alternative_controls[idx % len(alternative_controls)])
                    else:
                        target_idx = idx
            else:
                if len(self._control_indices) > 0:
                    baseline_idx = int(
                        self._control_indices[idx % len(self._control_indices)])
                else:
                    baseline_idx = idx
                target_idx = idx
            self.pairs[idx] = (baseline_idx, target_idx)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x_current = self.expression_data[idx]
        pert = self.perturbations[idx]
        time_emb = self.time_embeddings[idx]

        if self.is_train:
            # 训练时：随机采样 baseline-target
            if idx in self._control_set:
                x_baseline = x_current
                if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                    target_idx = int(self.rng.choice(
                        self._non_control_indices))
                    x_target = self.original_expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                else:
                    alternative_controls = self._control_indices[self._control_indices != idx]
                    if len(alternative_controls) > 0:
                        target_idx = int(self.rng.choice(alternative_controls))
                        x_target = self.original_expression_data[target_idx]
                        pert_target = self.perturbations[target_idx]
                    else:
                        x_target = self.original_expression_data[idx]
                        pert_target = pert
            else:
                if len(self._control_indices) > 0:
                    baseline_idx = int(self.rng.choice(self._control_indices))
                    x_baseline = self.expression_data[baseline_idx]
                else:
                    x_baseline = x_current
                x_target = self.original_expression_data[idx]
                pert_target = pert
        else:
            # 验证/测试：使用固定的 baseline-target pair
            baseline_idx, target_idx = self.pairs[idx]
            if baseline_idx == idx and idx in self._control_set:
                x_baseline = x_current
            else:
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.original_expression_data[target_idx]
            if target_idx == idx:
                pert_target = pert
            else:
                pert_target = self.perturbations[target_idx]

        x_target_delta = x_target - x_baseline

        # 数据增强仅用于训练集
        if self.augment and self.is_train:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask

        return (
            torch.FloatTensor(x_baseline),
            torch.FloatTensor(pert_target),
            torch.FloatTensor(time_emb),
            torch.FloatTensor(x_target_delta)
        )


class TrajectoryAwareEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, time_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_dim, hidden_dim)

        mlp_layers = []
        for _ in range(2):
            mlp_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.shared_backbone = nn.Sequential(
            *mlp_layers) if mlp_layers else nn.Identity()

        self.shared_mu = nn.Linear(hidden_dim, latent_dim // 2)
        self.shared_logvar = nn.Linear(hidden_dim, latent_dim // 2)
        self.condition_mu = nn.Linear(hidden_dim, latent_dim // 2)
        self.condition_logvar = nn.Linear(hidden_dim, latent_dim // 2)

    def forward(self, x, time_emb):
        x_proj = self.input_proj(x)               # [B, H]
        time_proj = self.time_embedding(time_emb)  # [B, H]
        fused = x_proj + time_proj
        encoded = self.shared_backbone(fused)

        shared_mu = self.shared_mu(encoded)
        shared_logvar = self.shared_logvar(encoded)
        condition_mu = self.condition_mu(encoded)
        condition_logvar = self.condition_logvar(encoded)

        shared_std = torch.exp(0.5 * shared_logvar)
        shared_eps = torch.randn_like(shared_std)
        shared_z = shared_mu + shared_eps * shared_std

        condition_std = torch.exp(0.5 * condition_logvar)
        condition_eps = torch.randn_like(condition_std)
        condition_z = condition_mu + condition_eps * condition_std

        z = torch.cat([shared_z, condition_z], dim=1)  # [B, latent_dim]
        return z, shared_mu, shared_logvar, condition_mu, condition_logvar


class PerturbationMixingModule(nn.Module):
    def __init__(self, latent_dim, pert_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.pert_dim = pert_dim

        self.pert_embedding = nn.Linear(pert_dim, hidden_dim)
        self.mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, pert):
        pert_emb = self.pert_embedding(pert)   # [B, H]
        x = torch.cat([z, pert_emb], dim=1)    # [B, latent+H]

        for layer in self.mixing_layers:
            x = layer(x)
            x = torch.cat([z, x], dim=1)       # maintain [B, latent+H]

        mixed_features = x[:, self.latent_dim:]  # take the last H dims
        output = self.output_proj(mixed_features)  # [B, latent]
        return output


class GraphRegularizedDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # Graph regularizer 目前只是一个线性变换，但我们在 loss 里用的是 delta_expr 的方差项
        self.graph_regularizer = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, z):
        delta_expr = self.decoder(z)                  # [B, G]
        regularized = self.graph_regularizer(delta_expr)
        return delta_expr, regularized


class CytokineTrajectoryModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, time_dim=3, latent_dim=128,
                 hidden_dim=512, model_in_dim=128, bottleneck_dims=(2048, 512),
                 use_bottleneck=True):
        super(CytokineTrajectoryModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
        self.pert_dim = pert_dim
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dims = bottleneck_dims

        if self.use_bottleneck:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, bottleneck_dims[0]),
                nn.LayerNorm(bottleneck_dims[0]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dims[0], bottleneck_dims[1]),
                nn.LayerNorm(bottleneck_dims[1]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dims[1], self.model_in_dim),
                nn.LayerNorm(self.model_in_dim),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, self.model_in_dim),
                nn.LayerNorm(self.model_in_dim),
                nn.GELU()
            )

        self.encoder = TrajectoryAwareEncoder(
            input_dim=self.model_in_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim
        )

        self.perturbation_mixing = PerturbationMixingModule(
            latent_dim=latent_dim,
            pert_dim=pert_dim,
            hidden_dim=hidden_dim
        )

        self.decoder = GraphRegularizedDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, pert, time_emb):
        assert x.dim() == 2 and x.size(1) == self.input_dim, \
            f"Expected x shape [B, {self.input_dim}], got {x.shape}"

        x_proj = self.input_proj(x)  # [B, model_in_dim]
        z, shared_mu, shared_logvar, condition_mu, condition_logvar = self.encoder(
            x_proj, time_emb)

        z_mixed = self.perturbation_mixing(z, pert)
        predicted_expr, regularized_expr = self.decoder(z_mixed)
        delta_expr = predicted_expr  # decoder output is already delta

        return {
            'predicted_expr': predicted_expr,
            'delta_expr': delta_expr,
            'regularized_expr': regularized_expr,
            'shared_mu': shared_mu,
            'shared_logvar': shared_logvar,
            'condition_mu': condition_mu,
            'condition_logvar': condition_logvar,
            'latent_z': z
        }


def train_model(model, train_loader, optimizer, scheduler, device,
                kl_weight=0.1, graph_weight=0.01):
    model.train()
    total_loss = 0.0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_baseline, pert, time_emb, x_target_delta = batch
        x_baseline = x_baseline.to(device)
        pert = pert.to(device)
        time_emb = time_emb.to(device)
        x_target_delta = x_target_delta.to(device)

        outputs = model(x_baseline, pert, time_emb)

        recon_loss = F.mse_loss(outputs['predicted_expr'], x_target_delta)

        shared_kl = -0.5 * torch.sum(
            1 + outputs['shared_logvar'] -
            outputs['shared_mu'].pow(2) -
            outputs['shared_logvar'].exp(), dim=1
        ).mean()

        condition_kl = -0.5 * torch.sum(
            1 + outputs['condition_logvar'] -
            outputs['condition_mu'].pow(2) -
            outputs['condition_logvar'].exp(), dim=1
        ).mean()

        kl_loss = shared_kl + condition_kl

        delta_expr = outputs['delta_expr']
        delta_mean = delta_expr.mean(dim=1, keepdim=True)
        laplacian_loss = torch.mean(
            (delta_expr - delta_mean).pow(2).sum(dim=1))
        graph_loss = laplacian_loss

        loss = recon_loss + kl_weight * kl_loss + graph_weight * graph_loss
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device,
                   kl_weight=0.1, graph_weight=0.01):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            x_baseline, pert, time_emb, x_target_delta = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            time_emb = time_emb.to(device)
            x_target_delta = x_target_delta.to(device)

            outputs = model(x_baseline, pert, time_emb)

            recon_loss = F.mse_loss(outputs['predicted_expr'], x_target_delta)

            shared_kl = -0.5 * torch.sum(
                1 + outputs['shared_logvar'] -
                outputs['shared_mu'].pow(2) -
                outputs['shared_logvar'].exp(), dim=1
            ).mean()

            condition_kl = -0.5 * torch.sum(
                1 + outputs['condition_logvar'] -
                outputs['condition_mu'].pow(2) -
                outputs['condition_logvar'].exp(), dim=1
            ).mean()

            kl_loss = shared_kl + condition_kl

            delta_expr = outputs['delta_expr']
            delta_mean = delta_expr.mean(dim=1, keepdim=True)
            laplacian_loss = torch.mean(
                (delta_expr - delta_mean).pow(2).sum(dim=1))
            graph_loss = laplacian_loss

            loss = recon_loss + kl_weight * kl_loss + graph_weight * graph_loss
            total_loss += loss.item()

            x_target_abs = x_baseline + x_target_delta
            x_pred_abs = x_baseline + outputs['predicted_expr']
            all_targets.append(x_target_abs.cpu().numpy())
            all_predictions.append(x_pred_abs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    true_mean_global = np.mean(all_targets)
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - true_mean_global) ** 2)
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    if np.isnan(r2):
        r2 = 0.0

    pred_flat = all_predictions.flatten()
    true_flat = all_targets.flatten()
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        pear = pearsonr(true_flat, pred_flat)[0]
        if np.isnan(pear):
            pear = 0.0
    else:
        pear = 0.0

    return {
        'loss': total_loss / len(test_loader),
        'r2': r2,
        'pearson': pear
    }


def calculate_detailed_metrics(pred, true, de_genes=None, control_baseline=None, perturbations=None):
    n_samples, n_genes = true.shape

    mse = np.mean((pred - true) ** 2)

    # global PCC
    true_mean_per_gene = np.mean(true, axis=0)
    pred_mean_per_gene = np.mean(pred, axis=0)
    true_centered = true - true_mean_per_gene
    pred_centered = pred - pred_mean_per_gene
    numerator = np.sum(true_centered * pred_centered)
    true_norm = np.sqrt(np.sum(true_centered ** 2))
    pred_norm = np.sqrt(np.sum(pred_centered ** 2))
    if true_norm > 1e-10 and pred_norm > 1e-10:
        pcc = numerator / (true_norm * pred_norm)
        if np.isnan(pcc):
            pcc = 0.0
    else:
        pcc = 0.0

    # global R2
    true_mean_vector = np.mean(true, axis=0)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true_mean_vector) ** 2)
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    if np.isnan(r2):
        r2 = 0.0

    # DE metrics
    if de_genes is not None:
        de_mask = np.zeros(n_genes, dtype=bool)
        de_mask[de_genes] = True
        if np.any(de_mask):
            mse_de = np.mean((pred[:, de_mask] - true[:, de_mask]) ** 2)
            true_de = true[:, de_mask]
            pred_de = pred[:, de_mask]

            true_de_mean_per_gene = np.mean(true_de, axis=0)
            pred_de_mean_per_gene = np.mean(pred_de, axis=0)
            true_de_centered = true_de - true_de_mean_per_gene
            pred_de_centered = pred_de - pred_de_mean_per_gene
            numerator_de = np.sum(true_de_centered * pred_de_centered)
            true_de_norm = np.sqrt(np.sum(true_de_centered ** 2))
            pred_de_norm = np.sqrt(np.sum(pred_de_centered ** 2))
            if true_de_norm > 1e-10 and pred_de_norm > 1e-10:
                pcc_de = numerator_de / (true_de_norm * pred_de_norm)
                if np.isnan(pcc_de):
                    pcc_de = 0.0
            else:
                pcc_de = 0.0

            true_de_mean_vector = np.mean(true_de, axis=0)
            ss_res_de = np.sum((true_de - pred_de) ** 2)
            ss_tot_de = np.sum((true_de - true_de_mean_vector) ** 2)
            if ss_tot_de > 1e-10:
                r2_de = 1.0 - (ss_res_de / ss_tot_de)
            else:
                r2_de = 0.0
            if np.isnan(r2_de):
                r2_de = 0.0
        else:
            mse_de = pcc_de = r2_de = np.nan
    elif control_baseline is not None and control_baseline.shape[0] > 0:
        epsilon = 1e-8
        if perturbations is not None and len(perturbations) == n_samples:
            pert_indices = np.argmax(perturbations, axis=1)
            unique_perts = np.unique(pert_indices)
            mse_de_list = []
            pcc_de_list = []
            r2_de_list = []

            for pert_idx in unique_perts:
                pert_mask = pert_indices == pert_idx
                if np.sum(pert_mask) < 2:
                    continue
                true_pert = true[pert_mask]
                pred_pert = pred[pert_mask]
                true_mean_pert = np.mean(true_pert, axis=0)
                control_mean = np.mean(control_baseline, axis=0)
                lfc = np.log2((true_mean_pert + epsilon) /
                              (control_mean + epsilon))
                lfc_abs = np.abs(lfc)
                K = min(20, n_genes)
                top_k_indices = np.argsort(lfc_abs)[-K:]
                de_mask = np.zeros(n_genes, dtype=bool)
                de_mask[top_k_indices] = True

                if np.any(de_mask):
                    true_de = true_pert[:, de_mask]
                    pred_de = pred_pert[:, de_mask]
                    mse_de_pert = np.mean((pred_de - true_de) ** 2)

                    true_de_mean_per_gene = np.mean(true_de, axis=0)
                    pred_de_mean_per_gene = np.mean(pred_de, axis=0)
                    true_de_centered = true_de - true_de_mean_per_gene
                    pred_de_centered = pred_de - pred_de_mean_per_gene
                    numerator_de = np.sum(true_de_centered * pred_de_centered)
                    true_de_norm = np.sqrt(np.sum(true_de_centered ** 2))
                    pred_de_norm = np.sqrt(np.sum(pred_de_centered ** 2))
                    if true_de_norm > 1e-10 and pred_de_norm > 1e-10:
                        pcc_de_pert = numerator_de / \
                            (true_de_norm * pred_de_norm)
                        if np.isnan(pcc_de_pert):
                            pcc_de_pert = 0.0
                    else:
                        pcc_de_pert = 0.0

                    true_de_mean_vector = np.mean(true_de, axis=0)
                    ss_res_de = np.sum((true_de - pred_de) ** 2)
                    ss_tot_de = np.sum((true_de - true_de_mean_vector) ** 2)
                    if ss_tot_de > 1e-10:
                        r2_de_pert = 1.0 - (ss_res_de / ss_tot_de)
                    else:
                        r2_de_pert = 0.0
                    if np.isnan(r2_de_pert):
                        r2_de_pert = 0.0

                    mse_de_list.append(mse_de_pert)
                    pcc_de_list.append(pcc_de_pert)
                    r2_de_list.append(r2_de_pert)

            if len(mse_de_list) > 0:
                mse_de = np.mean(mse_de_list)
                pcc_de = np.mean(pcc_de_list)
                r2_de = np.mean(r2_de_list)
            else:
                mse_de = pcc_de = r2_de = np.nan
        else:
            mse_de = pcc_de = r2_de = np.nan
    else:
        std = np.std(true, axis=0)
        de_mask = np.abs(true - np.mean(true, axis=0)) > std
        if np.any(de_mask):
            de_genes_indices = np.where(np.any(de_mask, axis=0))[0]
            if len(de_genes_indices) > 0:
                true_de = true[:, de_genes_indices]
                pred_de = pred[:, de_genes_indices]
                mse_de = np.mean((pred_de - true_de) ** 2)

                true_de_mean_per_gene = np.mean(true_de, axis=0)
                pred_de_mean_per_gene = np.mean(pred_de, axis=0)
                true_de_centered = true_de - true_de_mean_per_gene
                pred_de_centered = pred_de - pred_de_mean_per_gene
                numerator_de = np.sum(true_de_centered * pred_de_centered)
                true_de_norm = np.sqrt(np.sum(true_de_centered ** 2))
                pred_de_norm = np.sqrt(np.sum(pred_de_centered ** 2))
                if true_de_norm > 1e-10 and pred_de_norm > 1e-10:
                    pcc_de = numerator_de / (true_de_norm * pred_de_norm)
                    if np.isnan(pcc_de):
                        pcc_de = 0.0
                else:
                    pcc_de = 0.0

                true_de_mean_vector = np.mean(true_de, axis=0)
                ss_res_de = np.sum((true_de - pred_de) ** 2)
                ss_tot_de = np.sum((true_de - true_de_mean_vector) ** 2)
                if ss_tot_de > 1e-10:
                    r2_de = 1.0 - (ss_res_de / ss_tot_de)
                else:
                    r2_de = 0.0
                if np.isnan(r2_de):
                    r2_de = 0.0
            else:
                mse_de = pcc_de = r2_de = np.nan
        else:
            mse_de = pcc_de = r2_de = np.nan

    return {
        'MSE': mse,
        'PCC': pcc,
        'R2': r2,
        'MSE_DE': mse_de,
        'PCC_DE': pcc_de,
        'R2_DE': r2_de
    }


def evaluate_and_save_model(model, test_loader, device, save_path,
                            common_genes_info=None, pca_model=None, scaler=None, best_params=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []
    all_perts = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, time_emb, x_target_delta = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            time_emb = time_emb.to(device)
            x_target_delta = x_target_delta.to(device)

            outputs = model(x_baseline, pert, time_emb)
            x_target_abs = x_baseline + x_target_delta
            x_pred_abs = x_baseline + outputs['predicted_expr']

            all_predictions.append(x_pred_abs.cpu().numpy())
            all_targets.append(x_target_abs.cpu().numpy())
            all_baselines.append(x_baseline.cpu().numpy())
            all_perts.append(pert.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    all_perts = np.concatenate(all_perts, axis=0)

    control_baseline = all_baselines
    results = calculate_detailed_metrics(
        all_predictions,
        all_targets,
        control_baseline=control_baseline,
        perturbations=all_perts
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'baselines': all_baselines,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'best_params': best_params,
        'model_config': {
            'input_dim': model.input_dim,
            'pert_dim': model.pert_dim,
            'time_dim': model.time_dim,
            'latent_dim': model.latent_dim,
            'hidden_dim': model.hidden_dim,
            'model_in_dim': model.model_in_dim,
            'bottleneck_dims': model.bottleneck_dims,
            'use_bottleneck': model.use_bottleneck
        }
    }, save_path)

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    print("\nEvaluation Results:")
    print(metrics_df.to_string(index=False,
          float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")
    return results


def standardize_perturbation_encoding(*datasets):
    """
    对多个 Dataset 统一 one-hot 维度和顺序。
    """
    if len(datasets) == 0:
        raise ValueError(
            "standardize_perturbation_encoding requires at least one dataset.")

    # 收集所有 perturbation 名称
    all_pert_names = set()
    for ds in datasets:
        all_pert_names.update(ds.perturbation_names)
    all_pert_names = sorted(list(all_pert_names))

    # 对每个 dataset 重新 encode
    for ds in datasets:
        pert_df = pd.DataFrame(ds.adata.obs['perturbation'])
        pert_encoded = pd.get_dummies(pert_df['perturbation'])
        for name in all_pert_names:
            if name not in pert_encoded.columns:
                pert_encoded[name] = 0
        pert_encoded = pert_encoded.reindex(
            columns=all_pert_names, fill_value=0)
        ds.perturbations = pert_encoded.values.astype(np.float32)
        ds._pert_encoding_standardized = True

    actual_pert_dim = len(all_pert_names)
    print_log(
        f"Standardized perturbation encoding: {actual_pert_dim} dimensions")
    print_log(f"All perturbation types: {all_pert_names}")
    return actual_pert_dim, all_pert_names


def main(gpu_id=None):
    set_global_seed(42)
    global train_adata, train_dataset, test_dataset, device, pca_model, scaler

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'Training started at: {timestamp}')

    # ===== Device =====
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print_log(
                    f'Warning: Specified GPU {gpu_id} does not exist, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print_log(f'Using specified GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print_log(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print_log('CUDA not available, using CPU')

    # ===== Load data =====
    print_log('Loading data...')
    train_path = "datasets/SchiebingerLander2019_train_processed_filtered2.h5ad"
    test_path = "datasets/SchiebingerLander2019_test_processed_filtered2.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    print_log(f'Training data shape: {train_adata.shape}')
    print_log(f'Test data shape: {test_adata.shape}')

    # ===== Common genes =====
    print_log("Processing gene set consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print_log(f"Training genes: {len(train_genes)}")
    print_log(f"Test genes: {len(test_genes)}")
    print_log(f"Common genes: {len(common_genes)}")

    common_genes.sort()
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]

    pca_model = None
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X[:, train_gene_idx].toarray()
    else:
        train_data = train_adata.X[:, train_gene_idx]

    # 这里只是为了 fit scaler（log1p 后标准化），真正的 transform 在 Dataset 内部做
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)
    scaler = StandardScaler().fit(train_data)

    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }

    # ===== 划分 train / val =====
    n_train_cells = train_adata.n_obs
    indices = np.arange(n_train_cells)
    np.random.shuffle(indices)
    train_size = int(0.8 * n_train_cells)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_adata_train = train_adata[train_indices, :].copy()
    train_adata_val = train_adata[val_indices, :].copy()

    # ===== Build datasets =====
    train_dataset = CytokineTrajectoryDataset(
        train_adata_train,
        perturbation_key='perturbation',
        age_key='age',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )
    val_dataset = CytokineTrajectoryDataset(
        train_adata_val,
        perturbation_key='perturbation',
        age_key='age',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=False,     # 验证集不做数据增强
        is_train=False,    # 采用固定 baseline-target pairing（test 风格）
        common_genes_info=common_genes_info
    )
    test_dataset = CytokineTrajectoryDataset(
        test_adata,
        perturbation_key='perturbation',
        age_key='age',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
    )

    # ===== 统一 perturbation 编码 =====
    pert_dim, _ = standardize_perturbation_encoding(
        train_dataset, val_dataset, test_dataset)

    n_genes = train_dataset.n_genes
    print_log(
        f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")

    model = CytokineTrajectoryModel(
        input_dim=n_genes,
        output_dim=n_genes,
        pert_dim=pert_dim,
        time_dim=3,
        latent_dim=128,
        hidden_dim=512
    ).to(device)

    # ===== DataLoaders =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    best_params = {
        'input_dim': n_genes,
        'output_dim': n_genes,
        'pert_dim': pert_dim,
        'time_dim': 3,
        'latent_dim': 128,
        'hidden_dim': 512,
        'learning_rate': 5e-3,
        'weight_decay': 1e-4,
        'batch_size': 64
    }

    print_log('Training model...')
    best_loss = float('inf')
    best_model_state = None
    max_epochs = 100
    patience = 15
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_model(
            model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate_model(model, val_loader, device)

        if (epoch + 1) % 10 == 0:
            print_log(f'Epoch {epoch + 1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Validation Loss: {val_metrics["loss"]:.4f}')

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': val_metrics,
                'best_params': best_params
            }, f'cytokines_best_model_{timestamp}.pt')
            print_log(
                f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_log(f'Early stopping at epoch {epoch + 1}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print_log('Evaluating final model on test set...')
    results = evaluate_and_save_model(
        model,
        test_loader,
        device,
        f'cytokines_final_model_{timestamp}.pt',
        common_genes_info,
        pca_model,
        scaler,
        best_params
    )

    best_params_str = str(best_params)
    if len(best_params_str) > 50:
        best_params_str = best_params_str[:50] + '...'

    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [best_params_str] * 6
    })
    results_df.to_csv(
        f'cytokines_evaluation_results_{timestamp}.csv', index=False)
    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(
        index=False, float_format=lambda x: '{:.6f}'.format(x)))
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cytokine Trajectory Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID to use (e.g., --gpu 0 for GPU 0, --gpu 1 for GPU 1)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List all available GPUs and exit')
    args = parser.parse_args()

    if args.list_gpus:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f'Available GPUs: {gpu_count}')
            for i in range(gpu_count):
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print('CUDA not available')
        exit(0)

    print("=" * 60)
    print("Cytokine Trajectory Model Training")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using specified GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
