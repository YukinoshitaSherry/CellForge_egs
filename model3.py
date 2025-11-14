import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import optuna
import time
from datetime import datetime
import argparse
import math
def print_log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
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
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train
        self.common_genes_info = common_genes_info
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
        data = np.log1p(data)
        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)
        data = np.clip(data, -10, 10)
        data = data / 10.0
        self.original_expression_data = data.copy()
        self.expression_data = data
        self.n_genes = data.shape[1]
        self.pca = pca_model if pca_model is not None else None
        self.pca_dim = pca_dim if pca_model is not None else None
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print_log(f"{'Training' if is_train else 'Test'} set perturbation dimension: {self.perturbations.shape[1]}")
        perturbation_labels = adata.obs[perturbation_key].astype(str).values
        perturbation_labels = np.array(perturbation_labels, dtype='U')
        lower_labels = np.char.lower(perturbation_labels)
        negctrl_mask = np.char.find(lower_labels, 'negctrl') >= 0
        control_mask = (lower_labels == 'control') | \
            (lower_labels == 'ctrl') | negctrl_mask
        self._control_indices = np.where(control_mask)[0]
        self._non_control_indices = np.where(~control_mask)[0]
        self.perturbation_names = list(adata.obs[perturbation_key].unique())
        self._non_control_pert_names = [name for name in self.perturbation_names 
                                        if name not in ['control', 'Control', 'ctrl', 'Ctrl'] 
                                        and 'negctrl' not in name.lower()]
        if len(self._control_indices) == 0:
            if self.is_train:
                raise ValueError(
                    "CRITICAL ERROR: Training set has NO control samples! "
                    "Control samples are REQUIRED for perturbation prediction. "
                    "This model performs control -> perturbed prediction, NOT autoencoder. "
                    "Please ensure your training data includes control samples (labeled as 'control' or containing 'NegCtrl')."
                )
            else:
                raise ValueError(
                    "CRITICAL ERROR: Test set has NO control samples! "
                    "For unseen perturbation prediction, test set should use training set control baseline. "
                    "Please ensure training set has control samples."
                )
        self.time_embeddings = self._encode_timepoints(adata.obs[age_key])
        print_log(f"{'Training' if is_train else 'Test'} set time dimension: {self.time_embeddings.shape[1]}")
    def _encode_timepoints(self, timepoints):
        time_mapping = {}
        unique_times = sorted(timepoints.unique())
        for i, time_str in enumerate(unique_times):
            if time_str == 'iPSC' or time_str == 'iPSCs':
                time_mapping[time_str] = 20.0  
            elif time_str.startswith('D'):
                try:
                    time_mapping[time_str] = float(time_str[1:])
                except:
                    time_mapping[time_str] = 0.0
            else:
                time_mapping[time_str] = 0.0
        time_features = []
        for tp in timepoints:
            time_val = time_mapping[tp]
            sin_time = np.sin(2 * np.pi * time_val / 20.0)  
            cos_time = np.cos(2 * np.pi * time_val / 20.0)
            norm_time = time_val / 20.0
            time_features.append([sin_time, cos_time, norm_time])
        return np.array(time_features)
    def __len__(self):
        return len(self.adata)
    def __getitem__(self, idx):
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]
        time_emb = self.time_embeddings[idx]
        if idx in self._control_indices:
            if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                if self.is_train:
                    target_idx = int(np.random.choice(self._non_control_indices))
                else:
                    target_idx = int(self._non_control_indices[idx % len(self._non_control_indices)])
                x_target = self.original_expression_data[target_idx]
                pert_target = self.perturbations[target_idx]
            else:
                alternative_controls = self._control_indices[self._control_indices != idx]
                if len(alternative_controls) > 0:
                    if self.is_train:
                        target_idx = int(np.random.choice(alternative_controls))
                    else:
                        target_idx = int(alternative_controls[idx % len(alternative_controls)])
                    x_target = self.original_expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                else:
                    x_target = self.original_expression_data[idx]
                    pert_target = pert
        else:
            baseline_idx = None
            if len(self._control_indices) > 0:
                if self.is_train:
                    baseline_idx = int(np.random.choice(self._control_indices))
                else:
                    baseline_idx = int(self._control_indices[idx % len(self._control_indices)])
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.original_expression_data[idx]
            pert_target = pert
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask
        x_target_delta = x_target - x_baseline
        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor(time_emb), torch.FloatTensor(x_target_delta)
class TrajectoryAwareEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, time_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        mlp_layers = []
        for _ in range(2):
            mlp_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.shared_backbone = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_dim, hidden_dim)
        self.shared_mu = nn.Linear(hidden_dim, latent_dim // 2)
        self.shared_logvar = nn.Linear(hidden_dim, latent_dim // 2)
        self.condition_mu = nn.Linear(hidden_dim, latent_dim // 2)
        self.condition_logvar = nn.Linear(hidden_dim, latent_dim // 2)
    def forward(self, x, time_emb):
        x_proj = self.input_proj(x)
        time_proj = self.time_embedding(time_emb)
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
        z = torch.cat([shared_z, condition_z], dim=1)
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
        pert_emb = self.pert_embedding(pert)
        x = torch.cat([z, pert_emb], dim=1)  
        for layer in self.mixing_layers:
            x_input = x  
            x = layer(x)  
            x = torch.cat([z, x], dim=1)  
        mixed_features = x[:, self.latent_dim:]  
        output = self.output_proj(mixed_features)  
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
        self.graph_regularizer = nn.Linear(output_dim, output_dim, bias=False)
    def forward(self, z):
        delta_expr = self.decoder(z)
        regularized = self.graph_regularizer(delta_expr)
        return delta_expr, regularized
class CytokineTrajectoryModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, time_dim=3, latent_dim=128, 
                 hidden_dim=512, use_optimal_transport=True, use_contrastive=True,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super(CytokineTrajectoryModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
        self.pert_dim = pert_dim
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_optimal_transport = use_optimal_transport
        self.use_contrastive = use_contrastive
        self.use_bottleneck = use_bottleneck
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
        if use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 64)
            )
        if use_optimal_transport:
            self.ot_layer = nn.Linear(latent_dim, latent_dim)
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
        x_proj = self.input_proj(x)
        z, shared_mu, shared_logvar, condition_mu, condition_logvar = self.encoder(x_proj, time_emb)
        z_diffused = self.perturbation_mixing(z, pert)
        if self.use_optimal_transport:
            z_ot = self.ot_layer(z_diffused)
        else:
            z_ot = z_diffused
        predicted_expr, regularized_expr = self.decoder(z_ot)
        delta_expr = predicted_expr  
        contrastive_feat = None
        if self.use_contrastive:
            contrastive_feat = self.contrastive_head(z)
        return {
            'predicted_expr': predicted_expr,
            'delta_expr': delta_expr,
            'regularized_expr': regularized_expr,
            'shared_mu': shared_mu,
            'shared_logvar': shared_logvar,
            'condition_mu': condition_mu,
            'condition_logvar': condition_logvar,
            'contrastive_feat': contrastive_feat,
            'latent_z': z
        }
def train_model(model, train_loader, optimizer, scheduler, device, 
                aux_weight=0.1, contrastive_weight=0.05, ot_weight=0.01):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        x_baseline, pert, time_emb, x_target_delta = batch
        x_baseline, pert, time_emb, x_target_delta = x_baseline.to(device), pert.to(device), time_emb.to(device), x_target_delta.to(device)
        outputs = model(x_baseline, pert, time_emb)
        recon_loss = F.mse_loss(outputs['predicted_expr'], x_target_delta)
        shared_kl = -0.5 * torch.sum(1 + outputs['shared_logvar'] - 
                                    outputs['shared_mu'].pow(2) - 
                                    outputs['shared_logvar'].exp(), dim=1).mean()
        condition_kl = -0.5 * torch.sum(1 + outputs['condition_logvar'] - 
                                       outputs['condition_mu'].pow(2) - 
                                       outputs['condition_logvar'].exp(), dim=1).mean()
        kl_loss = shared_kl + condition_kl
        delta_expr = outputs['delta_expr']
        delta_mean = delta_expr.mean(dim=1, keepdim=True)
        laplacian_loss = torch.mean((delta_expr - delta_mean).pow(2).sum(dim=1))
        graph_loss = laplacian_loss
        contrastive_loss = 0
        loss = recon_loss + aux_weight * kl_loss + ot_weight * graph_loss
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(train_loader)
def evaluate_model(model, test_loader, device, aux_weight=0.1, contrastive_weight=0.05, ot_weight=0.01):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    with torch.no_grad():
        for batch in test_loader:
            x_baseline, pert, time_emb, x_target_delta = batch
            x_baseline, pert, time_emb, x_target_delta = x_baseline.to(device), pert.to(device), time_emb.to(device), x_target_delta.to(device)
            outputs = model(x_baseline, pert, time_emb)
            recon_loss = F.mse_loss(outputs['predicted_expr'], x_target_delta)
            shared_kl = -0.5 * torch.sum(1 + outputs['shared_logvar'] - 
                                        outputs['shared_mu'].pow(2) - 
                                        outputs['shared_logvar'].exp(), dim=1).mean()
            condition_kl = -0.5 * torch.sum(1 + outputs['condition_logvar'] - 
                                           outputs['condition_mu'].pow(2) - 
                                           outputs['condition_logvar'].exp(), dim=1).mean()
            kl_loss = shared_kl + condition_kl
            delta_expr = outputs['delta_expr']
            delta_mean = delta_expr.mean(dim=1, keepdim=True)
            laplacian_loss = torch.mean((delta_expr - delta_mean).pow(2).sum(dim=1))
            graph_loss = laplacian_loss
            contrastive_loss = 0
            loss = recon_loss + aux_weight * kl_loss + ot_weight * graph_loss
            total_loss += loss.item()
            x_target_abs = x_baseline + x_target_delta
            x_pred_abs = x_baseline + outputs['predicted_expr']
            x_target_np = x_target_abs.cpu().numpy()
            x_pred_np = x_pred_abs.cpu().numpy()
            true_mean_overall = np.mean(x_target_np)
            ss_res = np.sum((x_target_np - x_pred_np) ** 2)
            ss_tot = np.sum((x_target_np - true_mean_overall) ** 2)
            if ss_tot > 1e-10:
                r2 = 1.0 - (ss_res / ss_tot)
            else:
                r2 = 0.0
            if np.isnan(r2):
                r2 = 0.0
            total_r2 += r2
            pred_flat = x_pred_np.flatten()
            true_flat = x_target_np.flatten()
            if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
                pearson = pearsonr(true_flat, pred_flat)[0]
                if np.isnan(pearson):
                    pearson = 0.0
            else:
                pearson = 0.0
            total_pearson += pearson
    return {
        'loss': total_loss / len(test_loader),
        'r2': total_r2 / len(test_loader),
        'pearson': total_pearson / len(test_loader)
    }
def calculate_detailed_metrics(pred, true, de_genes=None, control_baseline=None):
    n_samples, n_genes = true.shape
    mse = np.mean((pred - true) ** 2)
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
    true_mean_vector = np.mean(true, axis=0)  
    ss_res = np.sum((true - pred) ** 2)  
    ss_tot = np.sum((true - true_mean_vector) ** 2)  
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    if np.isnan(r2):
        r2 = 0.0
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
        true_mean_pert = np.mean(true, axis=0)
        control_mean = np.mean(control_baseline, axis=0)
        lfc = np.log2((true_mean_pert + epsilon) / (control_mean + epsilon))
        lfc_abs = np.abs(lfc)
        K = min(20, n_genes)
        top_k_indices = np.argsort(lfc_abs)[-K:]
        de_mask = np.zeros(n_genes, dtype=bool)
        de_mask[top_k_indices] = True
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
                           common_genes_info=None, pca_model=None, scaler=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, time_emb, x_target_delta = batch
            x_baseline, pert, time_emb, x_target_delta = x_baseline.to(device), pert.to(device), time_emb.to(device), x_target_delta.to(device)
            outputs = model(x_baseline, pert, time_emb)
            x_target_abs = x_baseline + x_target_delta
            x_pred_abs = x_baseline + outputs['predicted_expr']
            all_predictions.append(x_pred_abs.cpu().numpy())
            all_targets.append(x_target_abs.cpu().numpy())
            all_baselines.append(x_baseline.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    control_baseline = None
    results = calculate_detailed_metrics(all_predictions, all_targets, control_baseline=control_baseline)
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'baselines': all_baselines,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': {
            'input_dim': model.input_dim,
            'pert_dim': model.pert_dim,
            'time_dim': 3,
            'latent_dim': 128,
            'hidden_dim': 512,
            'use_optimal_transport': True,
            'use_contrastive': True
        }
    }, save_path)
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'], 
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    print("\nEvaluation Results:")
    print(metrics_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")
    return results
def standardize_perturbation_encoding(train_dataset, test_dataset):
    train_pert_dim = train_dataset.perturbations.shape[1]
    test_pert_dim = test_dataset.perturbations.shape[1]
    max_pert_dim = max(train_pert_dim, test_pert_dim)
    all_pert_names = set(train_dataset.perturbation_names +
                         test_dataset.perturbation_names)
    all_pert_names = sorted(list(all_pert_names))
    train_pert_df = pd.DataFrame(train_dataset.adata.obs['perturbation'])
    train_pert_encoded = pd.get_dummies(train_pert_df['perturbation'])
    for pert_name in all_pert_names:
        if pert_name not in train_pert_encoded.columns:
            train_pert_encoded[pert_name] = 0
    train_pert_encoded = train_pert_encoded.reindex(
        columns=all_pert_names, fill_value=0)
    train_dataset.perturbations = train_pert_encoded.values.astype(np.float32)
    test_pert_df = pd.DataFrame(test_dataset.adata.obs['perturbation'])
    test_pert_encoded = pd.get_dummies(test_pert_df['perturbation'])
    for pert_name in all_pert_names:
        if pert_name not in test_pert_encoded.columns:
            test_pert_encoded[pert_name] = 0
    test_pert_encoded = test_pert_encoded.reindex(
        columns=all_pert_names, fill_value=0)
    test_dataset.perturbations = test_pert_encoded.values.astype(np.float32)
    actual_pert_dim = len(all_pert_names)
    print_log(
        f"Standardized perturbation encoding: {actual_pert_dim} dimensions")
    print_log(f"All perturbation types: {all_pert_names}")
    return actual_pert_dim, all_pert_names
def main(gpu_id=None):
    global train_adata, train_dataset, test_dataset, device, pca_model, scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'Training started at: {timestamp}')
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print_log(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print_log(f'Warning: Specified GPU {gpu_id} does not exist, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print_log(f'Using specified GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print_log(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print_log('CUDA not available, using CPU')
    print_log('Loading data...')
    train_path = "/datasets/SchiebingerLander2019_train_processed.h5ad"
    test_path = "/datasets/SchiebingerLander2019_test_processed.h5ad"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found: {train_path} or {test_path}")
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    print_log(f'Training data shape: {train_adata.shape}')
    print_log(f'Test data shape: {test_adata.shape}')
    print_log("Processing gene set consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print_log(f"Training genes: {len(train_genes)}")
    print_log(f"Test genes: {len(test_genes)}")
    print_log(f"Common genes: {len(common_genes)}")
    common_genes.sort()
    train_gene_idx = [train_adata.var_names.get_loc(gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(gene) for gene in common_genes]
    pca_model = None
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X[:, train_gene_idx].toarray()
    else:
        train_data = train_adata.X[:, train_gene_idx]
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0
    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }
    train_dataset = CytokineTrajectoryDataset(
        train_adata,
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
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    model = CytokineTrajectoryModel(
        input_dim=n_genes,
        output_dim=n_genes,  
        pert_dim=pert_dim,
        time_dim=3,
        latent_dim=128,
        hidden_dim=512,
        use_optimal_transport=True,
        use_contrastive=True
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
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
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    print_log('Training model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 100
    for epoch in range(max_epochs):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_model(model, test_loader, device)
        if (epoch + 1) % 10 == 0:
            print_log(f'Epoch {epoch+1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics
            }, f'cytokines_best_model_{timestamp}.pt')
            print_log(f"Saved best model with loss: {best_loss:.4f}")
    model.load_state_dict(best_model)
    print_log('Evaluating final model...')
    results = evaluate_and_save_model(model, test_loader, device, 
                                    f'cytokines_final_model_{timestamp}.pt', 
                                    common_genes_info, pca_model, scaler)
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'], 
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    results_df.to_csv(f'cytokines_evaluation_results_{timestamp}.csv', index=False)
    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    return results_df
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cytokine Trajectory Model Training')
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