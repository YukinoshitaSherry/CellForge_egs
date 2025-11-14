import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader, random_split
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
import warnings
from typing import Dict, List, Tuple, Optional, Union
import random
warnings.filterwarnings('ignore')
def print_log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None
common_genes_info = None
class ControlPerturbedDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None,
                 pca_model=None, pca_dim=128, fit_pca=False, augment=False,
                 is_train=True, common_genes_info=None, train_control_baseline=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.augment = augment
        self.is_train = is_train
        self.pca_dim = pca_dim
        self.common_genes_info = common_genes_info
        self.train_control_baseline = train_control_baseline  
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
        is_standardized_already = (abs(data_mean_before) < 0.5 and 0.5 < data_std_before < 2.0)
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
        self._create_control_perturbed_pairs()
        print_log(f"Dataset created: {len(self.pairs)} pairs, {self.expression_data.shape[1]} features")
    def _create_control_perturbed_pairs(self):
        perturbation_col = self.adata.obs[self.perturbation_key].astype(str)        
        negctrl_mask = perturbation_col.str.contains('NegCtrl', case=False, na=False)    
        lower_labels = perturbation_col.str.lower()
        exact_control_mask = (lower_labels == 'control') 
        other_control_mask = perturbation_col.isin(['Control', 'ctrl', 'Ctrl', 'CONTROL'])
        contains_control_mask = perturbation_col.str.contains('control|ctrl', case=False, na=False)
        control_mask = exact_control_mask | negctrl_mask | other_control_mask | contains_control_mask
        control_indices = np.where(control_mask)[0]
        perturbed_indices = np.where(~control_mask)[0]
        print_log(f"Control samples: {len(control_indices)}")
        print_log(f"Perturbed samples: {len(perturbed_indices)}")
        if len(control_indices) == 0:
            unique_perturbations = perturbation_col.unique()[:10]  
            print_log(f"DEBUG: Sample perturbation values: {unique_perturbations}")
            print_log("DEBUG: Attempting to find control samples with alternative patterns...")
        self.perturbation_names = list(perturbation_col.unique())
        self.perturbations = pd.get_dummies(perturbation_col).values.astype(np.float32)
        print_log(f"Perturbation dimension: {self.perturbations.shape[1]}")
        if len(control_indices) == 0:
            if self.is_train:     
                raise ValueError(
                    "CRITICAL ERROR: Training set has NO control samples! "
                    "Control samples are REQUIRED for perturbation prediction. "
                    "Please ensure your training data includes control samples (labeled as 'control' or containing 'NegCtrl')."
                )
            else:
                if self.train_control_baseline is None:
                    raise ValueError(
                        "CRITICAL ERROR: Test set has no control samples AND no training control baseline provided! "
                        "This is required for unseen perturbation prediction. "
                        "Please ensure training set has control samples."
                    )
                else:
                    print_log("=" * 60)
                    print_log("Test set: No control samples found in test data.")
                    print_log("Using training set control baseline (CORRECT for unseen perturbation task).")
                    print_log("=" * 60)
                    self.avg_baseline = self.train_control_baseline
                    self.use_avg_baseline = True
        else:
            print_log(f"Using {len(control_indices)} true control samples as baseline")
            control_baseline = np.mean(self.expression_data[control_indices], axis=0)
            self.control_baseline = control_baseline  
            self.use_avg_baseline = False
            self.avg_baseline = None
        self.pairs = []
        for i, perturbed_idx in enumerate(perturbed_indices):
            if self.use_avg_baseline:
                baseline_idx = -1
            else:
                available_controls = control_indices[control_indices != perturbed_idx]
                if len(available_controls) > 0:
                    if self.is_train:
                        baseline_idx = np.random.choice(available_controls)
                    else:
                        baseline_idx = available_controls[i % len(available_controls)]
                else:
                    if self.is_train:
                        baseline_idx = np.random.choice(control_indices)
                    else:
                        baseline_idx = control_indices[i % len(control_indices)]
            self.pairs.append({
                'baseline_idx': baseline_idx,
                'perturbed_idx': perturbed_idx,
                'perturbation': self.perturbations[perturbed_idx]
            })
        if len(control_indices) > 0 and len(perturbed_indices) > 0:
            for i, control_idx in enumerate(control_indices):
                if self.is_train:
                    target_idx = np.random.choice(perturbed_indices)
                else:
                    target_idx = perturbed_indices[i % len(perturbed_indices)]
                self.pairs.append({
                    'baseline_idx': control_idx,
                    'perturbed_idx': target_idx,
                    'perturbation': self.perturbations[target_idx]
                })
        elif len(control_indices) > 1:
            for i, control_idx in enumerate(control_indices):
                other_controls = control_indices[control_indices != control_idx]
                if len(other_controls) > 0:
                    if self.is_train:
                        baseline_idx = np.random.choice(other_controls)
                    else:
                        baseline_idx = other_controls[i % len(other_controls)]
                    self.pairs.append({
                        'baseline_idx': baseline_idx,
                        'perturbed_idx': control_idx,
                        'perturbation': self.perturbations[control_idx]
                    })
        print_log(f"Total pairs created: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        if pair['baseline_idx'] == -1:
            x_control = self.avg_baseline.copy()
        else:
            x_control = self.expression_data[pair['baseline_idx']]
        x_perturbed = self.expression_data[pair['perturbed_idx']]
        pert = pair['perturbation']
        if pair['baseline_idx'] != -1 and pair['baseline_idx'] == pair['perturbed_idx']:
            print_log(f"WARNING: Found self-pair (baseline_idx == perturbed_idx). This should not happen!")
            if hasattr(self, 'avg_baseline') and self.avg_baseline is not None:
                x_control = self.avg_baseline.copy()
            else:
                x_control = self.expression_data[0].copy()
            x_perturbed = self.expression_data[pair['perturbed_idx']]
            pert = pair['perturbation']
        if self.augment and self.is_train:
            noise = np.random.normal(0, 0.05, x_control.shape)
            x_control = x_control + noise
            mask = np.random.random(x_control.shape) > 0.05
            x_control = x_control * mask
        x_perturbed_original = self.original_expression_data[pair['perturbed_idx']]
        x_target_delta = x_perturbed_original - x_control
        return (torch.FloatTensor(x_control),
                torch.FloatTensor(pert),
                torch.FloatTensor(x_target_delta))
    def get_pert_dim(self):
        return self.perturbations.shape[1]
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = F.gelu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, z):
        h = F.gelu(self.fc1(z))
        return self.fc2(h)
class PerturbationEmbedding(nn.Module):
    def __init__(self, pert_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(pert_dim, emb_dim)
    def forward(self, pert):
        return self.embedding(pert)
class SingleTokenAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, n_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    def forward(self, query, key, value):
        batch_size = query.size(0)
        Q = self.query_proj(query).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)
        output = self.output_proj(attended)
        return output
class HybridAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, hidden_dim=512, n_layers=2, n_heads=8,
                 dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1, activation='gelu',
                 use_vae=False, vae_latent_dim=64, vae_hidden_dim=256,
                 use_pert_emb=False, pert_emb_dim=32, vae_beta=1.0,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super(HybridAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_bottleneck = use_bottleneck
        self.use_vae = use_vae
        self.vae_beta = vae_beta
        self.use_pert_emb = use_pert_emb
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
        if use_vae:
            self.vae_encoder = VAEEncoder(self.model_in_dim, vae_latent_dim, vae_hidden_dim)
            self.vae_decoder = VAEDecoder(vae_latent_dim, self.model_in_dim, vae_hidden_dim)
            expr_out_dim = vae_latent_dim
        else:
            self.expression_encoder = nn.Sequential(
                nn.Linear(self.model_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            expr_out_dim = hidden_dim
        if use_pert_emb:
            self.pert_encoder = PerturbationEmbedding(pert_dim, pert_emb_dim)
            pert_out_dim = pert_emb_dim
        else:
            self.pert_encoder = nn.Sequential(
                nn.Linear(pert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            pert_out_dim = hidden_dim
        self.cross_attention = SingleTokenAttention(
            query_dim=pert_out_dim,
            key_dim=expr_out_dim,
            value_dim=expr_out_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=attention_dropout
        )
        mlp_layers = []
        for _ in range(n_layers):
            mlp_layers.extend([
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(ffn_dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
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
    def forward(self, x_control, pert):
        assert x_control.dim() == 2 and x_control.size(1) == self.input_dim, \
            f"Expected x_control shape [B, {self.input_dim}], got {x_control.shape}"
        x_control_proj = self.input_proj(x_control)
        vae_kl = 0
        vae_recon = None
        if self.use_vae:
            z, mu, logvar = self.vae_encoder(x_control_proj)
            vae_recon = self.vae_decoder(z)
            vae_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            expr_feat = z
        else:
            expr_feat = self.expression_encoder(x_control_proj)
        pert_feat = self.pert_encoder(pert)
        attended_feat = self.cross_attention(pert_feat, expr_feat, expr_feat)
        x_trans = self.mlp(attended_feat)
        fused = self.fusion(x_trans)
        output = self.output(fused)
        return output, vae_recon, vae_kl, x_control_proj
def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1, vae_beta=1.0, max_pert_dim=None):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_control, pert, x_target_delta = batch
        x_control, pert, x_target_delta = x_control.to(device), pert.to(device), x_target_delta.to(device)
        if max_pert_dim is not None and pert.shape[1] < max_pert_dim:
            pad_size = max_pert_dim - pert.shape[1]
            pert = F.pad(pert, (0, pad_size), mode='constant', value=0)
        optimizer.zero_grad()
        output, vae_recon, vae_kl, x_control_proj = model(x_control, pert)
        main_loss = F.mse_loss(output, x_target_delta)
        vae_loss = 0
        if vae_recon is not None:
            vae_loss = F.mse_loss(vae_recon, x_control_proj)
        loss = main_loss + vae_beta * vae_kl + 0.1 * vae_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
def evaluate_model(model, test_loader, device, aux_weight=0.1, vae_beta=1.0, max_pert_dim=None):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    with torch.no_grad():
        for batch in test_loader:
            x_control, pert, x_target_delta = batch
            x_control, pert, x_target_delta = x_control.to(device), pert.to(device), x_target_delta.to(device)
            if max_pert_dim is not None and pert.shape[1] < max_pert_dim:
                pad_size = max_pert_dim - pert.shape[1]
                pert = F.pad(pert, (0, pad_size), mode='constant', value=0)
            output, vae_recon, vae_kl, x_control_proj = model(x_control, pert)
            main_loss = F.mse_loss(output, x_target_delta)
            vae_loss = 0
            if vae_recon is not None:
                vae_loss = F.mse_loss(vae_recon, x_control_proj)
            loss = main_loss + vae_beta * vae_kl + 0.1 * vae_loss
            total_loss += loss.item()
            x_target_abs = x_control + x_target_delta
            x_pred_abs = x_control + output
            r2 = r2_score(x_target_abs.cpu().numpy(), x_pred_abs.cpu().numpy())
            total_r2 += r2
            pearson = np.mean([pearsonr(x_target_abs[i].cpu().numpy(), x_pred_abs[i].cpu().numpy())[0]
                              for i in range(x_target_abs.size(0))])
            total_pearson += pearson
    return {
        'loss': total_loss / len(test_loader),
        'r2': total_r2 / len(test_loader),
        'pearson': total_pearson / len(test_loader)
    }
def standardize_perturbation_encoding(train_dataset, test_dataset=None, test_perturbation_names=None):
    if test_dataset is not None:
        if hasattr(train_dataset, '_pert_encoding_standardized') and hasattr(test_dataset, '_pert_encoding_standardized'):
            if train_dataset._pert_encoding_standardized and test_dataset._pert_encoding_standardized:
                if train_dataset.perturbations.shape[1] == test_dataset.perturbations.shape[1]:
                    return train_dataset.perturbations.shape[1], sorted(list(set(train_dataset.perturbation_names + test_dataset.perturbation_names)))
        test_pert_names = test_dataset.perturbation_names
    elif test_perturbation_names is not None:
        if hasattr(train_dataset, '_pert_encoding_standardized') and train_dataset._pert_encoding_standardized:
            all_pert_names = set(train_dataset.perturbation_names + test_perturbation_names)
            actual_pert_dim = len(all_pert_names)
            if train_dataset.perturbations.shape[1] == actual_pert_dim:
                return actual_pert_dim, sorted(list(all_pert_names))
        test_pert_names = test_perturbation_names
    else:
        test_pert_names = []
    train_pert_dim = train_dataset.perturbations.shape[1]
    all_pert_names = set(train_dataset.perturbation_names + test_pert_names)
    all_pert_names = sorted(list(all_pert_names))
    train_pert_df = pd.DataFrame(train_dataset.adata.obs['perturbation'])
    train_pert_encoded = pd.get_dummies(train_pert_df['perturbation'])
    for pert_name in all_pert_names:
        if pert_name not in train_pert_encoded.columns:
            train_pert_encoded[pert_name] = 0
    train_pert_encoded = train_pert_encoded.reindex(
        columns=all_pert_names, fill_value=0)
    train_dataset.perturbations = train_pert_encoded.values.astype(np.float32)
    for i, pair in enumerate(train_dataset.pairs):
        pert_idx = pair['perturbed_idx']
        train_dataset.pairs[i]['perturbation'] = train_dataset.perturbations[pert_idx]
    if test_dataset is not None:
        test_pert_df = pd.DataFrame(test_dataset.adata.obs['perturbation'])
        test_pert_encoded = pd.get_dummies(test_pert_df['perturbation'])
        for pert_name in all_pert_names:
            if pert_name not in test_pert_encoded.columns:
                test_pert_encoded[pert_name] = 0
        test_pert_encoded = test_pert_encoded.reindex(
            columns=all_pert_names, fill_value=0)
        test_dataset.perturbations = test_pert_encoded.values.astype(np.float32)
        for i, pair in enumerate(test_dataset.pairs):
            pert_idx = pair['perturbed_idx']
            test_dataset.pairs[i]['perturbation'] = test_dataset.perturbations[pert_idx]
        test_dataset._pert_encoding_standardized = True
    actual_pert_dim = len(all_pert_names)
    train_dataset._pert_encoding_standardized = True
    print_log(
        f"Standardized perturbation encoding: {actual_pert_dim} dimensions")
    print_log(f"All perturbation types: {all_pert_names}")
    return actual_pert_dim, all_pert_names
def objective(trial):
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler, common_genes_info
    n_hidden = trial.suggest_categorical('n_hidden', [256, 512, 1024])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    attention_dropout = trial.suggest_float('attention_dropout', 0.1, 0.2)
    ffn_dropout = trial.suggest_float('ffn_dropout', 0.1, 0.2)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    use_vae = trial.suggest_categorical('use_vae', [True, False])
    vae_latent_dim = trial.suggest_categorical('vae_latent_dim', [32, 64])
    vae_hidden_dim = trial.suggest_categorical('vae_hidden_dim', [128, 256])
    use_pert_emb = trial.suggest_categorical('use_pert_emb', [True, False])
    pert_emb_dim = trial.suggest_categorical('pert_emb_dim', [16, 32])
    vae_beta = trial.suggest_float('vae_beta', 0.1, 0.5)
    scaler = StandardScaler()
    train_data = train_adata.X.toarray() if scipy.sparse.issparse(train_adata.X) else train_adata.X
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0
    pca_model = None
    train_dataset = ControlPerturbedDataset(
        train_adata, scaler=scaler, pca_model=None, pca_dim=128,
        fit_pca=False, augment=True, is_train=True, common_genes_info=common_genes_info)
    test_perturbation_names = list(test_adata.obs['perturbation'].unique())
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset=None, test_perturbation_names=test_perturbation_names)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    model = HybridAttentionModel(
        input_dim=n_genes,
        output_dim=n_genes,
        pert_dim=pert_dim,
        hidden_dim=n_hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        use_vae=use_vae,
        vae_latent_dim=vae_latent_dim,
        vae_hidden_dim=vae_hidden_dim,
        use_pert_emb=use_pert_emb,
        pert_emb_dim=pert_emb_dim,
        vae_beta=vae_beta
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=3,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device, vae_beta=vae_beta, max_pert_dim=pert_dim)
        val_metrics = evaluate_model(model, val_loader, device, vae_beta=vae_beta, max_pert_dim=pert_dim)
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_val_loss
def main(gpu_id=None):
    set_global_seed(42)
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler, common_genes_info
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
    train_path = "/datasets/NormanWeissman2019_filtered_train_processed_unseenpert1.h5ad"
    test_path = "/datasets/NormanWeissman2019_filtered_test_processed_unseenpert1.h5ad"
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
    train_dataset = ControlPerturbedDataset(
        train_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )
    train_control_baseline = None
    if hasattr(train_dataset, 'control_baseline') and train_dataset.control_baseline is not None:
        train_control_baseline = train_dataset.control_baseline
        print_log(f"Extracted training set control baseline (shape: {train_control_baseline.shape}) for test set")
    test_dataset = ControlPerturbedDataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info,
        train_control_baseline=train_control_baseline
    )
    print_log("Starting hyperparameter optimization...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print_log(f'Best parameters: {study.best_params}')
    print_log(f'Best loss: {study.best_value}')
    best_params = study.best_params
    print_log("Training final model with best parameters...")
    pca_model = None
    train_dataset = ControlPerturbedDataset(
        train_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )
    train_control_baseline = None
    if hasattr(train_dataset, 'control_baseline') and train_dataset.control_baseline is not None:
        train_control_baseline = train_dataset.control_baseline
        print_log(f"Extracted training set control baseline (shape: {train_control_baseline.shape}) for test set")
    test_dataset = ControlPerturbedDataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info,
        train_control_baseline=train_control_baseline
    )
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    model = HybridAttentionModel(
        input_dim=n_genes,
        output_dim=n_genes,
        pert_dim=pert_dim,
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        use_vae=best_params['use_vae'],
        vae_latent_dim=best_params['vae_latent_dim'],
        vae_hidden_dim=best_params['vae_hidden_dim'],
        use_pert_emb=best_params['use_pert_emb'],
        pert_emb_dim=best_params['pert_emb_dim'],
        vae_beta=best_params['vae_beta']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        betas=(0.9, 0.999)
    )
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=best_params['learning_rate'],
        epochs=200,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    print_log("Starting training...")
    best_loss = float('inf')
    for epoch in range(200):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device,
                                vae_beta=best_params['vae_beta'], max_pert_dim=pert_dim)
        eval_metrics = evaluate_model(model, test_loader, device,
                                      vae_beta=best_params['vae_beta'], max_pert_dim=pert_dim)
        print_log(f'Epoch {epoch+1}/200:')
        print_log(f'Train Loss: {train_loss:.4f}')
        print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
        print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
        print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, f'model_best_{timestamp}.pt')
            print_log(f"Saved best model, loss: {best_loss:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model: Control -> Perturbed Prediction')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
    args = parser.parse_args()
    results_df = main(gpu_id=args.gpu)