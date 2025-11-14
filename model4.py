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
from IPython.display import display
import time
from datetime import datetime
import argparse
import warnings
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
class CITERNADataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, pca_model=None,
                 pca_dim=128, fit_pca=False, augment=False, is_train=True, common_genes_info=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
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
        self.perturbation_names = list(adata.obs[perturbation_key].unique())
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")
        self._create_baseline_perturbation_pairs()
    def _create_baseline_perturbation_pairs(self):
        perturbation_col = self.adata.obs[self.perturbation_key].astype(str)
        negctrl_mask = perturbation_col.str.contains('NegCtrl', case=False, na=False)
        lower_labels = perturbation_col.str.lower()
        exact_control_mask = (lower_labels == 'control')
        other_control_mask = perturbation_col.isin(['Control', 'ctrl', 'Ctrl', 'CONTROL'])
        contains_control_mask = perturbation_col.str.contains('control|ctrl', case=False, na=False)
        control_mask = exact_control_mask | negctrl_mask | other_control_mask | contains_control_mask
        control_indices = np.where(control_mask)[0]
        perturbed_mask = ~control_mask
        perturbed_indices = np.where(perturbed_mask)[0]
        print_log(f"Control samples: {len(control_indices)}")
        print_log(f"Perturbed samples: {len(perturbed_indices)}")
        if len(control_indices) == 0:
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
        self.pairs = []
        for i, perturbed_idx in enumerate(perturbed_indices):
            if self.is_train:
                baseline_idx = np.random.choice(control_indices)
            else:
                baseline_idx = control_indices[i % len(control_indices)]
            self.pairs.append({
                'baseline_idx': baseline_idx,
                'perturbed_idx': perturbed_idx,
                'perturbation': self.perturbations[perturbed_idx]
            })
        for i, control_idx in enumerate(control_indices):
            if len(perturbed_indices) > 0:
                if self.is_train:
                    target_idx = np.random.choice(perturbed_indices)
                else:
                    target_idx = perturbed_indices[i % len(perturbed_indices)]
                self.pairs.append({
                    'baseline_idx': control_idx,
                    'perturbed_idx': target_idx,
                    'perturbation': self.perturbations[target_idx]
                })
            else:
                alternative_controls = control_indices[control_indices != control_idx]
                if len(alternative_controls) > 0:
                    if self.is_train:
                        target_idx = np.random.choice(alternative_controls)
                    else:
                        target_idx = alternative_controls[i % len(alternative_controls)]
                    self.pairs.append({
                        'baseline_idx': control_idx,
                        'perturbed_idx': target_idx,
                        'perturbation': self.perturbations[target_idx]
                    })
        print_log(f"Total pairs created: {len(self.pairs)}")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        baseline_expr = self.expression_data[pair['baseline_idx']]
        perturbed_expr = self.original_expression_data[pair['perturbed_idx']]
        perturbation = pair['perturbation']
        x_target_delta = perturbed_expr - baseline_expr
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, baseline_expr.shape)
            baseline_expr = baseline_expr + noise
            mask = np.random.random(baseline_expr.shape) > 0.05
            baseline_expr = baseline_expr * mask
            scale = np.random.uniform(0.95, 1.05)
            baseline_expr = baseline_expr * scale
        return (torch.FloatTensor(baseline_expr),
                torch.FloatTensor(perturbation),
                torch.FloatTensor(x_target_delta))
class PerturbationEmbedding(nn.Module):
    def __init__(self, pert_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(pert_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, pert):
        x = self.embedding(pert)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x
class CITERNATransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, hidden_dim=512, n_layers=3, n_heads=8,
                 dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1,
                 use_pert_emb=True, pert_emb_dim=64,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super(CITERNATransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim
        self.use_pert_emb = use_pert_emb
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
        self.expression_encoder = nn.Sequential(
            nn.Linear(self.model_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
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
        fusion_dim = hidden_dim + pert_out_dim
        self.fusion_dim = ((fusion_dim + n_heads - 1) // n_heads) * n_heads
        if self.fusion_dim != fusion_dim:
            self.fusion_proj = nn.Linear(fusion_dim, self.fusion_dim)
        else:
            self.fusion_proj = nn.Identity()
        mlp_layers = []
        for _ in range(n_layers):
            mlp_layers.extend([
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(ffn_dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
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
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, pert_dim)
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
    def forward(self, baseline_expr, pert):
        assert baseline_expr.dim() == 2 and baseline_expr.size(1) == self.input_dim, \
            f"Expected baseline_expr shape [B, {self.input_dim}], got {baseline_expr.shape}"
        baseline_expr_proj = self.input_proj(baseline_expr)
        expr_feat = self.expression_encoder(baseline_expr_proj)
        pert_feat = self.pert_encoder(pert)
        fusion_input = torch.cat([expr_feat, pert_feat], dim=1)
        fusion_input = self.fusion_proj(fusion_input)
        x_trans = self.mlp(fusion_input)
        fused = self.fusion(x_trans)
        output = self.output(fused)
        pert_pred = self.perturbation_head(fused)
        return output, pert_pred
def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        baseline_expr, pert, target_expr = batch
        baseline_expr, pert, target_expr = baseline_expr.to(
            device), pert.to(device), target_expr.to(device)
        output, pert_pred = model(baseline_expr, pert)
        main_loss = F.mse_loss(output, target_expr)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = main_loss + aux_weight * aux_loss
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(train_loader)
def evaluate_model(model, test_loader, device, aux_weight=0.1):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    all_perts = []
    all_pert_preds = []
    with torch.no_grad():
        for batch in test_loader:
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)
            output, pert_pred = model(baseline_expr, pert)
            main_loss = F.mse_loss(output, target_expr)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss
            total_loss += loss.item()
            all_targets.append(target_expr.cpu().numpy())
            all_predictions.append(output.cpu().numpy())
            all_perts.append(pert.cpu().numpy())
            all_pert_preds.append(pert_pred.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_perts = np.concatenate(all_perts, axis=0)
    all_pert_preds = np.concatenate(all_pert_preds, axis=0)
    r2 = r2_score(all_targets, all_predictions)
    if np.isnan(r2):
        r2 = 0.0
    pred_flat = all_predictions.flatten()
    true_flat = all_targets.flatten()
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        pearson = pearsonr(true_flat, pred_flat)[0]
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    pert_r2 = r2_score(all_perts, all_pert_preds)
    if np.isnan(pert_r2):
        pert_r2 = 0.0
    return {
        'loss': total_loss / len(test_loader),
        'r2': r2,
        'pearson': pearson,
        'pert_r2': pert_r2
    }
def calculate_metrics(pred, true, control_baseline=None):
    n_samples, n_genes = true.shape
    mse = np.mean((pred - true) ** 2)
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        pcc = pearsonr(true_flat, pred_flat)[0]
        if np.isnan(pcc):
            pcc = 0.0
    else:
        pcc = 0.0
    true_mean_overall = np.mean(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true_mean_overall) ** 2)
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    if np.isnan(r2):
        r2 = 0.0
    if control_baseline is not None and control_baseline.shape[0] > 0:
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
            pred_de_flat = pred[:, de_mask].flatten()
            true_de_flat = true[:, de_mask].flatten()
            if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                pcc_de = pearsonr(true_de_flat, pred_de_flat)[0]
                if np.isnan(pcc_de):
                    pcc_de = 0.0
            else:
                pcc_de = 0.0
            true_de_mean_overall = np.mean(true[:, de_mask])
            ss_res_de = np.sum((true[:, de_mask] - pred[:, de_mask]) ** 2)
            ss_tot_de = np.sum((true[:, de_mask] - true_de_mean_overall) ** 2)
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
            mse_de = np.mean((pred[de_mask] - true[de_mask]) ** 2)
            pred_de_flat = pred[de_mask].flatten()
            true_de_flat = true[de_mask].flatten()
            if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                pcc_de = pearsonr(true_de_flat, pred_de_flat)[0]
                if np.isnan(pcc_de):
                    pcc_de = 0.0
            else:
                pcc_de = 0.0
            true_de_mean = np.mean(true[de_mask])
            ss_res_de = np.sum((pred[de_mask] - true[de_mask]) ** 2)
            ss_tot_de = np.sum((true[de_mask] - true_de_mean) ** 2)
            if ss_tot_de > 1e-10:
                r2_de = 1.0 - (ss_res_de / ss_tot_de)
            else:
                r2_de = 0.0
            if np.isnan(r2_de):
                r2_de = 0.0
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
def objective(trial, timestamp):
    global train_dataset, test_dataset, device, pca_model, test_adata
    params = {
        'pca_dim': 128,
        'n_hidden': trial.suggest_int('n_hidden', 256, 1024),
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'n_heads': trial.suggest_int('n_heads', 4, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.2),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0.1, 0.2),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'use_pert_emb': trial.suggest_categorical('use_pert_emb', [True, False]),
        'pert_emb_dim': trial.suggest_int('pert_emb_dim', 32, 128)
    }
    test_perturbation_names = list(test_adata.obs['perturbation'].unique())
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset=None, test_perturbation_names=test_perturbation_names)
    n_genes = train_dataset.n_genes
    print(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    model = CITERNATransformerModel(
        input_dim=n_genes,
        output_dim=n_genes,
        pert_dim=pert_dim,
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        attention_dropout=params['attention_dropout'],
        ffn_dropout=params['ffn_dropout'],
        use_pert_emb=params['use_pert_emb'],
        pert_emb_dim=params['pert_emb_dim']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_subset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    max_epochs = 150
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)
            optimizer.zero_grad()
            output, pert_pred = model(baseline_expr, pert)
            mse_loss = F.mse_loss(output, target_expr)
            pert_loss = F.mse_loss(pert_pred, pert)
            loss = mse_loss + params['aux_weight'] * pert_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                baseline_expr, pert, target_expr = batch
                baseline_expr, pert, target_expr = baseline_expr.to(
                    device), pert.to(device), target_expr.to(device)
                output, pert_pred = model(baseline_expr, pert)
                mse_loss = F.mse_loss(output, target_expr)
                pert_loss = F.mse_loss(pert_pred, pert)
                loss = mse_loss + params['aux_weight'] * pert_loss
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'best_rna_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        if (epoch + 1) % 20 == 0:
            print(
                f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
    return best_val_loss
def evaluate_and_save_model(model, test_loader, device, save_path, common_genes_info=None, pca_model=None, scaler=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)
            output, _ = model(baseline_expr, pert)
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target_expr.cpu().numpy())
            all_baselines.append(baseline_expr.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    control_baseline = None
    results = calculate_metrics(all_predictions, all_targets, control_baseline=control_baseline)
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
            'hidden_dim': getattr(model, 'hidden_dim', 512),
            'n_layers': getattr(model, 'n_layers', 3),
            'n_heads': getattr(model, 'n_heads', 8),
            'dropout': getattr(model, 'dropout', 0.1),
            'use_pert_emb': getattr(model, 'use_pert_emb', True)
        }
    }, save_path)
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    print("\nRNA Model Evaluation Results:")
    print(metrics_df.to_string(index=False,
          float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")
    return results
def main(gpu_id=None):
    set_global_seed(42)
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'RNA Model Training started at: {timestamp}')
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print(f'Warning: GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')
    print('Loading CITE-seq RNA data...')
    train_path = "/datasets/PapalexiSatija2021_eccite_RNA_train.h5ad"
    test_path = "/datasets/PapalexiSatija2021_eccite_RNA_test.h5ad"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    print(f'Training data shape: {train_adata.shape}')
    print(f'Test data shape: {test_adata.shape}')
    print("Processing gene consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print(f"Training genes: {len(train_genes)}")
    print(f"Test genes: {len(test_genes)}")
    print(f"Common genes: {len(common_genes)}")
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
    train_dataset = CITERNADataset(
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
    test_dataset = CITERNADataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=None,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
    )
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    print('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=30)
    print('Best parameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')
    best_params = study.best_params
    n_genes = train_dataset.n_genes
    print(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    final_model = CITERNATransformerModel(
        input_dim=n_genes,
        output_dim=n_genes,
        pert_dim=pert_dim,
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        use_pert_emb=best_params['use_pert_emb'],
        pert_emb_dim=best_params['pert_emb_dim']
    ).to(device)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_subset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    print('Training final RNA model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 200
    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device,
                                 aux_weight=best_params['aux_weight'])
        val_metrics = evaluate_model(final_model, val_loader, device,
                                      aux_weight=best_params['aux_weight'])
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
            print(f'Validation R2 Score: {val_metrics["r2"]:.4f}')
            print(f'Validation Pearson Correlation: {val_metrics["pearson"]:.4f}')
            print(f'Validation Perturbation R2: {val_metrics["pert_r2"]:.4f}')
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': val_metrics,
                'best_params': best_params
            }, f'cite_rna_best_model_{timestamp}.pt')
            print(f"Saved best RNA model with validation loss: {best_loss:.4f}")
    final_model.load_state_dict(best_model)
    print('Evaluating final RNA model on test set...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'cite_rna_final_model_{timestamp}.pt',
                                      common_genes_info, pca_model, scaler)
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [str(best_params)] * 6
    })
    results_df.to_csv(
        f'cite_rna_evaluation_results_{timestamp}.csv', index=False)
    print("\nFinal RNA Model Evaluation Results:")
    display(results_df)
    return results_df
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CITE-seq RNA Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID (0-7)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List available GPUs and exit')
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
    print("CITE-seq RNA Model Training")
    print("=" * 60)
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using default GPU settings")
    results_df = main(gpu_id=args.gpu)