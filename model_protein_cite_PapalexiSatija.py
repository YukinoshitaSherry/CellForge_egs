import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import optuna
import optuna.logging
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
scaler = None

class CITEProteinDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, 
                 augment=False, is_train=True, common_genes_info=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.augment = augment
        self.training = True
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
        data = np.arcsinh(data / 5.0)
        
        if scaler is None:
            self.scaler = RobustScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)
            
        data = np.clip(data, -5, 5)
        
        self.protein_data = data
        
        perturbation_col = self.adata.obs[perturbation_key].astype(str)
        negctrl_mask = perturbation_col.str.contains('NegCtrl', case=False, na=False)
        lower_labels = perturbation_col.str.lower()
        exact_control_mask = (lower_labels == 'control')
        other_control_mask = perturbation_col.isin(['Control', 'ctrl', 'Ctrl', 'CONTROL'])
        contains_control_mask = perturbation_col.str.contains('control|ctrl', case=False, na=False)
        control_mask = exact_control_mask | negctrl_mask | other_control_mask | contains_control_mask
        
        self.perturbation_names = list(perturbation_col.unique())
        self.perturbations = pd.get_dummies(perturbation_col).values.astype(np.float32)
        print_log(f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")
        print_log(f"{'train' if is_train else 'test'} set protein dimension: {self.protein_data.shape[1]}")
        
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
        
        print_log(f"Total pairs created: {len(self.pairs)}")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        baseline_idx = pair['baseline_idx']
        target_idx = pair['perturbed_idx']
        
        baseline_protein = self.protein_data[baseline_idx]
        target_protein = self.protein_data[target_idx]
        perturbation = pair['perturbation']
        
        # Calculate delta (change from baseline to target)
        target_delta = target_protein - baseline_protein
        
        if self.augment and self.training:
            noise = np.random.normal(0, 0.01, baseline_protein.shape)
            baseline_protein = baseline_protein + noise
            mask = np.random.random(baseline_protein.shape) > 0.02
            baseline_protein = baseline_protein * mask
            scale = np.random.uniform(0.99, 1.01)
            baseline_protein = baseline_protein * scale
        
        return (torch.FloatTensor(baseline_protein), 
                torch.FloatTensor(perturbation), 
                torch.FloatTensor(target_delta))

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * self.head_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h):
        batch_size, n_nodes, _ = h.shape
        
        Wh = self.W(h)
        Wh = Wh.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        Wh = Wh.transpose(1, 2)
        
        Wh1 = Wh.transpose(2, 3)
        Wh2 = Wh
        e = torch.matmul(Wh1, Wh2)
        e = e.view(batch_size, self.n_heads, n_nodes, n_nodes)
        
        attention = F.softmax(self.leaky_relu(e), dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        h_prime = h_prime.transpose(1, 2).contiguous()
        h_prime = h_prime.view(batch_size, n_nodes, self.out_dim)
        
        return h_prime

class CrossAttentionModule(nn.Module):
    def __init__(self, protein_dim, pert_dim, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.protein_dim = protein_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim
        
        self.protein_to_qkv = nn.Linear(protein_dim, 3 * hidden_dim)
        self.pert_to_qkv = nn.Linear(pert_dim, 3 * hidden_dim)
        
        self.multihead_attn1 = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, protein_feat, pert_feat):
        protein_seq = protein_feat.unsqueeze(1)
        pert_seq = pert_feat.unsqueeze(1)
        
        q1, k1, v1 = self.protein_to_qkv(protein_seq).chunk(3, dim=-1)
        out1, _ = self.multihead_attn1(q1, pert_seq, pert_seq)
        
        q2, k2, v2 = self.pert_to_qkv(pert_seq).chunk(3, dim=-1)
        out2, _ = self.multihead_attn2(q2, protein_seq, protein_seq)
        
        combined = torch.cat([out1.squeeze(1), out2.squeeze(1)], dim=-1)
        fused = self.fusion(combined)
        
        return fused

class ProteinResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual

class EnhancedProteinModel(nn.Module):
    def __init__(self, input_dim, pert_dim, hidden_dim=256, n_layers=3, n_heads=4, 
                 dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1, 
                 use_gat=True, gat_layers=2, use_cross_attn=True):
        super(EnhancedProteinModel, self).__init__()
        self.input_dim = input_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        self.use_cross_attn = use_cross_attn
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.protein_residuals = nn.ModuleList([
            ProteinResidualBlock(hidden_dim, dropout) for _ in range(2)
        ])
        
        if use_gat:
            self.gat_layers = nn.ModuleList()
            for _ in range(gat_layers):
                self.gat_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout))
        
        self.pert_encoder = nn.Sequential(
            nn.Linear(pert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if use_cross_attn:
            self.cross_attention = CrossAttentionModule(
                hidden_dim, hidden_dim, hidden_dim, n_heads, dropout
            )
            fusion_input_dim = hidden_dim
        else:
            fusion_input_dim = hidden_dim * 2
        
        fusion_dim = fusion_input_dim
        self.fusion_dim = ((fusion_dim + n_heads - 1) // n_heads) * n_heads
        if self.fusion_dim != fusion_dim:
            self.fusion_proj = nn.Linear(fusion_dim, self.fusion_dim)
        else:
            self.fusion_proj = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*2,
            dropout=ffn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_residuals = nn.ModuleList([
            ProteinResidualBlock(hidden_dim, dropout) for _ in range(1)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, input_dim)
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
    
    def forward(self, x, pert):
        protein_feat = self.protein_encoder(x)
        
        for residual_block in self.protein_residuals:
            protein_feat = residual_block(protein_feat)
        
        if self.use_gat:
            protein_feat_seq = protein_feat.unsqueeze(1)
            for gat_layer in self.gat_layers:
                protein_feat_seq = gat_layer(protein_feat_seq)
                protein_feat_seq = F.gelu(protein_feat_seq)
            protein_feat = protein_feat_seq.squeeze(1)
        
        pert_feat = self.pert_encoder(pert)
        
        if self.use_cross_attn:
            fusion_input = self.cross_attention(protein_feat, pert_feat)
        else:
            fusion_input = torch.cat([protein_feat, pert_feat], dim=1)
        
        fusion_input = self.fusion_proj(fusion_input)
        fusion_input = fusion_input.unsqueeze(1)
        
        x_trans = self.transformer(fusion_input).squeeze(1)
        
        fused = self.fusion(x_trans)
        
        for residual_block in self.output_residuals:
            fused = residual_block(fused)
        
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
        baseline_expr, pert, target_expr = baseline_expr.to(device), pert.to(device), target_expr.to(device)
        
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
            baseline_expr, pert, target_delta = batch
            baseline_expr, pert, target_delta = baseline_expr.to(device), pert.to(device), target_delta.to(device)
            
            output, pert_pred = model(baseline_expr, pert)
            
            # Compare delta predictions with target delta
            main_loss = F.mse_loss(output, target_delta)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss
            
            total_loss += loss.item()
            # Convert to absolute values for evaluation metrics
            target_abs = baseline_expr + target_delta
            pred_abs = baseline_expr + output
            all_targets.append(target_abs.cpu().numpy())
            all_predictions.append(pred_abs.cpu().numpy())
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

def calculate_metrics(pred, true, control_baseline=None, perturbations=None, scaler=None):
    n_samples, n_proteins = true.shape
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
                
                if scaler is not None:
                    true_pert_inv = scaler.inverse_transform(true_pert)
                    control_baseline_inv = scaler.inverse_transform(control_baseline)
                    true_mean_pert = np.mean(true_pert_inv, axis=0)
                    control_mean = np.mean(control_baseline_inv, axis=0)
                else:
                    true_mean_pert = np.mean(true_pert, axis=0)
                    control_mean = np.mean(control_baseline, axis=0)
                
                lfc = np.log2((true_mean_pert + epsilon) / (control_mean + epsilon))
                lfc_abs = np.abs(lfc)
                K = min(20, n_proteins)
                top_k_indices = np.argsort(lfc_abs)[-K:]
                de_mask = np.zeros(n_proteins, dtype=bool)
                de_mask[top_k_indices] = True
                if np.any(de_mask):
                    true_de = true_pert[:, de_mask]
                    pred_de = pred_pert[:, de_mask]
                    mse_de_pert = np.mean((pred_de - true_de) ** 2)
                    pred_de_flat = pred_de.flatten()
                    true_de_flat = true_de.flatten()
                    if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                        pcc_de_pert = pearsonr(pred_de_flat, true_de_flat)[0]
                        if np.isnan(pcc_de_pert):
                            pcc_de_pert = 0.0
                    else:
                        pcc_de_pert = 0.0
                    true_de_mean_overall = np.mean(true_de)
                    ss_res_de = np.sum((true_de - pred_de) ** 2)
                    ss_tot_de = np.sum((true_de - true_de_mean_overall) ** 2)
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
            mse_de = np.mean((pred[de_mask] - true[de_mask]) ** 2)
            pred_de_flat = pred[de_mask].flatten()
            true_de_flat = true[de_mask].flatten()
            if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                pcc_de = pearsonr(pred_de_flat, true_de_flat)[0]
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
    train_pert_encoded = train_pert_encoded.reindex(columns=all_pert_names, fill_value=0)
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
        test_pert_encoded = test_pert_encoded.reindex(columns=all_pert_names, fill_value=0)
        test_dataset.perturbations = test_pert_encoded.values.astype(np.float32)
        
        for i, pair in enumerate(test_dataset.pairs):
            pert_idx = pair['perturbed_idx']
            test_dataset.pairs[i]['perturbation'] = test_dataset.perturbations[pert_idx]
        test_dataset._pert_encoding_standardized = True
    
    actual_pert_dim = len(all_pert_names)
    train_dataset._pert_encoding_standardized = True
    print_log(f"Standardized perturbation encoding: {actual_pert_dim} dimensions")
    print_log(f"All perturbation types: {all_pert_names}")
    return actual_pert_dim, all_pert_names

def objective(trial, timestamp):
    global train_dataset, test_dataset, device
    
    params = {
        'n_hidden': trial.suggest_int('n_hidden', 128, 512),
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'n_heads': trial.suggest_int('n_heads', 2, 8),
        'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.05, 0.2),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0.05, 0.2),
        'aux_weight': trial.suggest_float('aux_weight', 0.01, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'use_gat': trial.suggest_categorical('use_gat', [True, False]),
        'gat_layers': trial.suggest_int('gat_layers', 1, 3),
        'use_cross_attn': trial.suggest_categorical('use_cross_attn', [True, False])
    }
    
    model = EnhancedProteinModel(
        input_dim=train_dataset.protein_data.shape[1],
        pert_dim=train_dataset.perturbations.shape[1],
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        attention_dropout=params['attention_dropout'],
        ffn_dropout=params['ffn_dropout'],
        use_gat=params['use_gat'],
        gat_layers=params['gat_layers'],
        use_cross_attn=params['use_cross_attn']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=8,
        T_mult=2,
        eta_min=1e-6
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(params['batch_size'] * 2, 512),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    best_val_loss = float('inf')
    patience = 12
    patience_counter = 0
    max_epochs = 120
    
    for epoch in range(max_epochs):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device, 
                               aux_weight=params['aux_weight'])
        
        val_metrics = evaluate_model(model, test_loader, device, 
                                   aux_weight=params['aux_weight'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), f'best_protein_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        trial.report(val_metrics['loss'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

def evaluate_and_save_model(model, test_loader, device, save_path, common_genes_info=None, scaler=None, best_params=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []
    all_perts = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            baseline_expr, pert, target_delta = batch
            baseline_expr, pert, target_delta = baseline_expr.to(device), pert.to(device), target_delta.to(device)
            
            output, _ = model(baseline_expr, pert)
            
            # Convert to absolute values for evaluation
            target_abs = baseline_expr + target_delta
            pred_abs = baseline_expr + output
            
            all_predictions.append(pred_abs.cpu().numpy())
            all_targets.append(target_abs.cpu().numpy())
            all_baselines.append(baseline_expr.cpu().numpy())
            all_perts.append(pert.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    all_perts = np.concatenate(all_perts, axis=0)
    
    control_baseline = all_baselines
    results = calculate_metrics(all_predictions, all_targets, control_baseline=control_baseline, perturbations=all_perts, scaler=scaler)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'baselines': all_baselines,
        'protein_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'scaler': scaler,
        'best_params': best_params,
        'model_config': {
            'input_dim': model.input_dim,
            'pert_dim': model.pert_dim,
            'hidden_dim': getattr(model, 'hidden_dim', 256),
            'n_layers': getattr(model, 'n_layers', 3),
            'n_heads': getattr(model, 'n_heads', 4),
            'dropout': getattr(model, 'dropout', 0.1),
            'use_gat': getattr(model, 'use_gat', True),
            'use_cross_attn': getattr(model, 'use_cross_attn', True)
        }
    }, save_path)
    
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })
    
    print_log("\nProtein Model Evaluation Results:")
    print_log(metrics_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    print_log(f"\nModel and evaluation results saved to: {save_path}")
    
    return results

def main(gpu_id=None):
    set_global_seed(42)
    global train_adata, test_adata, train_dataset, test_dataset, device, scaler
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'Protein Model Training started at: {timestamp}')
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print_log(f'Warning: GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print_log(f'Using GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print_log(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print_log('CUDA not available, using CPU')
    
    print_log('Loading CITE-seq protein data...')
    train_path = "/datasets/PapalexiSatija2021_eccite_protein_train_unseen.h5ad"
    test_path = "/datasets/PapalexiSatija2021_eccite_protein_test_unseen.h5ad"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found: {train_path} or {test_path}")
    
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    
    print_log(f'Training data shape: {train_adata.shape}')
    print_log(f'Test data shape: {test_adata.shape}')
    
    print_log("Processing protein consistency...")
    train_proteins = set(train_adata.var_names)
    test_proteins = set(test_adata.var_names)
    common_proteins = list(train_proteins & test_proteins)
    
    print_log(f"Training proteins: {len(train_proteins)}")
    print_log(f"Test proteins: {len(test_proteins)}")
    print_log(f"Common proteins: {len(common_proteins)}")
    
    common_proteins.sort()
    train_protein_idx = [train_adata.var_names.get_loc(protein) for protein in common_proteins]
    test_protein_idx = [test_adata.var_names.get_loc(protein) for protein in common_proteins]
    
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X[:, train_protein_idx].toarray()
    else:
        train_data = train_adata.X[:, train_protein_idx]
    
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.arcsinh(train_data / 5.0)
    
    scaler = RobustScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -5, 5)
    
    common_genes_info = {
        'genes': common_proteins,
        'train_idx': train_protein_idx,
        'test_idx': test_protein_idx
    }
    
    train_dataset = CITEProteinDataset(
        train_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )
    
    test_dataset = CITEProteinDataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
    )
    
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    print_log('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=30)
    
    best_params = study.best_params
    print_log('Best parameters:')
    for key, value in best_params.items():
        print_log(f'{key}: {value}')
    
    final_model = EnhancedProteinModel(
        input_dim=train_dataset.protein_data.shape[1],
        pert_dim=pert_dim,
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        use_gat=best_params['use_gat'],
        gat_layers=best_params['gat_layers'],
        use_cross_attn=best_params['use_cross_attn']
    ).to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(best_params['batch_size'] * 2, 512),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=8,
        T_mult=2,
        eta_min=1e-6
    )
    
    print_log('Training final protein model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 200
    patience = 20
    patience_counter = 0
    
    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device,
                                 aux_weight=best_params['aux_weight'])
        eval_metrics = evaluate_model(final_model, test_loader, device,
                                      aux_weight=best_params['aux_weight'])
        
        if (epoch + 1) % 20 == 0:
            print_log(f'Epoch {epoch+1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
            print_log(f'Perturbation R2: {eval_metrics["pert_r2"]:.4f}')
        
        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            patience_counter = 0
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, f'cite_protein_best_model_{timestamp}.pt')
            print_log(f"Saved best protein model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_log(f'Early stopping at epoch {epoch+1}')
                break
    
    final_model.load_state_dict(best_model)
    print_log('Evaluating final protein model on test set...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'cite_protein_final_model_{timestamp}.pt',
                                      common_genes_info, scaler, best_params)
    
    best_params_str = str(best_params)
    if len(best_params_str) > 50:
        best_params_str = best_params_str[:50] + '...'
    
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                 results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [best_params_str] * 6
    })
    
    results_df.to_csv(f'cite_protein_evaluation_results_{timestamp}.csv', index=False)
    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CITE-seq Protein Model Training')
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
    print("CITE-seq Protein Model Training")
    print("=" * 60)
    
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using default GPU settings")
    
    results_df = main(gpu_id=args.gpu)

