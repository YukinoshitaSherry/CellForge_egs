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
import optuna.logging
from IPython.display import display
import time
from datetime import datetime
import argparse
import math
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
class ATACDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, pca_model=None,
                 pca_dim=128, fit_pca=False, augment=False, is_train=True,
                 common_genes_info=None):
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
        perturbation_labels = adata.obs[perturbation_key].astype(str).values
        perturbation_labels = np.array(perturbation_labels, dtype='U')
        lower_labels = np.char.lower(perturbation_labels)
        negctrl_mask = np.char.find(lower_labels, 'negctrl') >= 0
        control_mask = (lower_labels == 'control') | \
            (lower_labels == 'ctrl') | negctrl_mask
        control_indices = np.where(control_mask)[0]
        non_control_indices = np.where(~control_mask)[0]
        self._pert_to_indices = {}
        for pert_name in self.perturbation_names:
            pert_mask = perturbation_labels == pert_name
            pert_indices = np.where(pert_mask)[0]
            pert_indices = pert_indices[~np.isin(pert_indices, control_indices)]
            if len(pert_indices) > 0:
                self._pert_to_indices[pert_name] = pert_indices
        self._non_control_pert_names = [name for name in self.perturbation_names
                                        if name not in ['control', 'Control', 'ctrl', 'Ctrl']
                                        and 'negctrl' not in name.lower()]
        self._control_indices = control_indices
        self._non_control_indices = non_control_indices
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
        self.rng = np.random.RandomState(42)
        if not self.is_train:
            self._create_fixed_pairs_for_test()
    def _create_fixed_pairs_for_test(self):
        n = len(self.adata)
        self.pairs = [None] * n
        for idx in range(n):
            if idx in self._control_indices:
                baseline_idx = idx
                if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                    pert_name = self._non_control_pert_names[idx % len(self._non_control_pert_names)]
                    if pert_name in self._pert_to_indices and len(self._pert_to_indices[pert_name]) > 0:
                        pert_indices = self._pert_to_indices[pert_name]
                        target_idx = int(pert_indices[(idx // len(self._non_control_pert_names)) % len(pert_indices)])
                    else:
                        target_idx = int(self._non_control_indices[idx % len(self._non_control_indices)])
                else:
                    alternative_controls = self._control_indices[self._control_indices != idx]
                    if len(alternative_controls) > 0:
                        target_idx = int(alternative_controls[idx % len(alternative_controls)])
                    else:
                        target_idx = idx
            else:
                if len(self._control_indices) > 0:
                    baseline_idx = int(self._control_indices[idx % len(self._control_indices)])
                else:
                    baseline_idx = idx
                target_idx = idx
            self.pairs[idx] = (baseline_idx, target_idx)
    def __len__(self):
        return len(self.adata)
    def __getitem__(self, idx):
        x_current = self.expression_data[idx]
        pert = self.perturbations[idx]
        if self.is_train:
            if idx in self._control_indices:
                x_baseline = x_current
                if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                    pert_name = self.rng.choice(self._non_control_pert_names)
                    if pert_name in self._pert_to_indices and len(self._pert_to_indices[pert_name]) > 0:
                        pert_indices = self._pert_to_indices[pert_name]
                        target_idx = int(self.rng.choice(pert_indices))
                        x_target = self.original_expression_data[target_idx]
                        pert_target = self.perturbations[target_idx]
                    else:
                        target_idx = int(self.rng.choice(self._non_control_indices))
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
            baseline_idx, target_idx = self.pairs[idx]
            if baseline_idx == idx and idx in self._control_indices:
                x_baseline = x_current
            else:
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.original_expression_data[target_idx]
            if target_idx == idx:
                pert_target = pert
            else:
                pert_target = self.perturbations[target_idx]
        x_target_delta = x_target - x_baseline
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask
        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor(x_target_delta)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if embeddings.shape[-1] != self.dim:
            if embeddings.shape[-1] < self.dim:
                padding = torch.zeros(
                    embeddings.shape[0], self.dim - embeddings.shape[-1], device=device)
                embeddings = torch.cat([embeddings, padding], dim=-1)
            else:
                embeddings = embeddings[:, :self.dim]
        return embeddings
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_conv = nn.Linear(
            in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, train_pert_dim, test_pert_dim, hidden_dim=512,
                 n_layers=4, n_heads=8, dropout=0.1, time_emb_dim=128,
                 diffusion_steps=1000, pert_emb_dim=64,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super(ConditionalDiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
        self.train_pert_dim = train_pert_dim
        self.test_pert_dim = test_pert_dim
        assert train_pert_dim == test_pert_dim, "After standardization, train_pert_dim and test_pert_dim must be equal"
        self.use_bottleneck = use_bottleneck
        self.hidden_dim = ((hidden_dim + n_heads - 1) // n_heads) * n_heads
        self.diffusion_steps = diffusion_steps
        self.time_emb_dim = time_emb_dim
        self.pert_emb_dim = pert_emb_dim
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )
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
        self.model_input_proj = nn.Linear(self.model_in_dim, self.hidden_dim)
        self.pert_encoder = nn.Sequential(
            nn.Linear(train_pert_dim, pert_emb_dim),
            nn.LayerNorm(pert_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cond_proj = nn.Linear(
            pert_emb_dim + time_emb_dim, self.hidden_dim)
        mlp_layers = []
        for _ in range(n_layers):
            mlp_layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.model_in_dim)
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.model_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )
        self.pert_head_shared = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.perturbation_head = nn.Linear(self.hidden_dim//2, train_pert_dim)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self, x, pert, timestep, is_train=True):
        assert x.dim() == 2 and x.size(1) == self.input_dim, \
            f"Expected x shape [B, {self.input_dim}], got {x.shape}"
        batch_size = x.shape[0]
        x_proj = self.input_proj(x)
        time_emb = self.time_embeddings(timestep)
        time_emb = self.time_mlp(time_emb)
        pert_emb = self.pert_encoder(pert)
        cond_emb = torch.cat([pert_emb, time_emb], dim=-1)
        cond_emb = self.cond_proj(cond_emb)
        x_proj_model = self.model_input_proj(x_proj)
        x_cond = x_proj_model + cond_emb
        x_trans = self.mlp(x_cond)
        noise_pred_proj = self.noise_proj(x_trans)
        noise_pred = self.output_head(noise_pred_proj)
        output = self.output_head(noise_pred_proj)  
        x_pert_feat = self.pert_head_shared(x_trans)
        pert_pred = self.perturbation_head(x_pert_feat)
        return noise_pred, pert_pred, output
class DiffusionModel(nn.Module):
    def __init__(self, model, diffusion_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.register_buffer('betas', torch.linspace(
            beta_start, beta_end, diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(
            1 - self.alpha_bars[t]).view(-1, 1)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    def p_sample(self, x, pert, t, is_train=True):
        with torch.no_grad():
            pred_noise, _, _ = self.model(x, pert, t, is_train)
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            beta_t = self.betas[t].view(-1, 1)
            
            
            eps = 1e-8
            sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=eps))
            sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=eps))
            sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=eps))
            
            pred_x_start = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            
            
            if torch.any(torch.isnan(pred_x_start)) or torch.any(torch.isinf(pred_x_start)):
                pred_x_start = torch.clamp(pred_x_start, min=-10.0, max=10.0)
                pred_x_start = torch.where(torch.isnan(pred_x_start) | torch.isinf(pred_x_start), 
                                           x, pred_x_start)
            
            if t[0].item() > 0:
                noise = torch.randn_like(x)
                mean = (pred_x_start * sqrt_alpha_bar_t + x * sqrt_one_minus_alpha_bar_t) / sqrt_alpha_t
                
                
                if torch.any(torch.isnan(mean)) or torch.any(torch.isinf(mean)):
                    mean = torch.clamp(mean, min=-10.0, max=10.0)
                    mean = torch.where(torch.isnan(mean) | torch.isinf(mean), x, mean)
                
                result = mean + torch.sqrt(torch.clamp(beta_t, min=eps)) * noise
                
                
                if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                    result = torch.clamp(result, min=-10.0, max=10.0)
                    result = torch.where(torch.isnan(result) | torch.isinf(result), x, result)
                
                return result
            else:
                return pred_x_start
    def p_sample_loop(self, shape, pert, is_train=True):
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full(
                (shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, pert, t_tensor, is_train)
        return x
    def p_sample_loop_from_baseline(self, x_baseline, pert, is_train=True, noise_scale=0.1):
        device = next(self.parameters()).device
        x = x_baseline + noise_scale * torch.randn_like(x_baseline)
        
        start_t = max(1, self.diffusion_steps // 4)  
        for t in reversed(range(start_t)):
            t_tensor = torch.full(
                (x.shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, pert, t_tensor, is_train)
            
            
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                x = torch.clamp(x, min=-10.0, max=10.0)
                x = torch.where(torch.isnan(x) | torch.isinf(x), x_baseline, x)
        
        return x
    def forward(self, x_start, pert, is_train=True):
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.diffusion_steps,
                          (batch_size,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise, pert_pred, _ = self.model(x_noisy, pert, t, is_train)
        return pred_noise, pert_pred, noise, t
def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        x_baseline, pert, x_target_delta = batch
        x_baseline, pert, x_target_delta = x_baseline.to(
            device), pert.to(device), x_target_delta.to(device)
        x_target_abs = x_baseline + x_target_delta
        pred_noise, pert_pred, noise, t = model(x_target_abs, pert, is_train=True)
        diffusion_loss = F.mse_loss(pred_noise, noise)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = diffusion_loss + aux_weight * aux_loss
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
    valid_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            x_baseline, pert, x_target_delta = batch
            x_baseline, pert, x_target_delta = x_baseline.to(
                device), pert.to(device), x_target_delta.to(device)
            x_pred_abs = model.p_sample_loop_from_baseline(x_baseline, pert, is_train=False)
            
            
            if torch.any(torch.isnan(x_pred_abs)) or torch.any(torch.isinf(x_pred_abs)):
                
                x_pred_abs = x_baseline.clone()
            
            x_target_abs = x_baseline + x_target_delta
            diffusion_loss = F.mse_loss(x_pred_abs, x_target_abs)
            
            
            if torch.isnan(diffusion_loss) or torch.isinf(diffusion_loss):
                continue  
            
            t_dummy = torch.randint(0, model.diffusion_steps, (x_baseline.shape[0],), device=device)
            _, pert_pred, _ = model.model(x_pred_abs, pert, t_dummy, is_train=False)
            
            if pert_pred.shape[1] == pert.shape[1]:
                aux_loss = F.mse_loss(pert_pred, pert)
                
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    aux_loss = torch.tensor(0.0, device=device)
            else:
                aux_loss = torch.tensor(0.0, device=device)
            
            loss = diffusion_loss + aux_weight * aux_loss
            
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue  
            
            total_loss += loss.item()
            valid_batches += 1
            all_targets.append(x_target_abs.cpu().numpy())
            all_predictions.append(x_pred_abs.cpu().numpy())
            if pert_pred.shape[1] == pert.shape[1]:
                all_perts.append(pert.cpu().numpy())
                all_pert_preds.append(pert_pred.cpu().numpy())
    if valid_batches == 0:
        
        return {
            'loss': float('inf'),
            'r2': 0.0,
            'pearson': 0.0,
            'pert_r2': 0.0
        }
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    
    all_predictions = np.nan_to_num(all_predictions, nan=0.0, posinf=10.0, neginf=-10.0)
    all_targets = np.nan_to_num(all_targets, nan=0.0, posinf=10.0, neginf=-10.0)
    
    r2 = r2_score(all_targets, all_predictions)
    if np.isnan(r2) or np.isinf(r2):
        r2 = 0.0
    pred_flat = all_predictions.flatten()
    true_flat = all_targets.flatten()
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        pearson = pearsonr(true_flat, pred_flat)[0]
        if np.isnan(pearson) or np.isinf(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    if len(all_perts) > 0:
        all_perts = np.concatenate(all_perts, axis=0)
        all_pert_preds = np.concatenate(all_pert_preds, axis=0)
        
        all_pert_preds = np.nan_to_num(all_pert_preds, nan=0.0, posinf=1.0, neginf=-1.0)
        pert_r2 = r2_score(all_perts, all_pert_preds)
        if np.isnan(pert_r2) or np.isinf(pert_r2):
            pert_r2 = 0.0
    else:
        pert_r2 = 0.0
    return {
        'loss': total_loss / valid_batches if valid_batches > 0 else float('inf'),
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
def evaluate_and_save_model(model, test_loader, device, save_path='atac_diffusion_best.pt',
                            common_genes_info=None, pca_model=None, scaler=None, best_params=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, x_target_delta = batch
            x_baseline, pert, x_target_delta = x_baseline.to(
                device), pert.to(device), x_target_delta.to(device)
            x_pred_abs = model.p_sample_loop_from_baseline(x_baseline, pert, is_train=False)
            x_target_abs = x_baseline + x_target_delta
            all_predictions.append(x_pred_abs.cpu().numpy())
            all_targets.append(x_target_abs.cpu().numpy())
            all_baselines.append(x_baseline.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    control_baseline = all_baselines
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
        'best_params': best_params,
        'model_config': {
            'input_dim': model.input_dim,   
            'output_dim': model.model.output_dim,
            'train_pert_dim': model.model.train_pert_dim,
            'test_pert_dim': model.model.test_pert_dim,
            'hidden_dim': getattr(model.model, 'hidden_dim', 512),
            'n_layers': getattr(model.model, 'n_layers', 4),
            'n_heads': getattr(model.model, 'n_heads', 8),
            'dropout': getattr(model.model, 'dropout', 0.1),
            'diffusion_steps': getattr(model, 'diffusion_steps', 1000),
            'time_emb_dim': model.model.time_emb_dim,
            'pert_emb_dim': model.model.pert_emb_dim
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
    if test_dataset is not None:
        test_pert_df = pd.DataFrame(test_dataset.adata.obs['perturbation'])
        test_pert_encoded = pd.get_dummies(test_pert_df['perturbation'])
        for pert_name in all_pert_names:
            if pert_name not in test_pert_encoded.columns:
                test_pert_encoded[pert_name] = 0
        test_pert_encoded = test_pert_encoded.reindex(
            columns=all_pert_names, fill_value=0)
        test_dataset.perturbations = test_pert_encoded.values.astype(np.float32)
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
        'n_layers': trial.suggest_int('n_layers', 2, 6),
        'n_heads': trial.suggest_int('n_heads', 4, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'time_emb_dim': trial.suggest_int('time_emb_dim', 64, 256),
        'pert_emb_dim': trial.suggest_int('pert_emb_dim', 32, 128),
        'diffusion_steps': trial.suggest_categorical('diffusion_steps', [500, 1000, 2000])
    }
    if test_adata is not None:
        test_perturbation_names = list(test_adata.obs['perturbation'].unique())
    else:
        
        if test_dataset is not None and hasattr(test_dataset, 'perturbation_names'):
            test_perturbation_names = test_dataset.perturbation_names
        else:
            
            test_perturbation_names = train_dataset.perturbation_names if hasattr(train_dataset, 'perturbation_names') else []
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset=None, test_perturbation_names=test_perturbation_names)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    base_model = ConditionalDiffusionModel(
        input_dim=n_genes,
        output_dim=n_genes,
        train_pert_dim=pert_dim,
        test_pert_dim=pert_dim,
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        time_emb_dim=params['time_emb_dim'],
        pert_emb_dim=params['pert_emb_dim']
    )
    model = DiffusionModel(
        base_model, diffusion_steps=params['diffusion_steps']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
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
            x_baseline, pert, x_target_delta = batch
            x_baseline, pert, x_target_delta = x_baseline.to(
                device), pert.to(device), x_target_delta.to(device)
            x_target_abs = x_baseline + x_target_delta
            optimizer.zero_grad()
            pred_noise, pert_pred, noise, t = model(
                x_target_abs, pert, is_train=True)
            diffusion_loss = F.mse_loss(pred_noise, noise)
            pert_loss = F.mse_loss(pert_pred, pert)
            loss = diffusion_loss + params['aux_weight'] * pert_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        val_batches_processed = 0
        max_val_batches = 50
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False):
                if val_batches_processed >= max_val_batches:
                    break
                x_baseline, pert, x_target_delta = batch
                x_baseline, pert, x_target_delta = x_baseline.to(
                    device), pert.to(device), x_target_delta.to(device)
                x_pred_abs = model.p_sample_loop_from_baseline(x_baseline, pert, is_train=False)
                x_target_abs = x_baseline + x_target_delta
                diffusion_loss = F.mse_loss(x_pred_abs, x_target_abs)
                t_dummy = torch.randint(0, model.diffusion_steps, (x_baseline.shape[0],), device=device)
                _, pert_pred, _ = model.model(x_pred_abs, pert, t_dummy, is_train=False)
                pert_loss = F.mse_loss(pert_pred, pert)
                loss = diffusion_loss + params['aux_weight'] * pert_loss
                val_loss += loss.item()
                val_batches_processed += 1
        val_loss /= val_batches_processed
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'best_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        print(
            f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
    return best_val_loss
def main(gpu_id=None):
    set_global_seed(42)
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Training started at: {timestamp}')
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print(
                    f'Warning: Specified GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using specified GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')
    print('Loading ATAC data...')
    train_path = "/datasets/LiscovitchBrauerSanjana2021_train_filtered2.h5ad"
    test_path = "/datasets/LiscovitchBrauerSanjana2021_test_filtered2.h5ad"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    print(f'Train data shape: {train_adata.shape}')
    print(f'Test data shape: {test_adata.shape}')
    print("Processing gene set consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print(f"Train genes: {len(train_genes)}")
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
    train_dataset = ATACDataset(
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
    test_dataset = ATACDataset(
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
    print('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=50)
    pert_dim, _ = standardize_perturbation_encoding(train_dataset, test_dataset)
    best_params = study.best_params
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    base_model = ConditionalDiffusionModel(
        input_dim=n_genes,
        output_dim=n_genes,
        train_pert_dim=pert_dim,
        test_pert_dim=pert_dim,
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        time_emb_dim=best_params['time_emb_dim'],
        pert_emb_dim=best_params['pert_emb_dim']
    )
    final_model = DiffusionModel(
        base_model, diffusion_steps=best_params['diffusion_steps']).to(device)
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
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    print('Training final model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 300
    patience = 25
    patience_counter = 0
    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device,
                                 aux_weight=best_params['aux_weight'])
        val_metrics = evaluate_model(final_model, val_loader, device,
                                      aux_weight=best_params['aux_weight'])
        if (epoch + 1) % 30 == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': val_metrics,
                'best_params': best_params
            }, f'atac_diffusion_best_model_{timestamp}.pt')
            print(f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    final_model.load_state_dict(best_model)
    print('Evaluating final model on test set...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'atac_diffusion_final_model_{timestamp}.pt',
                                      common_genes_info, pca_model, scaler, best_params)
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
        f'atac_diffusion_evaluation_results_{timestamp}.csv', index=False)
    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    return results_df
def load_model_for_prediction(model_path, device='cuda'):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state_dict']
    predictions = checkpoint['predictions']
    targets = checkpoint['targets']
    gene_names = checkpoint['gene_names']
    pca_model = checkpoint['pca_model']
    scaler = checkpoint['scaler']
    model_config = checkpoint['model_config']
    evaluation_results = checkpoint['evaluation_results']
    output_dim = model_config.get('output_dim', model_config['input_dim'])
    base_model = ConditionalDiffusionModel(
        input_dim=model_config['input_dim'],
        output_dim=output_dim,
        train_pert_dim=model_config['train_pert_dim'],
        test_pert_dim=model_config['test_pert_dim'],
        hidden_dim=model_config['hidden_dim'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        dropout=model_config['dropout']
    )
    model = DiffusionModel(
        base_model, diffusion_steps=model_config['diffusion_steps']).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Gene names: {len(gene_names) if gene_names else 'None'}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    return {
        'model': model,
        'predictions': predictions,
        'targets': targets,
        'gene_names': gene_names,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': model_config,
        'evaluation_results': evaluation_results
    }
def create_anndata_for_analysis(predictions, targets, gene_names, perturbation_info=None):
    import anndata as ad
    pred_adata = ad.AnnData(X=predictions)
    pred_adata.var_names = gene_names
    pred_adata.var['feature_types'] = 'ATAC Peaks'
    target_adata = ad.AnnData(X=targets)
    target_adata.var_names = gene_names
    target_adata.var['feature_types'] = 'ATAC Peaks'
    if perturbation_info is not None:
        pred_adata.obs['perturbation'] = perturbation_info
        target_adata.obs['perturbation'] = perturbation_info
    pred_adata.obs['sample_type'] = 'predicted'
    target_adata.obs['sample_type'] = 'observed'
    print(f"Created AnnData objects:")
    print(f"  Predictions: {pred_adata.shape}")
    print(f"  Targets: {target_adata.shape}")
    return pred_adata, target_adata
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ATAC Conditional Diffusion Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU number to use (e.g., --gpu 0 for GPU 0, --gpu 1 for GPU 1)')
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
    print("ATAC Conditional Diffusion Model Training")
    print("=" * 60)
    if args.gpu is not None:
        print(f"Using specified GPU: {args.gpu}")
    else:
        print("Using default GPU settings")
    results_df = main(gpu_id=args.gpu)