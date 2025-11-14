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
import warnings
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')
def print_log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None
class GeneExpressionDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', dose_key='dose_value',
                 scaler=None, pca_model=None, pca_dim=128, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None,
                 use_hvg=True, n_hvg=5000, reference_perturbation=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.augment = augment
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train
        self.common_genes_info = common_genes_info
        self.use_hvg = use_hvg
        self.n_hvg = n_hvg
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
        self.expression_data = data  # Full gene space [n_cells, n_genes]
        self.n_genes = data.shape[1]
        self.pca = pca_model if pca_model is not None else None
        self.pca_dim = pca_dim if pca_model is not None else None
        self.perturbations = pd.get_dummies(
            adata.obs[perturbation_key]).values.astype(np.float32)
        print_log(
            f"{'Training' if is_train else 'Test'} set perturbation dim: {self.perturbations.shape[1]}")
        self.perturbation_names = list(adata.obs[perturbation_key].unique())
        perturbation_labels = adata.obs[perturbation_key].astype(str).values
        perturbation_labels = np.array(perturbation_labels, dtype='U')
        lower_labels = np.char.lower(perturbation_labels)
        negctrl_mask = np.char.find(lower_labels, 'negctrl') >= 0
        control_mask = (lower_labels == 'control') | \
            (lower_labels == 'ctrl') | negctrl_mask
        self.reference_perturbation = reference_perturbation
        if len(np.where(control_mask)[0]) == 0 and reference_perturbation is not None:
            print_log("=" * 60)
            print_log(f"No control samples found. Using '{reference_perturbation}' as reference baseline.")
            print_log("Task: reference_perturbation + target_perturbation -> target_expression")
            print_log("=" * 60)
            ref_mask = perturbation_labels == reference_perturbation
            control_mask = control_mask | ref_mask
        self._control_indices = np.where(control_mask)[0]
        self._non_control_indices = np.where(~control_mask)[0]
        self._pert_to_indices = {}
        for pert_name in self.perturbation_names:
            pert_mask = perturbation_labels == pert_name
            pert_indices = np.where(pert_mask)[0]
            pert_indices = pert_indices[~np.isin(pert_indices, self._control_indices)]
            if len(pert_indices) > 0:
                self._pert_to_indices[pert_name] = pert_indices
        self._non_control_pert_names = [name for name in self.perturbation_names 
                                        if name not in ['control', 'Control', 'ctrl', 'Ctrl'] 
                                        and 'negctrl' not in name.lower()]
        if reference_perturbation is not None and reference_perturbation in self._non_control_pert_names:
            self._non_control_pert_names = [name for name in self._non_control_pert_names 
                                            if name != reference_perturbation]
        if len(self._control_indices) == 0:
            if self.is_train:
                raise ValueError(
                    "CRITICAL ERROR: Training set has NO control samples! "
                    "Control samples are REQUIRED for perturbation prediction. "
                    "This model performs control -> perturbed prediction, NOT autoencoder. "
                    "Options:\n"
                    "  1. Ensure your training data includes control samples (labeled as 'control' or containing 'NegCtrl').\n"
                    "  2. Use 'reference_perturbation' parameter to specify a reference perturbation as baseline.\n"
                    "     Example: reference_perturbation='SAHA' (for Srivatsan dataset)"
                )
            else:
                raise ValueError(
                    "CRITICAL ERROR: Test set has NO control samples! "
                    "For unseen perturbation prediction, test set should use training set control baseline. "
                    "Please ensure training set has control samples or use reference_perturbation parameter."
                )
        if dose_key in adata.obs.columns:
            dose_values = pd.to_numeric(adata.obs[dose_key], errors='coerce')
            dose_values = dose_values.fillna(0.0)
            if dose_values.max() > dose_values.min():
                self.dose_values = (dose_values - dose_values.min()) / \
                    (dose_values.max() - dose_values.min())
            else:
                self.dose_values = np.zeros_like(dose_values)
        else:
            self.dose_values = np.zeros(len(adata))
        print_log(
            f"Dataset created: {len(self.adata)} cells, {self.expression_data.shape[1]} features")
    def __len__(self):
        return len(self.adata)
    def __getitem__(self, idx):
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]
        dose = self.dose_values[idx]
        if idx in self._control_indices:
            if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                if self.is_train:
                    pert_name = np.random.choice(self._non_control_pert_names)
                else:
                    pert_name = self._non_control_pert_names[idx % len(self._non_control_pert_names)]
                if pert_name in self._pert_to_indices and len(self._pert_to_indices[pert_name]) > 0:
                    pert_indices = self._pert_to_indices[pert_name]
                    if self.is_train:
                        target_idx = int(np.random.choice(pert_indices))
                    else:
                        target_idx = int(pert_indices[(idx // len(self._non_control_pert_names)) % len(pert_indices)])
                    x_target = self.original_expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    if self.is_train:
                        target_idx = int(np.random.choice(self._non_control_indices))
                    else:
                        target_idx = int(self._non_control_indices[idx % len(self._non_control_indices)])
                    x_target = self.original_expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
            else:
                alternative_controls = self._control_indices[self._control_indices != idx]
                if len(alternative_controls) > 0:
                    if self.is_train:
                        target_idx = int(np.random.choice(alternative_controls))
                    else:
                        target_idx = int(alternative_controls[idx % len(alternative_controls)])
                    x_target = self.original_expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    x_target = self.original_expression_data[idx]
                    pert_target = pert
                    dose_target = dose
        else:
            if len(self._control_indices) > 0:
                if self.is_train:
                    baseline_idx = int(np.random.choice(self._control_indices))
                else:
                    baseline_idx = int(self._control_indices[idx % len(self._control_indices)])
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.original_expression_data[idx]
            pert_target = pert
            dose_target = dose
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask
        x_baseline = x_baseline.astype(np.float32)
        pert_target = pert_target.astype(np.float32)
        dose_target = np.float32(dose_target)
        x_target_delta = x_target - x_baseline
        if self.pca is not None:
            x_target_pca = self.pca.transform(x_target.reshape(1, -1)).flatten()
        else:
            x_target_pca = x_target.copy()  # Use full gene space if no PCA
        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor([dose_target]), torch.FloatTensor(x_target_delta), torch.FloatTensor(x_target_pca)
class GeneEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512, n_heads=8, n_layers=2, dropout=0.1):
        super(GeneEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        mlp_layers = []
        for _ in range(n_layers):
            mlp_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.mlp(x)
        z = self.output_proj(x)
        return z
class ChemEncoder(nn.Module):
    def __init__(self, pert_dim, dose_dim=1, chem_dim=64, hidden_dim=256, dropout=0.1):
        super(ChemEncoder, self).__init__()
        self.pert_dim = pert_dim
        self.dose_dim = dose_dim
        self.chem_dim = chem_dim
        self.pert_encoder = nn.Sequential(
            nn.Linear(pert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dose_encoder = nn.Sequential(
            nn.Linear(dose_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, chem_dim),
            nn.LayerNorm(chem_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, pert, dose):
        pert_feat = self.pert_encoder(pert)
        dose_feat = self.dose_encoder(dose)
        combined = torch.cat([pert_feat, dose_feat], dim=1)
        p = self.fusion(combined)
        return p
class ConditionalOT(nn.Module):
    def __init__(self, latent_dim, chem_dim, eps=0.1, max_iter=50):
        super(ConditionalOT, self).__init__()
        self.latent_dim = latent_dim
        self.chem_dim = chem_dim
        self.eps = eps
        self.max_iter = max_iter
    def sinkhorn(self, a, b, C, eps, max_iter):
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)
        for _ in range(max_iter):
            u = eps * (torch.log(a + 1e-8) -
                       torch.logsumexp((u.unsqueeze(1) + v.unsqueeze(0) - C) / eps, dim=1))
            v = eps * (torch.log(b + 1e-8) -
                       torch.logsumexp((u.unsqueeze(1) + v.unsqueeze(0) - C) / eps, dim=0))
        P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / eps)
        return P
    def forward(self, z_control, z_perturbed, p):
        batch_size = z_control.size(0)
        a = torch.ones(batch_size, device=z_control.device) / batch_size
        b = torch.ones(batch_size, device=z_control.device) / batch_size
        C = torch.cdist(z_control, z_perturbed, p=2) ** 2
        P = self.sinkhorn(a, b, C, self.eps, self.max_iter)
        z_ot_pred = torch.mm(P, z_perturbed)
        return z_ot_pred, P
class FlowRefiner(nn.Module):
    def __init__(self, latent_dim, chem_dim, n_layers=6, hidden_dim=256):
        super(FlowRefiner, self).__init__()
        self.latent_dim = latent_dim
        self.chem_dim = chem_dim
        self.n_layers = n_layers
        self.coupling_layers = nn.ModuleList()
        for i in range(n_layers):
            self.coupling_layers.append(
                CouplingLayer(latent_dim, chem_dim,
                              hidden_dim, mask_type=i % 2)
            )
    def forward(self, z_ot, p):
        z = z_ot
        log_det_jac = 0
        for layer in self.coupling_layers:
            z, log_det = layer(z, p)
            log_det_jac += log_det
        return z, log_det_jac
class CouplingLayer(nn.Module):
    def __init__(self, latent_dim, chem_dim, hidden_dim, mask_type=0):
        super(CouplingLayer, self).__init__()
        self.latent_dim = latent_dim
        self.mask_type = mask_type
        mask = torch.zeros(latent_dim)
        mask[mask_type::2] = 1
        self.register_buffer('mask', mask)
        self.scale_net = nn.Sequential(
            nn.Linear(latent_dim // 2 + chem_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2 + chem_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim // 2)
        )
    def forward(self, z, p):
        masked_z = z * self.mask
        unmasked_z = z * (1 - self.mask)
        z1, z2 = torch.chunk(unmasked_z, 2, dim=1)
        scale_input = torch.cat([z1, p], dim=1)
        scale = self.scale_net(scale_input)
        translate = self.translate_net(scale_input)
        z2_new = z2 * torch.exp(scale) + translate
        z_new = masked_z + torch.cat([z1, z2_new], dim=1) * (1 - self.mask)
        log_det_jac = scale.sum(dim=1)
        return z_new, log_det_jac
class GeneDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512, n_modules=10, dropout=0.1):
        super(GeneDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modules = n_modules
        self.main_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.module_heads = nn.ModuleList()
        module_size = output_dim // n_modules
        for i in range(n_modules):
            start_idx = i * module_size
            end_idx = start_idx + module_size if i < n_modules - 1 else output_dim
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, end_idx - start_idx)
            )
            self.module_heads.append(head)
        self.global_residual = nn.Linear(latent_dim, output_dim)
    def forward(self, z):
        main_feat = self.main_decoder(z)
        module_outputs = []
        for head in self.module_heads:
            module_outputs.append(head(main_feat))
        module_pred = torch.cat(module_outputs, dim=1)
        global_pred = self.global_residual(z)
        output = module_pred + 0.1 * global_pred
        return output
class CondOTGRNModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, dose_dim=1, latent_dim=128, chem_dim=64,
                 hidden_dim=512, n_heads=8, n_layers=2, dropout=0.1,
                 ot_eps=0.1, ot_max_iter=50, flow_layers=6,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super(CondOTGRNModel, self).__init__()
        self.input_dim = input_dim  # n_genes (full gene space)
        self.output_dim = output_dim  # n_genes (full gene space)
        self.model_in_dim = model_in_dim  # Internal working dimension (previously PCA dim)
        self.pert_dim = pert_dim
        self.dose_dim = dose_dim
        self.latent_dim = latent_dim
        self.chem_dim = chem_dim
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
        self.gene_encoder = GeneEncoder(
            self.model_in_dim, latent_dim, hidden_dim, n_heads, n_layers, dropout)
        self.chem_encoder = ChemEncoder(
            pert_dim, dose_dim, chem_dim, hidden_dim, dropout)
        self.conditional_ot = ConditionalOT(
            latent_dim, chem_dim, ot_eps, ot_max_iter)
        self.flow_refiner = FlowRefiner(
            latent_dim, chem_dim, flow_layers, hidden_dim)
        self.gene_decoder = GeneDecoder(
            latent_dim, self.model_in_dim, hidden_dim, n_modules=10, dropout=dropout)
        self.output_head = nn.Sequential(
            nn.Linear(self.model_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
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
    def forward(self, x_control, pert, dose, is_training=True):
        assert x_control.dim() == 2 and x_control.size(1) == self.input_dim, \
            f"Expected x_control shape [B, {self.input_dim}], got {x_control.shape}"
        x_control_proj = self.input_proj(x_control)  # [B, model_in_dim]
        z_control = self.gene_encoder(x_control_proj)
        p = self.chem_encoder(pert, dose)
        noise_scale = torch.norm(p, dim=1, keepdim=True) * 0.1
        noise = torch.randn_like(z_control) * noise_scale
        z_refined = z_control + noise
        z_refined, log_det_jac = self.flow_refiner(z_refined, p)
        x_pred_proj = self.gene_decoder(z_refined)  # [B, model_in_dim]
        x_pred = self.output_head(x_pred_proj)  # [B, n_genes]
        return {
            'x_pred': x_pred,  # Delta prediction in full gene space
            'z_control': z_control,
            'z_refined': z_refined,
            'p': p,
            'transport_plan': None,  
            'log_det_jac': log_det_jac
        }
def train_model(model, train_loader, optimizer, scheduler, device, loss_weights=None):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()
    if loss_weights is None:
        loss_weights = {
            'recon': 1.0,
            'ot': 0.1,
            'flow': 0.1,
            'de': 0.5,
            'grn': 0.01,
            'reg': 1e-5
        }
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_baseline, pert, dose, x_target_delta, x_target_pca = batch
        x_baseline, pert, dose, x_target_delta, x_target_pca = x_baseline.to(
            device), pert.to(device), dose.to(device), x_target_delta.to(device), x_target_pca.to(device)
        outputs = model(x_baseline, pert, dose, is_training=True)
        recon_loss = F.mse_loss(outputs['x_pred'], x_target_delta)
        with torch.no_grad():
            x_target_abs = x_baseline + x_target_delta
            if x_target_abs.size(1) == model.input_dim:
                x_target_proj = model.input_proj(x_target_abs)
            else:
                x_target_proj = x_target_abs
            z_target = model.gene_encoder(x_target_proj)
        z_ot_pred, transport_plan = model.conditional_ot(
            outputs['z_control'], z_target.detach(), outputs['p'])
        cost_matrix = torch.cdist(outputs['z_control'], z_target, p=2) ** 2
        ot_loss = torch.sum(transport_plan * cost_matrix) / x_baseline.size(0)
        flow_loss = F.mse_loss(outputs['z_refined'], z_ot_pred.detach()) + 0.01 * outputs['log_det_jac'].pow(2).mean()
        grn_loss = torch.tensor(0.0, device=device)
        reg_loss = sum(p.pow(2.0).mean() for p in model.parameters())
        total_loss_batch = (loss_weights['recon'] * recon_loss +
                            loss_weights['ot'] * ot_loss +
                            loss_weights['flow'] * flow_loss +
                            loss_weights['grn'] * grn_loss +
                            loss_weights['reg'] * reg_loss)
        total_loss_batch = total_loss_batch / accumulation_steps
        total_loss_batch.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += total_loss_batch.item() * accumulation_steps
    return total_loss / len(train_loader)
def evaluate_model(model, test_loader, device, loss_weights=None):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    if loss_weights is None:
        loss_weights = {
            'recon': 1.0,
            'ot': 0.1,
            'flow': 0.1,
            'de': 0.5,
            'grn': 0.01,
            'reg': 1e-5
        }
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            x_baseline, pert, dose, x_target_delta, x_target_pca = batch
            x_baseline, pert, dose, x_target_delta, x_target_pca = x_baseline.to(
                device), pert.to(device), dose.to(device), x_target_delta.to(device), x_target_pca.to(device)
            outputs = model(x_baseline, pert, dose, is_training=False)
            recon_loss = F.mse_loss(outputs['x_pred'], x_target_delta)
            with torch.no_grad():
                x_target_abs = x_baseline + x_target_delta
                if x_target_abs.size(1) == model.input_dim:
                    x_target_proj = model.input_proj(x_target_abs)
                else:
                    x_target_proj = x_target_abs
                z_target = model.gene_encoder(x_target_proj)
            z_ot_pred, transport_plan = model.conditional_ot(
                outputs['z_control'], z_target.detach(), outputs['p'])
            cost_matrix = torch.cdist(outputs['z_control'], z_target, p=2) ** 2
            ot_loss = torch.sum(transport_plan * cost_matrix) / x_baseline.size(0)
            flow_loss = 0.01 * outputs['log_det_jac'].pow(2).mean()
            grn_loss = torch.tensor(0.0, device=device)
            reg_loss = sum(p.pow(2.0).mean() for p in model.parameters())
            loss = (loss_weights['recon'] * recon_loss +
                    loss_weights['ot'] * ot_loss +
                    loss_weights['flow'] * flow_loss +
                    loss_weights['grn'] * grn_loss +
                    loss_weights['reg'] * reg_loss)
            total_loss += loss.item()
            x_target_abs = x_baseline + x_target_delta
            x_target_np = x_target_abs.cpu().numpy()
            x_pred_abs = x_baseline + outputs['x_pred']
            x_pred_np = x_pred_abs.cpu().numpy()
            if np.any(np.isnan(x_target_np)) or np.any(np.isnan(x_pred_np)) or \
               np.any(np.isinf(x_target_np)) or np.any(np.isinf(x_pred_np)):
                r2 = 0.0
                pearson = 0.0
            else:
                try:
                    r2 = r2_score(x_target_np, x_pred_np)
                    if np.isnan(r2):
                        r2 = 0.0
                except:
                    r2 = 0.0
                try:
                    pearson = np.mean([pearsonr(x_target_abs[i].cpu().numpy(), x_pred_abs[i].cpu().numpy())[0]
                                       for i in range(x_target_abs.size(0))])
                    if np.isnan(pearson):
                        pearson = 0.0
                except:
                    pearson = 0.0
            total_r2 += r2
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
def objective(trial, timestamp):
    global train_dataset, test_dataset, device, pca_model, scaler
    params = {
        'latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256]),
        'chem_dim': trial.suggest_categorical('chem_dim', [32, 64, 128]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768, 1024]),
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'ot_eps': trial.suggest_float('ot_eps', 0.05, 0.2),
        'ot_max_iter': trial.suggest_int('ot_max_iter', 30, 100),
        'flow_layers': trial.suggest_int('flow_layers', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'alpha': trial.suggest_float('alpha', 0.5, 2.0),
        'beta': trial.suggest_float('beta', 0.1, 1.0),
        'gamma': trial.suggest_float('gamma', 0.05, 0.2),
        'eta': trial.suggest_float('eta', 0.1, 1.0)
    }
    while params['hidden_dim'] % params['n_heads'] != 0:
        if params['hidden_dim'] < 1024:
            params['hidden_dim'] += params['n_heads']
        else:
            params['hidden_dim'] = 512
    max_pert_dim, all_pert_names = standardize_perturbation_encoding(
        train_dataset, test_dataset)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    model = CondOTGRNModel(
        input_dim=n_genes,  # Full gene space
        output_dim=n_genes,  # Full gene space  
        pert_dim=max_pert_dim,
        dose_dim=1,
        latent_dim=params['latent_dim'],
        chem_dim=params['chem_dim'],
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        ot_eps=params['ot_eps'],
        ot_max_iter=params['ot_max_iter'],
        flow_layers=params['flow_layers']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
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
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    max_epochs = 100
    loss_weights = {
        'recon': params['alpha'],
        'ot': params['beta'],
        'flow': params['gamma'],
        'de': params['eta'],
        'grn': 0.01,
        'reg': 1e-5
    }
    for epoch in range(max_epochs):
        train_loss = train_model(
            model, train_loader, optimizer, scheduler, device, loss_weights)
        val_metrics = evaluate_model(model, test_loader, device, loss_weights)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'best_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        trial.report(val_metrics['loss'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_val_loss
def evaluate_and_save_model(model, test_loader, device, save_path, common_genes_info=None, pca_model=None, scaler=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_perturbations = []
    all_baselines = []  
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, dose, x_target_delta, x_target_pca = batch
            x_baseline, pert, dose, x_target_delta, x_target_pca = x_baseline.to(
                device), pert.to(device), dose.to(device), x_target_delta.to(device), x_target_pca.to(device)
            outputs = model(x_baseline, pert, dose, is_training=False)
            x_pred_delta = outputs['x_pred']  # Predicted delta
            x_pred_abs = x_baseline + x_pred_delta
            x_target_abs = x_baseline + x_target_delta  # Ground-truth absolute expression
            all_predictions.append(x_pred_abs.cpu().numpy())
            all_targets.append(x_target_abs.cpu().numpy())
            all_perturbations.append(pert.cpu().numpy())
            all_baselines.append(x_baseline.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_perturbations = np.concatenate(all_perturbations, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    control_baseline = all_baselines  
    results = calculate_detailed_metrics(all_predictions, all_targets, control_baseline=control_baseline)
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'perturbations': all_perturbations,
        'baselines': all_baselines,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': {
            'input_dim': model.input_dim,
            'pert_dim': train_dataset.perturbations.shape[1],
            'dose_dim': 1,
            'latent_dim': model.latent_dim,
            'chem_dim': model.chem_dim,
            'hidden_dim': model.gene_encoder.input_proj.out_features,
            'n_layers': len([m for m in model.gene_encoder.mlp if isinstance(m, nn.Linear)]) // 2 if hasattr(model.gene_encoder, 'mlp') else 2,
            'n_heads': 8,  
            'dropout': model.gene_encoder.output_proj[3].p if len(model.gene_encoder.output_proj) > 3 else 0.1,
            'ot_eps': model.conditional_ot.eps,
            'ot_max_iter': model.conditional_ot.max_iter,
            'flow_layers': len(model.flow_refiner.coupling_layers)
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
def load_model_for_analysis(model_path, device='cuda'):
    print_log(f"Loading model from {model_path}")
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state_dict']
    predictions = checkpoint['predictions']
    targets = checkpoint['targets']
    perturbations = checkpoint.get('perturbations', None)
    gene_names = checkpoint['gene_names']
    pca_model = checkpoint['pca_model']
    scaler = checkpoint['scaler']
    model_config = checkpoint['model_config']
    evaluation_results = checkpoint['evaluation_results']
    class DummyModel:
        def __init__(self):
            self.eval = lambda: None
    model = DummyModel()
    print_log(f"Model loaded successfully!")
    print_log(f"Gene names: {len(gene_names) if gene_names else 'None'}")
    print_log(f"Predictions shape: {predictions.shape}")
    print_log(f"Targets shape: {targets.shape}")
    return {
        'model': model,
        'predictions': predictions,
        'targets': targets,
        'perturbations': perturbations,
        'gene_names': gene_names,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': model_config,
        'evaluation_results': evaluation_results
    }
def create_anndata_for_analysis(predictions, targets, gene_names, perturbations=None):
    import anndata as ad
    pred_adata = ad.AnnData(X=predictions)
    pred_adata.var_names = gene_names
    pred_adata.var['feature_types'] = 'Gene Expression'
    target_adata = ad.AnnData(X=targets)
    target_adata.var_names = gene_names
    target_adata.var['feature_types'] = 'Gene Expression'
    if perturbations is not None:
        pred_adata.obs['perturbation'] = perturbations
        target_adata.obs['perturbation'] = perturbations
    pred_adata.obs['sample_type'] = 'predicted'
    target_adata.obs['sample_type'] = 'observed'
    print_log(f"Created AnnData objects:")
    print_log(f"  Predictions: {pred_adata.shape}")
    print_log(f"  Targets: {target_adata.shape}")
    return pred_adata, target_adata
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
def perform_downstream_analysis(model_path: str,
                                output_dir: str = './analysis_results',
                                device: str = 'cuda') -> Dict:
    try:
        from downstream_analysis import analyze_model_results
        return analyze_model_results(model_path, output_dir, device)
    except ImportError:
        print_log(
            "Warning: downstream_analysis module not found. Using basic analysis.")
        results = load_model_for_analysis(model_path, device)
        pred_adata, target_adata = create_anndata_for_analysis(
            results['predictions'],
            results['targets'],
            results['gene_names'],
            results['perturbations']
        )
        return {
            'model_results': results,
            'pred_adata': pred_adata,
            'target_adata': target_adata,
            'message': 'Basic analysis completed. Install downstream_analysis for full functionality.'
        }
def main(gpu_id=None):
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'Training started at: {timestamp}')
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print_log(f'GPU {i}: {torch.cuda.get_device_name(i)}')
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
    print_log('Loading data...')
    train_path = "/disk/disk_20T/yzy/split_new_done/datasets/SrivatsanTrapnell2020_train_filtered2.h5ad"
    test_path = "/disk/disk_20T/yzy/split_new_done/datasets/SrivatsanTrapnell2020_test_filtered2.h5ad"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")
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
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]
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
    pca_model = None
    print_log("=" * 60)
    print_log("CRITICAL: Training in FULL GENE SPACE (no PCA)")
    print_log(f"Gene count: {len(common_genes)}")
    print_log("Model will use input_proj to project to internal working dimension")
    print_log("=" * 60)
    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }
    train_pert_col = train_adata.obs['perturbation'].astype(str)
    train_pert_counts = train_pert_col.value_counts()
    has_control = train_pert_col.str.lower().isin(['control', 'ctrl']).any() or \
                  train_pert_col.str.contains('NegCtrl', case=False, na=False).any()
    reference_pert = None
    if not has_control:
        reference_pert = train_pert_counts.index[0]
        print_log("=" * 60)
        print_log(f"No control samples found. Using '{reference_pert}' as reference baseline.")
        print_log(f"Task: {reference_pert}(baseline) + target_perturbation -> target_expression")
        print_log(f"Training: {reference_pert} -> {', '.join(train_pert_counts.index[1:].tolist())}")
        print_log(f"Testing: {reference_pert} -> Nutlin (unseen perturbation)")
        print_log("=" * 60)
    train_dataset = GeneExpressionDataset(
        train_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=None,  # No PCA - use full gene space
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info,
        reference_perturbation=reference_pert
    )
    test_dataset = GeneExpressionDataset(
        test_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=None,  # No PCA - use full gene space
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info,
        reference_perturbation=reference_pert  # Use same reference for test set
    )
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    print_log('Starting hyperparameter optimization...')
    trials = 50
    with tqdm(total=trials, desc="Hyperparameter Optimization") as pbar:
        def objective_with_progress(trial):
            result = objective(trial, timestamp)
            pbar.update(1)
            pbar.set_postfix({
                'trial': trial.number,
                'value': f"{result:.4f}" if result is not None else "pruned"
            })
            return result
        study.optimize(objective_with_progress, n_trials=trials)
    print_log('Best parameters:')
    for key, value in study.best_params.items():
        print_log(f'{key}: {value}')
    max_pert_dim, all_pert_names = standardize_perturbation_encoding(
        train_dataset, test_dataset)
    n_genes = train_dataset.n_genes
    print_log(f"Model input dim (full genes): {n_genes}, Model output dim (full genes): {n_genes}")
    best_params = study.best_params
    final_model = CondOTGRNModel(
        input_dim=n_genes,  # Full gene space
        output_dim=n_genes,  # Full gene space  
        pert_dim=max_pert_dim,
        dose_dim=1,
        latent_dim=best_params['latent_dim'],
        chem_dim=best_params['chem_dim'],
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        ot_eps=best_params['ot_eps'],
        ot_max_iter=best_params['ot_max_iter'],
        flow_layers=best_params['flow_layers']
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
    print_log('Training final model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 200
    loss_weights = {
        'recon': best_params['alpha'],
        'ot': best_params['beta'],
        'flow': best_params['gamma'],
        'de': best_params['eta'],
        'grn': 0.01,
        'reg': 1e-5
    }
    for epoch in range(max_epochs):
        train_loss = train_model(
            final_model, train_loader, optimizer, scheduler, device, loss_weights)
        eval_metrics = evaluate_model(
            final_model, test_loader, device, loss_weights)
        if (epoch + 1) % 20 == 0:
            print_log(f'Epoch {epoch+1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, f'condot_grn_best_model_{timestamp}.pt')
            print_log(f"Saved best model with loss: {best_loss:.4f}")
    final_model.load_state_dict(best_model)
    print_log('Evaluating final model...')
    results = evaluate_and_save_model(
        final_model, test_loader, device,
        f'condot_grn_final_model_{timestamp}.pt',
        common_genes_info, pca_model, scaler
    )
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [str(best_params)] * 6
    })
    results_df.to_csv(
        f'condot_grn_evaluation_results_{timestamp}.csv', index=False)
    print_log("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False,
          float_format=lambda x: '{:.6f}'.format(x)))
    return results_df
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CondOT-GRN Model Training for Drug Perturbation Prediction')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID to use (0-7, e.g., --gpu 0)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List all available GPUs and exit')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of hyperparameter optimization trials (default: 50)')
    args = parser.parse_args()
    if args.list_gpus:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_log(f'Available GPUs: {gpu_count}')
            for i in range(gpu_count):
                print_log(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print_log('CUDA not available')
        exit(0)
    print_log("=" * 80)
    print_log("CondOT-GRN Model for Single-Cell Drug Perturbation Prediction")
    print_log("=" * 80)
    print_log(f"Epochs: {args.epochs}")
    print_log(f"Hyperparameter optimization trials: {args.trials}")
    if args.gpu is not None:
        print_log(f"Using specified GPU: {args.gpu}")
    else:
        print_log("Using default GPU settings")
    results_df = main(gpu_id=args.gpu)
    print_log("Training completed successfully!")
    print_log("=" * 80)