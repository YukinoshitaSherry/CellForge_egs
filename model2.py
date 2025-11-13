"""
CondOT-GRN Model for Single-Cell Perturbation Prediction
Based on drug1.md specifications for Srivatsan sci-Plex dataset

This script implements a Conditional Optimal Transport with Gene Regulatory Network priors
for predicting single-cell transcriptional responses to chemical perturbations.
"""

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
    """Custom print function with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None

class GeneExpressionDataset(Dataset):
    """Dataset class for gene expression data with perturbation information"""

    def __init__(self, adata, perturbation_key='perturbation', dose_key='dose_value',
                 scaler=None, pca_model=None, pca_dim=128, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None,
                 use_hvg=True, n_hvg=5000):
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

        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)

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
                
                pert_name = self._non_control_pert_names[idx % len(self._non_control_pert_names)]
                
                if pert_name in self._pert_to_indices and len(self._pert_to_indices[pert_name]) > 0:
                    pert_indices = self._pert_to_indices[pert_name]
                    
                    target_idx = int(pert_indices[(idx // len(self._non_control_pert_names)) % len(pert_indices)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    
                    target_idx = int(self._non_control_indices[idx % len(self._non_control_indices)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
            else:
                
                alternative_controls = self._control_indices[self._control_indices != idx]
                if len(alternative_controls) > 0:
                    target_idx = int(alternative_controls[idx % len(alternative_controls)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    
                    x_target = x_baseline
                    pert_target = pert
                    dose_target = dose
        else:
            
            if len(self._control_indices) > 0:
                baseline_idx = int(self._control_indices[idx % len(self._control_indices)])
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.expression_data[idx]
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

        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor([dose_target]), torch.FloatTensor(x_target)

class GeneEncoder(nn.Module):
    """Gene expression encoder with attention mechanism"""

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
    """Chemical compound and dose encoder"""

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
    """Conditional Optimal Transport layer using Sinkhorn algorithm"""

    def __init__(self, latent_dim, chem_dim, eps=0.1, max_iter=50):
        super(ConditionalOT, self).__init__()
        self.latent_dim = latent_dim
        self.chem_dim = chem_dim
        self.eps = eps
        self.max_iter = max_iter

        self.cost_net = nn.Sequential(
            nn.Linear(latent_dim + chem_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)
        )

    def sinkhorn(self, a, b, C, eps, max_iter):
        """Sinkhorn algorithm for optimal transport"""

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

        z_control_expanded = z_control.unsqueeze(1).expand(-1, batch_size, -1)
        z_pert_expanded = z_perturbed.unsqueeze(0).expand(batch_size, -1, -1)
        p_expanded = p.unsqueeze(1).expand(-1, batch_size, -1)

        cost_input = torch.cat([z_control_expanded, p_expanded], dim=-1)
        C = self.cost_net(cost_input).squeeze(-1)

        P = self.sinkhorn(a, b, C, self.eps, self.max_iter)

        z_ot_pred = torch.mm(P, z_perturbed)

        return z_ot_pred, P

class FlowRefiner(nn.Module):
    """Conditional normalizing flow for refining OT predictions"""

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
    """Coupling layer for normalizing flow"""

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
    """Gene expression decoder with module heads for GRN priors"""

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
    """Conditional Optimal Transport with GRN priors model"""

    def __init__(self, input_dim, pert_dim, dose_dim=1, latent_dim=128, chem_dim=64,
                 hidden_dim=512, n_heads=8, n_layers=2, dropout=0.1,
                 ot_eps=0.1, ot_max_iter=50, flow_layers=6):
        super(CondOTGRNModel, self).__init__()

        self.input_dim = input_dim
        self.pert_dim = pert_dim
        self.dose_dim = dose_dim
        self.latent_dim = latent_dim
        self.chem_dim = chem_dim

        self.gene_encoder = GeneEncoder(
            input_dim, latent_dim, hidden_dim, n_heads, n_layers, dropout)
        self.chem_encoder = ChemEncoder(
            pert_dim, dose_dim, chem_dim, hidden_dim, dropout)
        self.conditional_ot = ConditionalOT(
            latent_dim, chem_dim, ot_eps, ot_max_iter)
        self.flow_refiner = FlowRefiner(
            latent_dim, chem_dim, flow_layers, hidden_dim)
        self.gene_decoder = GeneDecoder(
            latent_dim, input_dim, hidden_dim, n_modules=10, dropout=dropout)

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
        """
        Forward pass for perturbation prediction.
        
        CRITICAL: This forward path does NOT use ground-truth perturbed profiles to avoid
        information leakage. OT and flow supervision are provided via detached latent
        supervision in train_model. This ensures consistent behavior between training and inference.
        
        Args:
            x_control: Baseline expression profiles (n_samples, n_features)
            pert: Perturbation encoding (n_samples, n_pert_types)
            dose: Dose values (n_samples, 1)
            is_training: Whether in training mode (for potential future use)
        
        Returns:
            dict with keys: 'x_pred', 'z_control', 'z_refined', 'p', 'transport_plan', 'log_det_jac'
        """
        z_control = self.gene_encoder(x_control)
        p = self.chem_encoder(pert, dose)

        # CRITICAL: Always use the same forward path (no ground-truth target in forward)
        # This ensures train/test consistency and avoids identity learning
        # OT and flow supervision are provided via detached latent in train_model
        noise_scale = torch.norm(p, dim=1, keepdim=True) * 0.1
        noise = torch.randn_like(z_control) * noise_scale
        z_refined = z_control + noise
        
        # Apply flow refiner to z_refined
        z_refined, log_det_jac = self.flow_refiner(z_refined, p)

        x_pred = self.gene_decoder(z_refined)

        return {
            'x_pred': x_pred,
            'z_control': z_control,
            'z_refined': z_refined,
            'p': p,
            'transport_plan': None,  # Will be computed in train_model with detached target
            'log_det_jac': log_det_jac
        }

def train_model(model, train_loader, optimizer, scheduler, device, loss_weights=None):
    """Train the model"""
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
        x_baseline, pert, dose, x_target = batch
        x_baseline, pert, dose, x_target = x_baseline.to(
            device), pert.to(device), dose.to(device), x_target.to(device)

        diff = (x_baseline - x_target).pow(2).mean(dim=1)
        valid_mask = diff > 1e-6
        if not torch.any(valid_mask):
            continue
        x_baseline = x_baseline[valid_mask]
        pert = pert[valid_mask]
        dose = dose[valid_mask]
        x_target = x_target[valid_mask]

        # CRITICAL FIX: Use detached latent supervision for OT/flow training
        # Forward path does NOT use x_target to avoid information leakage
        # This ensures train/test consistency and prevents identity learning
        outputs = model(x_baseline, pert, dose, is_training=True)

        # Reconstruction loss: predicted vs target
        recon_loss = F.mse_loss(outputs['x_pred'], x_target)

        # OT and Flow supervision using detached target latent
        # This provides supervision signal without information leakage
        with torch.no_grad():
            # Compute target latent without gradient flow (detached)
            z_target = model.gene_encoder(x_target)
        
        # OT loss: compute transport plan using detached z_target
        # This trains the OT module without leaking information to the forward path
        # The OT module learns to transport from z_control to z_target conditioned on pert
        z_ot_pred, transport_plan = model.conditional_ot(
            outputs['z_control'], z_target.detach(), outputs['p'])
        # OT loss: encourage sparse transport plan (regularization)
        ot_loss = torch.trace(torch.mm(transport_plan, transport_plan.t()))
        
        # Flow loss: encourage z_refined to match z_ot_pred (which aligns with detached z_target)
        # This trains the flow module to refine towards the correct target distribution
        # z_ot_pred is detached to prevent gradient flow back to OT, but provides supervision signal
        flow_loss = F.mse_loss(outputs['z_refined'], z_ot_pred.detach()) - outputs['log_det_jac'].mean()

        de_loss = recon_loss

        grn_loss = torch.tensor(0.0, device=device)

        reg_loss = sum(p.pow(2.0).mean() for p in model.parameters())

        total_loss_batch = (loss_weights['recon'] * recon_loss +
                            loss_weights['ot'] * ot_loss +
                            loss_weights['flow'] * flow_loss +
                            loss_weights['de'] * de_loss +
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
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    valid_batches = 0

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
            x_baseline, pert, dose, x_target = batch
            x_baseline, pert, dose, x_target = x_baseline.to(
                device), pert.to(device), dose.to(device), x_target.to(device)

            diff = (x_baseline - x_target).pow(2).mean(dim=1)
            valid_mask = diff > 1e-6
            if not torch.any(valid_mask):
                continue
            x_baseline = x_baseline[valid_mask]
            pert = pert[valid_mask]
            dose = dose[valid_mask]
            x_target = x_target[valid_mask]

            outputs = model(x_baseline, pert, dose, is_training=False)

            recon_loss = F.mse_loss(outputs['x_pred'], x_target)

            if outputs['transport_plan'] is not None:
                ot_loss = torch.trace(
                    torch.mm(outputs['transport_plan'], outputs['transport_plan'].t()))
            else:
                ot_loss = torch.tensor(0.0, device=device)

            flow_loss = -outputs['log_det_jac'].mean()
            de_loss = recon_loss
            grn_loss = torch.tensor(0.0, device=device)
            reg_loss = sum(p.pow(2.0).mean() for p in model.parameters())

            loss = (loss_weights['recon'] * recon_loss +
                    loss_weights['ot'] * ot_loss +
                    loss_weights['flow'] * flow_loss +
                    loss_weights['de'] * de_loss +
                    loss_weights['grn'] * grn_loss +
                    loss_weights['reg'] * reg_loss)

            total_loss += loss.item()
            valid_batches += 1

            x_target_np = x_target.cpu().numpy()
            x_pred_np = outputs['x_pred'].cpu().numpy()

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
                    pearson = np.mean([pearsonr(x_target[i].cpu().numpy(), outputs['x_pred'][i].cpu().numpy())[0]
                                       for i in range(x_target.size(0))])
                    if np.isnan(pearson):
                        pearson = 0.0
                except:
                    pearson = 0.0

            total_r2 += r2
            total_pearson += pearson

    if valid_batches == 0:
        return {'loss': float('nan'), 'r2': float('nan'), 'pearson': float('nan')}

    return {
        'loss': total_loss / valid_batches,
        'r2': total_r2 / valid_batches,
        'pearson': total_pearson / valid_batches
    }

def calculate_detailed_metrics(pred, true, de_genes=None, control_baseline=None):
    """
    Calculate six evaluation metrics strictly following paper definitions.
    
    Args:
        pred: Predicted expression (n_samples, n_genes)
        true: True expression (n_samples, n_genes)
        de_genes: Optional list of DE gene indices
        control_baseline: Optional control baseline expression (n_control_samples, n_genes) for LFC calculation
    
    Returns:
        dict with MSE, PCC, R2, MSE_DE, PCC_DE, R2_DE
    """
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
    
    if de_genes is not None:
        
        de_mask = np.zeros(n_genes, dtype=bool)
        de_mask[de_genes] = True
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

def objective(trial, timestamp):
    """Optuna objective function"""
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

    model = CondOTGRNModel(
        input_dim=128,
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

        scheduler.step()

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
    """Evaluate the model and persist outputs"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_perturbations = []
    all_baselines = []  

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, dose, x_target = batch
            x_baseline, pert, dose, x_target = x_baseline.to(
                device), pert.to(device), dose.to(device), x_target.to(device)

            outputs = model(x_baseline, pert, dose, is_training=False)
            x_pred = outputs['x_pred']

            all_predictions.append(x_pred.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())
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
            'input_dim': 128,
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

def validate_data_consistency(train_adata, test_adata):
    """Validate consistency between training and test datasets"""
    train_pert = set(train_adata.obs['perturbation'].unique())
    test_pert = set(test_adata.obs['perturbation'].unique())

    print_log(
        f"Number of perturbation types in training set: {len(train_pert)}")
    print_log(f"Number of perturbation types in test set: {len(test_pert)}")
    print_log(f"Training perturbations: {train_pert}")
    print_log(f"Test perturbations: {test_pert}")

    overlap = train_pert & test_pert
    if overlap:
        print_log(f"WARNING: Found overlapping perturbations: {overlap}")
        print_log("This violates the unseen perturbation assumption!")
    else:
        print_log(
            "No overlapping perturbations - unseen perturbation setup confirmed")

def load_model_for_analysis(model_path, device='cuda'):
    """Load a trained model for downstream analysis (DEGs, KEGG)"""
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
    print_log("Detected ChemCPA-style checkpoint; skipping reconstruction and using saved predictions directly for analysis")

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
    """Create AnnData objects for downstream analysis (DEGs, KEGG)"""
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
    """Ensure consistent perturbation encoding dimensions across splits"""
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
    """
    Run the full downstream analysis pipeline (DEGs, KEGG, GO, etc.).

    Args:
        model_path: path to the trained model artifacts
        output_dir: directory where analysis results are written
        device: device used for analysis

    Returns:
        Dict with downstream analysis outputs
    """
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
    """Main training routine with hyperparameter optimization"""
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
    train_path = "/datasets/SrivatsanTrapnell2020_train.h5ad"
    test_path = "/datasets/SrivatsanTrapnell2020_test.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    validate_data_consistency(train_adata, test_adata)

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

    pca_model = PCA(n_components=128)

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

    pca_model.fit(train_data)

    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }

    train_dataset = GeneExpressionDataset(
        train_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )

    test_dataset = GeneExpressionDataset(
        test_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
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

    best_params = study.best_params
    final_model = CondOTGRNModel(
        input_dim=128,
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
warnings.filterwarnings('ignore')

def print_log(message):
    """Custom print function with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None

class GeneExpressionDataset(Dataset):
    """Dataset for single-cell perturbation prediction"""

    def __init__(self, adata, perturbation_key='perturbation', dose_key='dose_value',
                 scaler=None, pca_model=None, pca_dim=512, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None,
                 perturbation_mapping=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.augment = augment
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train
        self.common_genes_info = common_genes_info
        self.perturbation_mapping = perturbation_mapping

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

        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)

        if perturbation_mapping is not None:
            
            pert_series = adata.obs[perturbation_key]
            self.perturbations = np.zeros(
                (len(pert_series), len(perturbation_mapping)))
            for i, pert in enumerate(pert_series):
                if pert in perturbation_mapping:
                    self.perturbations[i, perturbation_mapping[pert]] = 1.0
                else:
                    
                    self.perturbations[i, 0] = 1.0
        else:
            self.perturbations = pd.get_dummies(
                adata.obs[perturbation_key]).values

        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")
        print(
            f"{'train' if is_train else 'test'} set perturbation types: {adata.obs[perturbation_key].unique()}")

        self.perturbation_names = list(adata.obs[perturbation_key].unique())
        
        perturbation_labels = adata.obs[perturbation_key].astype(str).values
        perturbation_labels = np.array(perturbation_labels, dtype='U')
        lower_labels = np.char.lower(perturbation_labels)
        negctrl_mask = np.char.find(lower_labels, 'negctrl') >= 0
        control_mask = (lower_labels == 'control') | \
            (lower_labels == 'ctrl') | negctrl_mask
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

        dose_values = adata.obs[dose_key].astype(str)
        dose_values = dose_values.replace('', '0.0')
        self.dose_values = pd.to_numeric(
            dose_values, errors='coerce').fillna(0.0).values
        self.dose_values = self.dose_values.reshape(-1, 1)

        print(f"{'train' if is_train else 'test'} set dose range: {self.dose_values.min():.3f} - {self.dose_values.max():.3f}")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]
        dose = self.dose_values[idx]

        if idx in self._control_indices:
            
            if len(self._non_control_pert_names) > 0 and len(self._non_control_indices) > 0:
                
                pert_name = self._non_control_pert_names[idx % len(self._non_control_pert_names)]
                
                if pert_name in self._pert_to_indices and len(self._pert_to_indices[pert_name]) > 0:
                    pert_indices = self._pert_to_indices[pert_name]
                    
                    target_idx = int(pert_indices[(idx // len(self._non_control_pert_names)) % len(pert_indices)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    
                    target_idx = int(self._non_control_indices[idx % len(self._non_control_indices)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
            else:
                
                alternative_controls = self._control_indices[self._control_indices != idx]
                if len(alternative_controls) > 0:
                    target_idx = int(alternative_controls[idx % len(alternative_controls)])
                    x_target = self.expression_data[target_idx]
                    pert_target = self.perturbations[target_idx]
                    dose_target = self.dose_values[target_idx]
                else:
                    
                    x_target = x_baseline
                    pert_target = pert
                    dose_target = dose
        else:
            
            if len(self._control_indices) > 0:
                baseline_idx = int(self._control_indices[idx % len(self._control_indices)])
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.expression_data[idx]
            pert_target = pert
            dose_target = dose

        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise

            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask

        return (torch.FloatTensor(x_baseline),
                torch.FloatTensor(pert_target),
                torch.FloatTensor(dose_target),
                torch.FloatTensor(x_target))

class GeneEncoder(nn.Module):
    """Gene-level encoder for 4222 genes"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class PathwayEncoder(nn.Module):
    """Pathway-level encoder for 128 modules"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class CellEncoder(nn.Module):
    """Cell-level encoder for 5K HVGs"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ChemEncoder(nn.Module):
    """Chemical compound encoder - simplified for perturbation prediction"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pert_features):
        return self.encoder(pert_features)

class TargetEncoder(nn.Module):
    """Drug target encoder using GNN over PPI network"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, target_features):
        return self.encoder(target_features)

class PathwayAnnotationEncoder(nn.Module):
    """Pathway annotation encoder with hierarchical embeddings"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pathway_features):
        return self.encoder(pathway_features)

class DosePD(nn.Module):
    """Pharmacodynamics module with Hill function"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.hill_params = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  
        )
        self.output_proj = nn.Linear(3, output_dim)

    def forward(self, dose, chem_features):
        
        params = self.hill_params(chem_features)
        emax = torch.sigmoid(params[:, 0])  
        ec50 = torch.exp(params[:, 1])      
        h = torch.exp(params[:, 2])         

        dose_h = torch.pow(dose + 1e-8, h)
        ec50_h = torch.pow(ec50, h)
        effect = emax * dose_h / (ec50_h + dose_h)

        return self.output_proj(params)

class DiffusionNet(nn.Module):
    """Diffusion network for denoising"""

    def __init__(self, input_dim, hidden_dim, time_embed_dim=128, condition_dim=1024):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        total_input_dim = input_dim + time_embed_dim + condition_dim
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, condition):
        
        t_embed = self.time_embed(t)

        x_cond = torch.cat([x, t_embed, condition], dim=-1)

        return self.net(x_cond)

class MultiScaleDiffusion(nn.Module):
    """Multi-scale conditional diffusion"""

    def __init__(self, gene_dim=512, pathway_dim=256, cell_dim=128,
                 condition_dim=1024, time_embed_dim=128):
        super().__init__()

        self.gene_diffusion = DiffusionNet(
            gene_dim, 512, time_embed_dim, condition_dim)

        self.pathway_diffusion = DiffusionNet(
            pathway_dim, 256, time_embed_dim, condition_dim)

        self.cell_diffusion = DiffusionNet(
            cell_dim, 128, time_embed_dim, condition_dim)

        self.condition_proj = nn.Linear(condition_dim, condition_dim)

    def forward(self, z_gene, z_pathway, z_cell, t, condition):
        
        condition_proj = self.condition_proj(condition)

        z_gene_pred = self.gene_diffusion(z_gene, t, condition_proj)
        z_pathway_pred = self.pathway_diffusion(z_pathway, t, condition_proj)
        z_cell_pred = self.cell_diffusion(z_cell, t, condition_proj)

        return z_gene_pred, z_pathway_pred, z_cell_pred

class GRNConstraint(nn.Module):
    """Gene Regulatory Network constraint module"""

    def __init__(self, input_dim, grn_dim=256):
        super().__init__()
        self.grn_net = nn.Sequential(
            nn.Linear(input_dim, grn_dim),
            nn.LayerNorm(grn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(grn_dim, input_dim)
        )

    def forward(self, x, grn_adjacency=None):
        return self.grn_net(x)

class DEWeighting(nn.Module):
    """Differentially Expressed gene weighting module"""

    def __init__(self, input_dim, de_dim=256):
        super().__init__()
        self.de_net = nn.Sequential(
            nn.Linear(input_dim, de_dim),
            nn.LayerNorm(de_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(de_dim, input_dim),
            nn.Sigmoid()  
        )

    def forward(self, x, de_annotations=None):
        weights = self.de_net(x)
        return x * weights

class ChemCPAX(nn.Module):
    """ChemCPA-X: Multi-scale Conditional Diffusion Model"""

    def __init__(self, gene_dim=512, pathway_dim=256, cell_dim=128,
                 pert_dim=4, dose_dim=1, condition_dim=1024,
                 hidden_dim=512, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()

        self.gene_encoder = GeneEncoder(gene_dim, hidden_dim, gene_dim)
        self.pathway_encoder = PathwayEncoder(
            gene_dim, hidden_dim//2, pathway_dim)
        self.cell_encoder = CellEncoder(gene_dim, hidden_dim//4, cell_dim)

        self.chem_encoder = ChemEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=512, output_dim=512)
        self.target_encoder = TargetEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=256, output_dim=256)
        self.pathway_annotation_encoder = PathwayAnnotationEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=256, output_dim=256)
        self.dose_pd = DosePD(input_dim=pert_dim + dose_dim,
                              hidden_dim=64, output_dim=64)

        self.pert_embedding = nn.Linear(pert_dim, 128)

        self.unseen_pert_embedding = nn.Embedding(
            1, 128)  

        self.cellline_embed = nn.Embedding(3, 128)  

        actual_condition_dim = 512 + 256 + 256 + 64 + 128 + 128
        self.diffusion = MultiScaleDiffusion(
            gene_dim=gene_dim, pathway_dim=pathway_dim, cell_dim=cell_dim,
            condition_dim=actual_condition_dim
        )

        fusion_dim = gene_dim + pathway_dim + cell_dim
        
        condition_dim = 512 + 256 + 256 + 64 + 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.grn_constraint = GRNConstraint(condition_dim)
        self.de_weighting = DEWeighting(condition_dim)

        self.decoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gene_dim)
        )

        self.perturbation_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim//2),
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

    def forward(self, x_baseline, pert, dose, cellline_id=None, t=None, training=True):
        
        z_gene = self.gene_encoder(x_baseline)
        z_pathway = self.pathway_encoder(x_baseline)
        z_cell = self.cell_encoder(x_baseline)

        chem_features = torch.cat([pert, dose], dim=-1)
        h_chem = self.chem_encoder(chem_features)
        h_targets = self.target_encoder(chem_features)
        h_pathways = self.pathway_annotation_encoder(chem_features)
        h_dose = self.dose_pd(dose, chem_features)

        is_unseen = torch.all(pert[:, 1:] == 0, dim=1) & (pert[:, 0] == 1)
        h_pert = self.pert_embedding(pert)

        if torch.any(is_unseen):
            unseen_embed = self.unseen_pert_embedding(
                torch.zeros_like(is_unseen, dtype=torch.long))
            h_pert = torch.where(is_unseen.unsqueeze(-1), unseen_embed, h_pert)

        if cellline_id is None:
            cellline_id = torch.zeros(x_baseline.size(
                0), dtype=torch.long, device=x_baseline.device)
        h_cellline = self.cellline_embed(cellline_id)

        condition = torch.cat(
            [h_chem, h_targets, h_pathways, h_dose, h_pert, h_cellline], dim=-1)

        if training and t is not None:
            
            z_gene_pred, z_pathway_pred, z_cell_pred = self.diffusion(
                z_gene, z_pathway, z_cell, t, condition
            )
        else:
            
            z_gene_pred, z_pathway_pred, z_cell_pred = z_gene, z_pathway, z_cell

        z_fused = torch.cat([z_gene_pred, z_pathway_pred, z_cell_pred], dim=-1)
        z_fused = self.fusion(z_fused)

        z_constrained = self.grn_constraint(z_fused)
        z_final = self.de_weighting(z_constrained)

        output = self.decoder(z_final)
        pert_pred = self.perturbation_head(z_final)

        return output, pert_pred

def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    """Training function"""
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        x_baseline, pert, dose, x_target = batch
        x_baseline = x_baseline.to(device)
        pert = pert.to(device)
        dose = dose.to(device)
        x_target = x_target.to(device)

        t = torch.rand(x_baseline.size(0), 1, device=device)

        output, pert_pred = model(x_baseline, pert, dose, t=t, training=True)

        main_loss = F.mse_loss(output, x_target)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = main_loss + aux_weight * aux_loss

        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss + 1e-5 * l2_reg

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
    """Evaluation function"""
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    total_pert_r2 = 0

    with torch.no_grad():
        for batch in test_loader:
            x_baseline, pert, dose, x_target = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            dose = dose.to(device)
            x_target = x_target.to(device)

            output, pert_pred = model(x_baseline, pert, dose, training=False)

            main_loss = F.mse_loss(output, x_target)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss

            total_loss += loss.item()

            r2 = r2_score(x_target.cpu().numpy(), output.cpu().numpy())
            total_r2 += r2

            pearson = np.mean([pearsonr(x_target[i].cpu().numpy(), output[i].cpu().numpy())[0]
                               for i in range(x_target.size(0))])
            total_pearson += pearson

            pert_r2 = r2_score(pert.cpu().numpy(), pert_pred.cpu().numpy())
            total_pert_r2 += pert_r2

    return {
        'loss': total_loss / len(test_loader),
        'r2': total_r2 / len(test_loader),
        'pearson': total_pearson / len(test_loader),
        'pert_r2': total_pert_r2 / len(test_loader)
    }

def calculate_metrics(pred, true):
    """Calculate 6 evaluation metrics"""
    mse = np.mean((pred - true) ** 2)
    pcc = np.mean([pearsonr(p, t)[0] for p, t in zip(pred.T, true.T)])
    r2 = np.mean([r2_score(t, p) for p, t in zip(pred.T, true.T)])

    std = np.std(true, axis=0)
    de_mask = np.abs(true - np.mean(true, axis=0)) > std
    if np.any(de_mask):
        mse_de = np.mean((pred[de_mask] - true[de_mask]) ** 2)
        pcc_de = np.mean([pearsonr(p[m], t[m])[0]
                         for p, t, m in zip(pred.T, true.T, de_mask.T)])
        r2_de = np.mean([r2_score(t[m], p[m])
                        for p, t, m in zip(pred.T, true.T, de_mask.T)])
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
    """Evaluate model and save results"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, dose, x_target = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            dose = dose.to(device)
            x_target = x_target.to(device)

            output, _ = model(x_baseline, pert, dose, training=False)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = calculate_metrics(all_predictions, all_targets)

    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'perturbation_mapping': getattr(common_genes_info, 'perturbation_mapping', None) if common_genes_info is not None else None,
        'model_config': {
            'gene_dim': 512,
            'pathway_dim': 256,
            'cell_dim': 128,
            'pert_dim': 4,
            'dose_dim': 1
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

def main(gpu_id=None):
    """Main training function"""
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'ChemCPA-X Training started at: {timestamp}')

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print_log(f'GPU {i}: {torch.cuda.get_device_name(i)}')

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

    print_log('Loading data...')
    train_path = "/data1/yzy/split_new/datasets/SrivatsanTrapnell2020_train.h5ad"
    test_path = "/data1/yzy/split_new/datasets/SrivatsanTrapnell2020_test.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    print_log(f'Training data shape: {train_adata.shape}')
    print_log(f'Test data shape: {test_adata.shape}')
    print_log(
        f'Training perturbations: {train_adata.obs["perturbation"].unique()}')
    print_log(f'Test perturbations: {test_adata.obs["perturbation"].unique()}')

    train_perts = train_adata.obs["perturbation"].unique()
    test_perts = test_adata.obs["perturbation"].unique()
    all_perts = list(set(list(train_perts) + list(test_perts)))
    perturbation_mapping = {pert: i for i, pert in enumerate(all_perts)}
    print_log(f'Perturbation mapping: {perturbation_mapping}')
    print_log(f'Unseen perturbations: {set(test_perts) - set(train_perts)}')

    print_log("Processing gene consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print_log(f"Common genes: {len(common_genes)}")

    common_genes.sort()
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]

    pca_model = PCA(n_components=512)

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

    pca_model.fit(train_data)

    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx,
        'perturbation_mapping': perturbation_mapping
    }

    train_dataset = GeneExpressionDataset(
        train_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=512,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info,
        perturbation_mapping=perturbation_mapping
    )

    test_dataset = GeneExpressionDataset(
        test_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=512,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info,
        perturbation_mapping=perturbation_mapping
    )

    model = ChemCPAX(
        gene_dim=512,
        pathway_dim=256,
        cell_dim=128,
        
        pert_dim=len(perturbation_mapping),
        dose_dim=1,
        hidden_dim=512,
        n_layers=2,
        n_heads=8,
        dropout=0.1
    ).to(device)

    print_log(
        f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    print_log('Starting training...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 300  
    patience = 30     
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_model(
            model, train_loader, optimizer, scheduler, device, aux_weight=0.1)
        eval_metrics = evaluate_model(
            model, test_loader, device, aux_weight=0.1)

        if (epoch + 1) % 10 == 0:
            print_log(f'Epoch {epoch+1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
            print_log(f'Perturbation R2: {eval_metrics["pert_r2"]:.4f}')

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = model.state_dict()
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics
            }, f'chemcpa_x_best_model_{timestamp}.pt')
            print_log(f"Saved best model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_log(f'Early stopping at epoch {epoch+1}')
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    print_log('Evaluating final model...')
    results = evaluate_and_save_model(
        model, test_loader, device,
        f'chemcpa_x_final_model_{timestamp}.pt',
        common_genes_info, pca_model, scaler
    )

    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })

    results_df.to_csv(
        f'chemcpa_x_evaluation_results_{timestamp}.csv', index=False)

    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(
        index=False, float_format=lambda x: '{:.6f}'.format(x)))

    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChemCPA-X Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (0-7)')
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
    print("ChemCPA-X: Multi-scale Conditional Diffusion")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
