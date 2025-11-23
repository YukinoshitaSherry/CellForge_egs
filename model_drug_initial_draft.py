import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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

class CouplingLayer(nn.Module):
    def __init__(self, latent_dim, chem_dim, hidden_dim, mask_type=0):
        super().__init__()
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

class FlowRefiner(nn.Module):
    def __init__(self, latent_dim, chem_dim, n_layers=6, hidden_dim=256):
        super().__init__()
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(latent_dim, chem_dim, hidden_dim, mask_type=i % 2)
            for i in range(n_layers)
        ])

    def forward(self, z, p):
        log_det_jac = 0
        for layer in self.coupling_layers:
            z, log_det = layer(z, p)
            log_det_jac += log_det
        return z, log_det_jac

class GeneDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512, n_modules=10, dropout=0.1):
        super().__init__()
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
        module_outputs = [head(main_feat) for head in self.module_heads]
        module_pred = torch.cat(module_outputs, dim=1)
        global_pred = self.global_residual(z)
        output = module_pred + 0.1 * global_pred
        return output

class CondOTGRNModel(nn.Module):
    def __init__(self, input_dim, output_dim, pert_dim, dose_dim=1, latent_dim=128, chem_dim=64,
                 hidden_dim=512, n_heads=8, n_layers=2, dropout=0.1,
                 ot_eps=0.1, ot_max_iter=50, flow_layers=6,
                 model_in_dim=128, bottleneck_dims=(2048, 512), use_bottleneck=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_in_dim = model_in_dim
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
        self.gene_encoder = GeneEncoder(self.model_in_dim, latent_dim, hidden_dim, n_heads, n_layers, dropout)
        self.chem_encoder = ChemEncoder(pert_dim, dose_dim, chem_dim, hidden_dim, dropout)
        self.conditional_ot = ConditionalOT(latent_dim, chem_dim, ot_eps, ot_max_iter)
        self.flow_refiner = FlowRefiner(latent_dim, chem_dim, flow_layers, hidden_dim)
        self.gene_decoder = GeneDecoder(latent_dim, self.model_in_dim, hidden_dim, n_modules=10, dropout=dropout)
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
        assert x_control.dim() == 2 and x_control.size(1) == self.input_dim, f"Expected x_control shape [B, {self.input_dim}], got {x_control.shape}"
        x_control_proj = self.input_proj(x_control)
        z_control = self.gene_encoder(x_control_proj)
        p = self.chem_encoder(pert, dose)
        noise_scale = torch.norm(p, dim=1, keepdim=True) * 0.1
        noise = torch.randn_like(z_control) * noise_scale
        z_refined = z_control + noise
        z_refined, log_det_jac = self.flow_refiner(z_refined, p)
        x_pred_proj = self.gene_decoder(z_refined)
        x_pred_delta = self.output_head(x_pred_proj)
        x_pred = x_control + x_pred_delta
        return {
            'x_pred': x_pred,
            'x_pred_delta': x_pred_delta,
            'z_control': z_control,
            'z_refined': z_refined,
            'p': p,
            'transport_plan': None,
            'log_det_jac': log_det_jac
        }