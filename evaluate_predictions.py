import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import sys
import os

def calculate_metrics_from_matrices(pred, true, perturbations=None, control_baseline=None):
    n_samples, n_features = true.shape
    mse = mean_squared_error(true, pred)
    true_mean_overall = np.mean(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true_mean_overall) ** 2)
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    if np.isnan(r2):
        r2 = 0.0
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        pcc = pearsonr(true_flat, pred_flat)[0]
        if np.isnan(pcc) or np.isinf(pcc):
            pcc = 0.0
    else:
        pcc = 0.0
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
                true_mean_pert = np.mean(true_pert, axis=0)
                
                if control_baseline.ndim == 2:
                    control_mean = np.mean(control_baseline, axis=0)
                else:
                    control_mean = control_baseline
                
                lfc = np.log2((true_mean_pert + epsilon) / (control_mean + epsilon))
                lfc_abs = np.abs(lfc)
                
                K = min(20, n_features)
                top_k_indices = np.argsort(lfc_abs)[-K:]
                if len(top_k_indices) == 0:
                    continue
                
                de_mask = np.zeros(n_features, dtype=bool)
                de_mask[top_k_indices] = True
                if not np.any(de_mask):
                    continue
                true_de = true_pert[:, de_mask]
                pred_de = pred_pert[:, de_mask]
                
                mse_de_pert = mean_squared_error(true_de, pred_de)
                
                pred_de_flat = pred_de.flatten()
                true_de_flat = true_de.flatten()
                if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                    pcc_de_pert = pearsonr(true_de_flat, pred_de_flat)[0]
                    if np.isnan(pcc_de_pert) or np.isinf(pcc_de_pert):
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
            
            mse_de = mean_squared_error(true[de_mask], pred[de_mask])
            
            pred_de_flat = pred[de_mask].flatten()
            true_de_flat = true[de_mask].flatten()
            if len(pred_de_flat) > 1 and np.std(pred_de_flat) > 1e-10 and np.std(true_de_flat) > 1e-10:
                pcc_de = pearsonr(true_de_flat, pred_de_flat)[0]
                if np.isnan(pcc_de) or np.isinf(pcc_de):
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
    
    return {'MSE': mse, 'PCC': pcc, 'R2': r2, 'MSE_DE': mse_de, 'PCC_DE': pcc_de, 'R2_DE': r2_de}

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_predictions.py <pred_path.npy> <true_path.npy> [perturbations_path.npy] [baselines_path.npy]")
        print("  pred_path.npy: numpy array of predictions (n_samples, n_features)")
        print("  true_path.npy: numpy array of true values (n_samples, n_features)")
        print("  perturbations_path.npy: optional, numpy array of perturbations (n_samples, n_perturbations)")
        print("  baselines_path.npy: optional, numpy array of baselines (n_samples, n_features) or (n_features,)")
        sys.exit(1)
    
    pred_path = sys.argv[1]
    true_path = sys.argv[2]
    perturbations_path = sys.argv[3] if len(sys.argv) > 3 else None
    baselines_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    pred = np.load(pred_path)
    true = np.load(true_path)
    perturbations = np.load(perturbations_path) if perturbations_path is not None and os.path.exists(perturbations_path) else None
    baselines = np.load(baselines_path) if baselines_path is not None and os.path.exists(baselines_path) else None
    
    control_baseline = baselines if baselines is not None else None   
    results = calculate_metrics_from_matrices(pred, true, perturbations=perturbations, control_baseline=control_baseline)
    
    mse_de_str = f"{results['MSE_DE']:.6f}" if not np.isnan(results['MSE_DE']) else "N/A"
    pcc_de_str = f"{results['PCC_DE']:.6f}" if not np.isnan(results['PCC_DE']) else "N/A"
    r2_de_str = f"{results['R2_DE']:.6f}" if not np.isnan(results['R2_DE']) else "N/A"
    print(f"MSE: {results['MSE']:.6f}, PCC: {results['PCC']:.6f}, R2: {results['R2']:.6f}, "
          f"MSE_DE: {mse_de_str}, PCC_DE: {pcc_de_str}, R2_DE: {r2_de_str}")
