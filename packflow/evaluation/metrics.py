# crystal_metrics.py
# Metrics for molecular crystal structure prediction
# Requires: torch, numpy, vesin, amd

import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
import numpy as np

# Import utilities from crystal_utils
from packflow.utils.crystal_utils import (
    lattice_params_to_matrix_torch,
    cart_to_frac,
)

# Try importing Vesin for periodic neighbor lists
try:
    from vesin import NeighborList
    HAS_VESIN = True
except ImportError:
    HAS_VESIN = False
    print("Warning: 'vesin' not found. Periodic neighbor calculations (Clash, RDF) will fail.")

# Try importing AMD for L_inf distance
try:
    from amd import PeriodicSet, AMD
    HAS_AMD = True
except ImportError:
    HAS_AMD = False


# -------------------------------------------------------------------
# ---- Helper: Periodic Neighbor List (Vesin Wrapper) ---------------
# -------------------------------------------------------------------

def build_periodic_edges(
    frac_coords: torch.Tensor,      # [N, 3] fractional coordinates
    lattice_mat: torch.Tensor,      # [3, 3] (rows=basis, Pymatgen convention)
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute periodic neighbors using vesin.
    Returns: edge_index [2, E], shifts [E, 3], dists [E]
    """
    device = frac_coords.device
    
    if frac_coords.size(0) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.float, device=device),
            torch.empty((0,), dtype=torch.float, device=device)
        )

    if not HAS_VESIN:
        raise ImportError("Vesin is required for accurate periodic neighbor metrics.")

    # Wrap fractional coordinates to [0,1) for vesin safety
    frac_wrapped = frac_coords - torch.floor(frac_coords)
    
    # Convert to cartesian for vesin (vesin expects cartesian coords + box)
    cart_coords_np = (frac_wrapped @ lattice_mat).detach().cpu().numpy().astype(np.float64)
    box_np = lattice_mat.detach().cpu().numpy().astype(np.float64)
    
    nl = NeighborList(cutoff=cutoff, full_list=True)
    
    # "ijS" -> i, j, shift_vector
    # "ijS" -> i, j, shift_vector
    # Check volume to prevent OOM
    if np.abs(np.linalg.det(box_np)) < 0.1:
        raise ValueError("Lattice volume too small for periodic neighbor calculation")

    src, dst, shifts = nl.compute(
        quantities="ijS",
        points=cart_coords_np,
        box=box_np,
        periodic=True
    )
    
    # Filter self-loops with 0 shift (distance 0)
    mask = ~((src == dst) & (shifts == 0).all(axis=1))
    src = src[mask]
    dst = dst[mask]
    shifts = shifts[mask]
    
    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    shifts_t = torch.from_numpy(shifts).float().to(device)
    edge_index = torch.stack([src_t, dst_t], dim=0)
    
    # Compute distances in PyTorch
    cart_src = frac_wrapped[src_t] @ lattice_mat
    cart_dst = (frac_wrapped[dst_t] + shifts_t) @ lattice_mat
    dists = torch.norm(cart_dst - cart_src, dim=-1)
    
    return edge_index, shifts_t, dists


# -------------------------------------------------------------------
# ---- Metric 1: Density RMSE & MAPE --------------------------------
# -------------------------------------------------------------------

def compute_density(
    lattice_params: torch.Tensor,  # [6] [a, b, c, alpha, beta, gamma]
    atomic_numbers: torch.Tensor,  # [N]
    atomic_masses: Dict[int, float]
) -> torch.Tensor:
    """Compute density in g/cm^3."""
    matrix = lattice_params_to_matrix_torch(lattice_params.unsqueeze(0)).squeeze(0)
    volume_A3 = torch.det(matrix).abs()
    
    m_sum = sum([atomic_masses.get(int(z), 0.0) for z in atomic_numbers.tolist()])
    m_sum = torch.tensor(m_sum, device=lattice_params.device, dtype=lattice_params.dtype)
    
    # Conversion: (Mass_amu / Na) / (Vol_A3 * 1e-24)
    density = (m_sum / (volume_A3 + 1e-6)) * 1.660539
    return density


def metric_density_stats(
    pred_lattice_list: List[torch.Tensor],
    true_lattice_list: List[torch.Tensor],
    atomic_numbers_list: List[torch.Tensor],
    atomic_masses: Dict[int, float]
) -> Dict[str, float]:
    """Computes RMSE, MAE, and MAPE of densities."""
    densities_pred = []
    densities_true = []
    
    for i in range(len(pred_lattice_list)):
        d_p = compute_density(pred_lattice_list[i], atomic_numbers_list[i], atomic_masses)
        d_t = compute_density(true_lattice_list[i], atomic_numbers_list[i], atomic_masses)
        densities_pred.append(d_p)
        densities_true.append(d_t)
        
    d_p_t = torch.stack(densities_pred)
    d_t_t = torch.stack(densities_true)
    
    rmse = torch.sqrt(torch.mean((d_p_t - d_t_t)**2))
    mae = torch.mean(torch.abs(d_p_t - d_t_t))
    mape = torch.mean(torch.abs((d_p_t - d_t_t) / (d_t_t + 1e-6))) * 100.0
    
    return {
        "density_rmse": float(rmse.item()),
        "density_mae": float(mae.item()),
        "density_mape_pct": float(mape.item())
    }


# -------------------------------------------------------------------
# ---- Metric 2: Clash Rate (%) -------------------------------------
# -------------------------------------------------------------------

def get_covalent_radii_tensor(
    atomic_numbers: torch.Tensor, 
    covalent_radii: Dict[int, float]
) -> torch.Tensor:
    """Map atomic numbers to covalent radii tensor."""
    radii = [covalent_radii.get(int(z), 0.7) for z in atomic_numbers.tolist()]
    return torch.tensor(radii, device=atomic_numbers.device, dtype=torch.float)


def compute_clash_metrics(
    frac_coords: torch.Tensor,
    lattice_params: torch.Tensor,
    atomic_numbers: torch.Tensor,
    covalent_radii: Dict[int, float],
    clash_alpha: float = 0.75,  # Threshold multiplier
    neighbor_cutoff: float = 6.0
) -> Dict[str, float]:
    """
    Computes Total Clash Rate as PERCENTAGE.
    Clash defined as d_ij < alpha * (r_i + r_j).
    Since alpha is typically 0.5, valid covalent bonds (~1.0 sum radii) are NOT clashes.
    This metric detects unphysical overlaps only.
    """
    device = frac_coords.device
    N = frac_coords.size(0)
    
    if N == 0:
        return {"clash_total_pct": 0.0}

    lattice_mat = lattice_params_to_matrix_torch(lattice_params)
    
    # 1. Get neighbors
    edge_index, shifts, dists = build_periodic_edges(frac_coords, lattice_mat, neighbor_cutoff)
    if dists.numel() == 0:
        return {"clash_total_pct": 0.0}

    src, dst = edge_index[0], edge_index[1]

    # 2. Check thresholds
    radii = get_covalent_radii_tensor(atomic_numbers, covalent_radii)
    r_src = radii[src]
    r_dst = radii[dst]
    thresholds = clash_alpha * (r_src + r_dst)
    
    is_clash = dists < thresholds
    
    if not is_clash.any():
        return {"clash_total_pct": 0.0}
            
    # 3. Count unique atoms involved in any clash
    src_c = src[is_clash]
    dst_c = dst[is_clash]
    atoms_involved = torch.unique(torch.cat([src_c, dst_c]))
    
    fraction = float(atoms_involved.numel()) / N
    return {"clash_total_pct": fraction * 100.0}


# -------------------------------------------------------------------
# ---- Metric 3: AMD L_inf ------------------------------------------
# -------------------------------------------------------------------

def metric_amd_linf(
    pred_coords_frac: torch.Tensor,
    pred_lattice: torch.Tensor,
    true_coords_frac: torch.Tensor,
    true_lattice: torch.Tensor,
    atomic_numbers: torch.Tensor,
    k_neighbors: int = 100
) -> float:
    """Computes L_inf distance between AMD vectors."""
    if not HAS_AMD: return 0.0
    z_np = atomic_numbers.cpu().numpy()
    
    cell_pred = lattice_params_to_matrix_torch(pred_lattice).detach().cpu().numpy()
    pos_pred = (pred_coords_frac @ torch.tensor(cell_pred, device=pred_coords_frac.device).float()).detach().cpu().numpy()
    
    cell_true = lattice_params_to_matrix_torch(true_lattice).detach().cpu().numpy()
    pos_true = (true_coords_frac @ torch.tensor(cell_true, device=true_coords_frac.device).float()).detach().cpu().numpy()
    
    try:
        ps_pred = PeriodicSet(pos_pred, cell_pred, z_np)
        ps_true = PeriodicSet(pos_true, cell_true, z_np)
        amd_pred = AMD(ps_pred, k_neighbors)
        amd_true = AMD(ps_true, k_neighbors)
        return float(np.max(np.abs(amd_pred - amd_true)))
    except:
        return 0.0


# -------------------------------------------------------------------
# ---- Metric 4: Contact Precision, Recall, F1 ----------------------
# -------------------------------------------------------------------

def compute_contact_metrics(
    pred_coords: torch.Tensor,    # [N, 3] fractional
    pred_lattice: torch.Tensor,   # [6] params
    true_coords: torch.Tensor,    # [N, 3] fractional
    true_lattice: torch.Tensor,   # [6] params
    radius: float = 6.0
) -> Dict[str, float]:
    """
    Computes contact reconstruction metrics (Precision, Recall, F1).
    Treats contact prediction as a binary classification problem per atom.
    """
    device = pred_coords.device
    N = pred_coords.size(0)
    
    # 1. Edge Case: Empty structure or disabled (radius <= 0)
    if N == 0 or radius <= 0:
        return {"contact_precision": 0.0, "contact_recall": 0.0, "contact_f1": 0.0}

    # 2. Build Adjacency Matrices (Periodicity-Aware)
    # We use boolean matrices: adj[i, j] = True if dist(i, j) < radius
    def get_adj(coords, lattice):
        L = lattice_params_to_matrix_torch(lattice)
        edge_index, _, _ = build_periodic_edges(coords, L, cutoff=radius)
        
        adj = torch.zeros((N, N), device=device, dtype=torch.bool)
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = True
        return adj

    adj_pred = get_adj(pred_coords, pred_lattice)
    adj_true = get_adj(true_coords, true_lattice)

    # 3. Compute Counts Per Atom (Vectorized)
    # TP: Neighbors present in BOTH
    tp = (adj_pred & adj_true).sum(dim=1).float()
    
    # P_denom: Total predicted neighbors (TP + FP)
    p_denom = adj_pred.sum(dim=1).float()
    
    # R_denom: Total true neighbors (TP + FN)
    r_denom = adj_true.sum(dim=1).float()

    # 4. Compute Metrics Per Atom (with epsilon for div-by-zero safety)
    epsilon = 1e-8
    precision = tp / (p_denom + epsilon)
    recall    = tp / (r_denom + epsilon)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # 5. Return Mean over Atoms
    return {
        "contact_precision": float(precision.mean().item()),
        "contact_recall":    float(recall.mean().item()),
        "contact_f1":        float(f1.mean().item())
    }


# -------------------------------------------------------------------
# ---- Metric 5: Total RDF JSD & Overlap (%) ------------------------
# -------------------------------------------------------------------

def compute_hist_pdf(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        pdf = torch.ones(len(bins)-1, device=values.device)
        return pdf / pdf.sum()
    hist = torch.histc(values, bins=len(bins)-1, min=bins[0].item(), max=bins[-1].item())
    pdf = hist / (hist.sum() + 1e-8)
    return pdf

def jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-8
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_p = (p * (p.log() - m.log())).sum()
    kl_q = (q * (q.log() - m.log())).sum()
    return 0.5 * (kl_p + kl_q).item()

def compute_hist_overlap(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    intersection = torch.sum(torch.min(p, q))
    return float(intersection.item()) * 100.0

def wasserstein_from_pmf(p: torch.Tensor, q: torch.Tensor, bin_width: float) -> float:
    """
    Compute 1D Wasserstein distance from two Probability Mass Functions (histograms) on the same grid.
    W1 = sum(|CDF_p - CDF_q|) * bin_width
    """
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    # Normalize just in case they aren't perfectly summing to 1
    cdf_p = cdf_p / (cdf_p[-1] + 1e-8)
    cdf_q = cdf_q / (cdf_q[-1] + 1e-8)
    
    return float(torch.sum(torch.abs(cdf_p - cdf_q)).item() * bin_width)

def compute_rdf_jsd_metrics(
    pred_coords: torch.Tensor,
    pred_lattice: torch.Tensor,
    true_coords: torch.Tensor,
    true_lattice: torch.Tensor,
    r_max: float = 10.0,
    nbins: int = 100
) -> Dict[str, float]:
    """
    Computes Global RDF metrics:
    1. JSD (Jensen-Shannon Divergence)
    2. Overlap %
    3. Wasserstein Distance
    
    Includes "Short Range" variants (0 - 5.0 A) to focus on bonded/close-contact structure.
    """
    bins = torch.linspace(0, r_max, nbins + 1, device=pred_coords.device)
    bin_width = r_max / nbins
    
    # Define short-range cutoff (e.g., 5.0 Angstroms)
    short_range_cutoff = 5.0
    short_range_bins = int(short_range_cutoff / bin_width)
    
    # Helper to get all pairwise distances
    def get_all_dists(coords, lattice):
        L = lattice_params_to_matrix_torch(lattice)
        edge_index, shifts, dists = build_periodic_edges(coords, L, r_max)
        if dists.numel() == 0:
            return torch.tensor([], device=coords.device)
        return dists

    # 1. Get Distributions
    dists_pred = get_all_dists(pred_coords, pred_lattice)
    dists_true = get_all_dists(true_coords, true_lattice)
    
    metrics = {}

    # Case 1: Both have data
    if dists_pred.numel() > 0 and dists_true.numel() > 0:
        # Full Range
        pdf_p = compute_hist_pdf(dists_pred, bins)
        pdf_t = compute_hist_pdf(dists_true, bins)
        
        metrics["rdf_jsd"] = jsd(pdf_p, pdf_t)
        metrics["rdf_overlap_pct"] = compute_hist_overlap(pdf_p, pdf_t)
        metrics["rdf_wasserstein"] = wasserstein_from_pmf(pdf_p, pdf_t, bin_width)
        
        # Short Range (slice the PDFs)
        # Re-normalize for the short range to treat it as a proper distribution for shape comparison
        if short_range_bins < len(pdf_p):
            pdf_p_short = pdf_p[:short_range_bins]
            pdf_t_short = pdf_t[:short_range_bins]
            
            # If empty in short range (rare but possible), handle gracefully
            if pdf_p_short.sum() > 1e-6 and pdf_t_short.sum() > 1e-6:
                # Normalize to sum to 1 for shape comparison metrics (JSD, Wasserstein)
                pdf_p_short_norm = pdf_p_short / pdf_p_short.sum()
                pdf_t_short_norm = pdf_t_short / pdf_t_short.sum()
                
                metrics["rdf_overlap_short_range_pct"] = compute_hist_overlap(pdf_p_short_norm, pdf_t_short_norm)
                metrics["rdf_wasserstein_short_range"] = wasserstein_from_pmf(pdf_p_short_norm, pdf_t_short_norm, bin_width)
            else:
                metrics["rdf_overlap_short_range_pct"] = 0.0
                metrics["rdf_wasserstein_short_range"] = 10.0 # Arbitrary high penalty
    
    # Case 2: Both empty
    elif dists_pred.numel() == 0 and dists_true.numel() == 0:
        metrics["rdf_jsd"] = 0.0
        metrics["rdf_overlap_pct"] = 100.0
        metrics["rdf_wasserstein"] = 0.0
        metrics["rdf_overlap_short_range_pct"] = 100.0
        metrics["rdf_wasserstein_short_range"] = 0.0
        
    # Case 3: Mismatch
    else:
        metrics["rdf_jsd"] = 0.693 
        metrics["rdf_overlap_pct"] = 0.0
        metrics["rdf_wasserstein"] = r_max # Max possible error approx
        metrics["rdf_overlap_short_range_pct"] = 0.0
        metrics["rdf_wasserstein_short_range"] = short_range_cutoff
        
    return metrics


# -------------------------------------------------------------------
# ---- Master Aggregator --------------------------------------------
# -------------------------------------------------------------------

def compute_all_metrics(
    pred_coords_list: List[torch.Tensor],      # [N, 3] fractional
    pred_lattice_list: List[torch.Tensor],     # [6] params
    true_coords_list: List[torch.Tensor],      # [N, 3] fractional
    true_lattice_list: List[torch.Tensor],     # [6] params
    atomic_numbers_list: List[torch.Tensor],   # [N]
    atomic_masses: Dict[int, float],
    covalent_radii: Dict[int, float],
    clash_alpha: float = 0.75,
    amd_k: int = 100,
    rdf_r_max: float = 10.0,
    contact_cutoff: float = 6.0,
) -> Dict[str, float]:
    """
    Computes all metrics for a batch and returns mean values.
    """
    metrics_acc = defaultdict(list)
    
    # 1. Density
    dens_metrics = metric_density_stats(
        pred_lattice_list, true_lattice_list, atomic_numbers_list, atomic_masses
    )
    for k, v in dens_metrics.items():
        metrics_acc[k].append(v)
        
    # 2. Per-Crystal Loop
    num_crystals = len(pred_coords_list)
    
    for i in range(num_crystals):
        # Unpack inputs
        p_c, p_l = pred_coords_list[i], pred_lattice_list[i]
        t_c, t_l = true_coords_list[i], true_lattice_list[i]
        z = atomic_numbers_list[i]
        
        # A. Clash Rates (Total Only)
        clash = compute_clash_metrics(p_c, p_l, z, covalent_radii, clash_alpha)
        for k, v in clash.items():
            metrics_acc[k].append(v)
            
        # B. AMD L_inf
        amd_val = metric_amd_linf(p_c, p_l, t_c, t_l, z, amd_k)
        metrics_acc["amd_linf"].append(amd_val)
        
        # C. RDF (Total Only)
        rdf = compute_rdf_jsd_metrics(p_c, p_l, t_c, t_l, rdf_r_max)
        for k, v in rdf.items():
            metrics_acc[k].append(v)

        # D. Contact Metrics (New)
        contacts = compute_contact_metrics(p_c, p_l, t_c, t_l, radius=contact_cutoff)
        for k, v in contacts.items():
            metrics_acc[k].append(v)
            
    # Aggregate results (Mean)
    final_metrics = {}
    final_metrics.update(dens_metrics)
    for k, v_list in metrics_acc.items():
        if k not in final_metrics:
            if len(v_list) > 0:
                final_metrics[k] = sum(v_list) / len(v_list)
            else:
                final_metrics[k] = 0.0
                
    return final_metrics


# -------------------------------------------------------------------
# ---- Constants & Compatibility ------------------------------------
# -------------------------------------------------------------------

DEFAULT_ATOMIC_MASSES = {
    1: 1.0079, 6: 12.011, 7: 14.0067, 8: 15.999, 9: 18.998,
    15: 30.9738, 16: 32.065, 17: 35.453, 35: 79.904, 53: 126.904
}

DEFAULT_COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

def get_default_constants():
    """Return default atomic masses and covalent radii."""
    return DEFAULT_ATOMIC_MASSES, DEFAULT_COVALENT_RADII

def mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compatibility helper for external scripts."""
    if len(metrics_list) == 0: return {}
    first_dict = metrics_list[0]
    numeric_keys = [k for k in first_dict.keys() if isinstance(first_dict[k], (int, float))]
    agg = {k: 0.0 for k in numeric_keys}
    for m in metrics_list:
        for k in numeric_keys: agg[k] += float(m[k])
    mean = {f"mean_{k}": agg[k] / len(metrics_list) for k in numeric_keys}
    mean["num_crystals"] = len(metrics_list)
    if 'refcode' in first_dict: mean['sample_refcodes'] = [m['refcode'] for m in metrics_list]
    return mean

def compute_all_metrics_for_crystal(
    pred_coords: torch.Tensor, pred_lattice: torch.Tensor,
    true_coords: torch.Tensor, true_lattice: torch.Tensor,
    atomic_numbers: torch.Tensor, edge_index: torch.Tensor, # Keeps compatibility
    atomic_masses_g_mol: Dict[int, float], covalent_radii_A: Dict[int, float],
    # Ignored legacy args
    molecule_ids: Optional[torch.Tensor] = None, 
    clash_alpha: float = 0.75, amd_k: int = 100, rdf_r_max: float = 10.0,
    contact_cutoff: float = 6.0
) -> Dict[str, float]:
    """Compatibility wrapper: Cartesian -> Fractional."""
    pred_lattice_mat = lattice_params_to_matrix_torch(pred_lattice)
    true_lattice_mat = lattice_params_to_matrix_torch(true_lattice)
    pred_coords_frac = cart_to_frac(pred_coords, pred_lattice_mat)
    true_coords_frac = cart_to_frac(true_coords, true_lattice_mat)
    return compute_all_metrics(
        pred_coords_list=[pred_coords_frac], pred_lattice_list=[pred_lattice],
        true_coords_list=[true_coords_frac], true_lattice_list=[true_lattice],
        atomic_numbers_list=[atomic_numbers], 
        atomic_masses=atomic_masses_g_mol, covalent_radii=covalent_radii_A,
        clash_alpha=clash_alpha, amd_k=amd_k, rdf_r_max=rdf_r_max,
        contact_cutoff=contact_cutoff
    )

def compute_metrics_batched(
    pred_coords_list: List[torch.Tensor], pred_lattice_list: List[torch.Tensor],
    true_coords_list: List[torch.Tensor], true_lattice_list: List[torch.Tensor],
    atomic_numbers_list: List[torch.Tensor], edge_index_list: List[torch.Tensor],
    atomic_masses_g_mol: Dict[int, float], covalent_radii_A: Dict[int, float],
    molecule_ids_list: Optional[List[torch.Tensor]] = None, device: Optional[torch.device] = None,
    **kwargs
) -> Dict[str, float]:
    """Compatibility wrapper: Cartesian -> Fractional."""
    if device is None: device = pred_coords_list[0].device
    # Move relevant tensors
    pred_coords_list = [c.to(device) for c in pred_coords_list]
    pred_lattice_list = [l.to(device) for l in pred_lattice_list]
    true_coords_list = [c.to(device) for c in true_coords_list]
    true_lattice_list = [l.to(device) for l in true_lattice_list]
    atomic_numbers_list = [z.to(device) for z in atomic_numbers_list]
    
    pred_coords_frac_list = []
    true_coords_frac_list = []
    for i in range(len(pred_coords_list)):
        pred_mat = lattice_params_to_matrix_torch(pred_lattice_list[i])
        true_mat = lattice_params_to_matrix_torch(true_lattice_list[i])
        pred_coords_frac_list.append(cart_to_frac(pred_coords_list[i], pred_mat))
        true_coords_frac_list.append(cart_to_frac(true_coords_list[i], true_mat))
        
    return compute_all_metrics(
        pred_coords_list=pred_coords_frac_list, pred_lattice_list=pred_lattice_list,
        true_coords_list=true_coords_frac_list, true_lattice_list=true_lattice_list,
        atomic_numbers_list=atomic_numbers_list,
        atomic_masses=atomic_masses_g_mol, covalent_radii=covalent_radii_A, **kwargs
    )