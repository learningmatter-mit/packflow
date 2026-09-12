import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter, scatter_add
from torch_geometric.data import Data, Batch
from math import sqrt
from vesin import NeighborList

# Import utility functions from crystal_utils
from packflow.utils.crystal_utils import (
    frac_to_cart, 
    cart_to_frac,
    frac_to_cart_batched,
    cart_to_frac_batched,
    lattice_params_to_matrix_torch,
    lattice_matrix_to_params_torch,
    lattice_matrix_to_k_basis_torch,
    k_basis_to_lattice_matrix_torch
)

def _generate_run_name():
    """Generate a unique run name for offline wandb mode (directory naming only)."""
    import datetime
    import random
    import string

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"run_{timestamp}_{random_suffix}"


def create_time_grid(n_steps: int, grid_type: str = "linear", device: str = "cpu") -> torch.Tensor:
    """
    Create time grid for sampling based on different schedules.
    
    Args:
        n_steps: Number of integration steps
        grid_type: Type of time grid ("linear", "quadratic", "exponential")
        device: Device to create tensor on
        
    Returns:
        time_grid: Tensor of shape [n_steps] with values in [0, 1]
    """
    if grid_type == "linear":
        # Linear time grid: t_n = n / N
        time_grid = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]  # Exclude t=1
    elif grid_type == "quadratic":
        # Quadratic time grid: t_n = (n / N)^2
        linear_grid = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]
        time_grid = linear_grid ** 2
    elif grid_type == "exponential":
        # Exponential time grid: t_n = (1 - 10^(-2*n/N)) / (1 - 10^(-2))
        # This gives fast early progress as described in the paper
        n = torch.arange(n_steps, device=device, dtype=torch.float32)
        time_grid = (1 - 10**(-2 * n / n_steps)) / (1 - 10**(-2))
        # Ensure the last value is exactly 1.0
        time_grid[-1] = 1.0
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}. Must be one of: linear, quadratic, exponential")
    
    return time_grid


def create_dual_time_grids(n_steps: int, coords_grid_type: str = "linear", 
                          lattice_grid_type: str = "linear", device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create separate time grids for coordinates and lattice sampling.
    
    Args:
        n_steps: Number of integration steps
        coords_grid_type: Type of time grid for coordinates ("linear", "quadratic", "exponential")
        lattice_grid_type: Type of time grid for lattice ("linear", "quadratic", "exponential")
        device: Device to create tensors on
        
    Returns:
        coords_time_grid: Tensor of shape [n_steps] for coordinates
        lattice_time_grid: Tensor of shape [n_steps] for lattice
    """
    coords_time_grid = create_time_grid(n_steps, coords_grid_type, device)
    lattice_time_grid = create_time_grid(n_steps, lattice_grid_type, device)
    
    return coords_time_grid, lattice_time_grid


# Removed: lattice_matrix_from_params_torch and lattice_params_from_H_torch
# Now using functions from packflow.utils.crystal_utils:
# - lattice_params_to_matrix_torch (replaces lattice_matrix_from_params_torch)
# - lattice_matrix_to_params_torch (replaces lattice_params_from_H_torch)


def H_from_vec(h_vec: torch.Tensor) -> torch.Tensor:
    """Flattened-to-matrix. h_vec: [B,9] -> H: [B,3,3] (row-major)."""
    return h_vec.view(h_vec.size(0), 3, 3)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

def _group_name(param_name: str) -> str:
    """
    Group parameters into readable layer buckets:
    - transformer.layers.{k}
    - attn_bias_builder, riem_bias_composer, cart_coords_embedder, lattice_embedder, ...
    """
    parts = param_name.split(".")
    if len(parts) >= 3 and parts[0] == "transformer" and parts[1] == "layers":
        return ".".join(parts[:3])  # transformer.layers.{idx}
    return parts[0]  # top-level module (e.g., cart_coords_embedder)


def compute_grad_norms_by_group(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict: group -> {'grad_norm': float, 'param_norm': float, 'num_params': int}
    Only includes parameters that require grad and have a gradient.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        group = _group_name(name)
        d = out.setdefault(group, {"grad_sq": 0.0, "param_sq": 0.0, "num_params": 0})
        d["grad_sq"] += float(g.float().pow(2).sum().item())
        d["param_sq"] += float(p.detach().float().pow(2).sum().item())
        d["num_params"] += p.numel()
    
    # finalize norms
    for k, d in out.items():
        d["grad_norm"] = math.sqrt(d.pop("grad_sq")) if d["num_params"] > 0 else 0.0
        d["param_norm"] = math.sqrt(d.pop("param_sq")) if d["num_params"] > 0 else 0.0
    
    return out


class LatticeTransform:
    """
    Handles transformation between constrained and unconstrained lattice parameters.
    Adapted from flowmm lattice_params.py logic.
    """
    
    def __init__(self, angle_bounds: Tuple[float, float] = (60.0, 120.0)):
        self.angle_bounds = angle_bounds
    
    def constrained_to_unconstrained(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """
        Convert lattice parameters [a, b, c, α, β, γ] to unconstrained space.
        Lengths stay the same, angles get sigmoid-transformed to unconstrained space.
        
        Args:
            lattice_params: [batch_size, 6] - [a, b, c, α, β, γ] where angles are in degrees
        Returns:
            unconstrained: [batch_size, 6] - [a, b, c, α_uncon, β_uncon, γ_uncon]
        """
        lengths = lattice_params[..., :3]  # [a, b, c] - keep as is
        angles = lattice_params[..., 3:]   # [α, β, γ] in degrees
        
        # Get unconstrained distribution transform
        low = torch.tensor(self.angle_bounds[0], device=lattice_params.device, dtype=lattice_params.dtype)
        high = torch.tensor(self.angle_bounds[1], device=lattice_params.device, dtype=lattice_params.dtype)
        
        # Create uniform distribution for angles and get inverse transform
        uniform_dist = torch.distributions.Uniform(low, high)
        transform = torch.distributions.biject_to(uniform_dist.support).inv
        
        # Transform angles to unconstrained space  
        angles_unconstrained = transform(angles)
        
        return torch.cat([lengths, angles_unconstrained], dim=-1)
    
    def unconstrained_to_constrained(self, unconstrained_params: torch.Tensor) -> torch.Tensor:
        """
        Convert unconstrained lattice parameters back to constrained space.
        
        Args:
            unconstrained_params: [batch_size, 6] - [a, b, c, α_uncon, β_uncon, γ_uncon]
        Returns:
            lattice_params: [batch_size, 6] - [a, b, c, α, β, γ] where angles are in degrees
        """
        lengths = unconstrained_params[..., :3]  # [a, b, c] - keep as is
        angles_unconstrained = unconstrained_params[..., 3:]   # [α_uncon, β_uncon, γ_uncon]
        
        # Get constrained distribution transform
        low = torch.tensor(self.angle_bounds[0], device=unconstrained_params.device, dtype=unconstrained_params.dtype)
        high = torch.tensor(self.angle_bounds[1], device=unconstrained_params.device, dtype=unconstrained_params.dtype)
        
        # Create uniform distribution for angles and get forward transform
        uniform_dist = torch.distributions.Uniform(low, high)
        transform = torch.distributions.biject_to(uniform_dist.support)
        
        # Transform angles back to constrained space
        angles = transform(angles_unconstrained)
        
        return torch.cat([lengths, angles], dim=-1)


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding 


class CrystalTransformerEncoder(nn.Module):
    """Transformer for crystal flow matching (Cartesian). Only covalent bonds + RDKit bond features
    are supported for attention bias. No non-bonded edges, no PBC graph building, no distance/RBF/dir attrs.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
        time_embed_dim: int = 64,
        cart_coords_dim: int = 3,
        lattice_dim: int = 6,
        # --- attention bias ---
        use_attention_bias_from_graph: bool = True,
        attn_bias_heads: Optional[int] = None,
        attn_bias_combine: str = "sum",
        attn_bias_baseline: float = 0.0,
        bias_scale: float = 1.0,
        learnable_bias_scale: bool = False,
        # --- data features ---
        use_positional_embeddings: bool = True,
        use_rdkit_features: bool = True,
        lattice_token: bool = False,
        rdkit_bond_feat_dim: int = 2,
        rdkit_node_feat_dim: int = 10,
        # --- periodic coordinate embedding ---
        periodic_coord_emb: bool = False,
        periodic_nmax: int = 5,
        periodic_topk: int = 512,
        # --- optional fractional-coordinate embedding ---
        use_fractional_coords: bool = False,
        # --- GNN for bonds (alternative to attention bias) ---
        use_gnn_for_bonds: bool = False,
        num_gnn_layers: int = 3,
        gat_heads: int = 4,
        gat_concat: bool = False,
        gat_dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.cart_coords_dim = cart_coords_dim
        self.lattice_rep = "invariant"
        self.lattice_dim = lattice_dim if lattice_dim is not None else 6
        self.use_positional_embeddings = bool(use_positional_embeddings)
        self.use_rdkit_features = bool(use_rdkit_features)
        self.use_lattice_token = bool(lattice_token)

        self.use_attention_bias_from_graph = bool(use_attention_bias_from_graph)
        self.use_gnn_for_bonds = bool(use_gnn_for_bonds)
        
        # Mutual exclusivity validation
        if self.use_gnn_for_bonds and self.use_attention_bias_from_graph:
            raise ValueError("Cannot use both use_gnn_for_bonds and use_attention_bias_from_graph. Choose one.")
        
        self.attn_bias_heads = attn_bias_heads if attn_bias_heads is not None else nhead
        self.attn_bias_combine = attn_bias_combine
        self.attn_bias_baseline = float(attn_bias_baseline)
        if self.use_attention_bias_from_graph and self.attn_bias_heads != nhead:
            raise ValueError("attn_bias_heads must equal nhead for additive mask shaping.")

        self.bias_scale_val = float(bias_scale)
        self.learnable_bias_scale = bool(learnable_bias_scale)

        self.use_fractional_coords = bool(use_fractional_coords)

        self.cart_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.lattice_embedder = nn.Sequential(
            nn.Linear(self.lattice_dim, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        if self.use_lattice_token:
            self.register_parameter("lattice_token_type", nn.Parameter(torch.randn(1, d_model) * 0.02))
        else:
            self.lattice_token_type = None

        # Fractional coordinate embedder (u_t in fractional basis, no wrapping)
        if self.use_fractional_coords:
            self.frac_coords_embedder = nn.Sequential(
                nn.Linear(3, d_model, bias=False),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.frac_coords_embedder = None

        self._node_feat_mlp = nn.Sequential(
            nn.Linear(rdkit_node_feat_dim, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.rdkit_bond_feat_dim = rdkit_bond_feat_dim

        self.lattice_transform = LatticeTransform()

        # Periodic coordinate embedding (axis-aligned only; no Top-K/ρ selection)
        self.periodic_coord_emb = bool(periodic_coord_emb)
        self.periodic_nmax = int(periodic_nmax)
        self.periodic_topk = int(periodic_topk)  # kept for backward-compatibility; ignored here

        if self.periodic_coord_emb:
            # Build a FIXED axis-aligned bank once
            fb_axis = self._build_axis_aligned_bank(self.periodic_nmax)  # [K,3], Long
            if fb_axis.numel() == 0:
                raise ValueError("periodic_nmax must be >= 1 when periodic_coord_emb=True.")
            self.register_buffer("freq_bank", fb_axis)  # fixed, axis-aligned
            K = fb_axis.size(0)
            # Projection for 2*K → d_model (cos + sin)
            self.periodic_proj = nn.Linear(2 * K, d_model, bias=False)
        else:
            self.freq_bank = None
            self.periodic_proj = None

        # GNN layers for bond information injection
        if self.use_gnn_for_bonds:
            self.gnn_layers = nn.ModuleList([
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model,
                    heads=gat_heads,
                    concat=gat_concat,
                    edge_dim=rdkit_bond_feat_dim,
                    dropout=gat_dropout,
                ) for _ in range(num_gnn_layers)
            ])
            print(f"🔧 GNN FOR BONDS: Created {num_gnn_layers} GAT layers with {gat_heads} heads, concat={gat_concat}")
        else:
            self.gnn_layers = None

        if self.use_attention_bias_from_graph:
            class GraphAttentionBias(nn.Module):
                def __init__(self, d_edge: int, nhead: int, bias_init: float = 0.0, combine: str = "sum"):
                    super().__init__()
                    assert combine in ("sum", "mean")
                    self.combine = combine
                    self.nhead = nhead
                    hidden = max(64, d_edge)
                    in_dim = max(d_edge, 1)
                    self.edge_to_headlogits = nn.Sequential(
                        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, nhead)
                    )
                    self.head_bias_base = nn.Parameter(torch.full((nhead,), float(bias_init)))

                def forward(self, edge_index: torch.Tensor, edge_attr_scalar: torch.Tensor,
                            batch: torch.Tensor, nodes_dense: torch.Tensor, mask: torch.Tensor, M_total: int) -> torch.Tensor:
                    device = edge_index.device
                    B, M = nodes_dense.size(0), nodes_dense.size(1)
                    H = self.nhead

                    if edge_attr_scalar is None:
                        raise ValueError("edge_attr_scalar is None: RDKit bond features are required for bias.")

                    # Map global -> local positions
                    local_pos = torch.full((batch.size(0),), -1, device=device, dtype=torch.long)
                    pos_grid = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
                    local_pos[nodes_dense[mask]] = pos_grid[mask]

                    head_logits_e = self.edge_to_headlogits(edge_attr_scalar)
                    src, dst = edge_index
                    b_src = batch[src]
                    b_dst = batch[dst]
                    same = (b_src == b_dst)

                    bias = self.head_bias_base.view(1, H, 1, 1).expand(B, H, M_total, M_total).clone()
                    if not same.any():
                        return bias

                    src = src[same]; dst = dst[same]
                    b_e = b_src[same]
                    i_local = local_pos[src]; j_local = local_pos[dst]
                    valid = (i_local >= 0) & (j_local >= 0)
                    if not valid.any():
                        return bias

                    b_e = b_e[valid]; i_local = i_local[valid]; j_local = j_local[valid]
                    head_logits_e = head_logits_e[same][valid]

                    bias_flat = bias[:, :, :M, :M].contiguous().view(B * H, M, M)
                    E = head_logits_e.size(0)
                    h_ar = torch.arange(H, device=device).view(1, H).expand(E, H)
                    b_ar = b_e.view(E, 1).expand(E, H)
                    i_ar = i_local.view(E, 1).expand(E, H)
                    j_ar = j_local.view(E, 1).expand(E, H)
                    bh_index = (b_ar * H + h_ar).reshape(-1)
                    i_index = i_ar.reshape(-1)
                    j_index = j_ar.reshape(-1)
                    vals    = head_logits_e.reshape(-1)
                    bias_flat.index_put_((bh_index, i_index, j_index), vals, accumulate=True)
                    bias[:, :, :M, :M] = bias_flat.view(B, H, M, M)
                    return bias

            class GraphTransformerEncoderLayer(nn.Module):
                def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, activation: nn.Module, norm_first: bool, bias: bool):
                    super().__init__()
                    self.nhead = nhead
                    self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, bias=bias)
                    self.linear1 = nn.Linear(d_model, dim_ff, bias=bias)
                    self.linear2 = nn.Linear(dim_ff, d_model, bias=bias)
                    self.dropout = nn.Dropout(dropout)
                    self.dropout1 = nn.Dropout(dropout)
                    self.dropout2 = nn.Dropout(dropout)
                    self.norm_first = norm_first
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.act = activation

                def _with_bias(self, B, H, M, attn_bias: torch.Tensor) -> torch.Tensor:
                    return attn_bias.reshape(B * H, M, M)

                def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
                    B, M, _ = x.shape
                    additive_mask = self._with_bias(B, self.nhead, M, attn_bias)

                    def sa_block(inp):
                        out, _ = self.mha(inp, inp, inp, attn_mask=additive_mask, key_padding_mask=key_padding_mask, need_weights=False)
                        return out

                    def ff_block(inp):
                        return self.linear2(self.dropout(self.act(self.linear1(inp))))

                    if self.norm_first:
                        x = x + self.dropout1(sa_block(self.norm1(x)))
                        x = x + self.dropout2(ff_block(self.norm2(x)))
                    else:
                        x = self.norm1(x + self.dropout1(sa_block(x)))
                        x = self.norm2(x + self.dropout2(ff_block(x)))
                    return x

            class GraphTransformerEncoder(nn.Module):
                def __init__(self, num_layers: int, d_model: int, nhead: int, dim_ff: int, dropout: float, activation: nn.Module, norm_first: bool, bias: bool):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        GraphTransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation, norm_first, bias)
                    for _ in range(num_layers)])
                    self.norm = nn.LayerNorm(d_model)

                def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
                    for layer in self.layers:
                        x = layer(x, key_padding_mask, attn_bias)
                    return self.norm(x)

            self.attn_bias_builder = GraphAttentionBias(
                d_edge=self.rdkit_bond_feat_dim,
                nhead=self.attn_bias_heads,
                bias_init=self.attn_bias_baseline,
                combine=self.attn_bias_combine,
            )
            if self.learnable_bias_scale:
                self.attn_bias_scale = nn.Parameter(torch.tensor(self.bias_scale_val, dtype=torch.float32))
            else:
                self.register_buffer("attn_bias_scale", torch.tensor(self.bias_scale_val, dtype=torch.float32))

        self.time_cart_embedder = nn.Sequential(nn.Linear(time_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time_lattice_embedder = nn.Sequential(nn.Linear(time_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        activation_mod = {"gelu": nn.GELU(approximate="tanh"), "relu": nn.ReLU()}[activation]
        if self.use_attention_bias_from_graph:
            self.transformer = GraphTransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout, activation_mod, norm_first, bias)
        else:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           activation=activation_mod, dropout=dropout, batch_first=True,
                                           norm_first=norm_first, bias=bias),
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

        self.cart_coords_output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, cart_coords_dim))
        self.lattice_output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, self.lattice_dim))

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim)
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    @staticmethod
    def _build_axis_aligned_bank(nmax: int) -> torch.Tensor:
        """
        Build axis-aligned integer modes only:
          N = {(n,0,0), (0,n,0), (0,0,n)} for n = 1..nmax

        Returns: LongTensor[K, 3] with K = 3 * nmax. Zero mode excluded.
        """
        if nmax < 1:
            return torch.empty(0, 3, dtype=torch.long)
        ns = torch.arange(1, nmax + 1, dtype=torch.long)
        ex = torch.stack([ns, torch.zeros_like(ns), torch.zeros_like(ns)], dim=1)
        ey = torch.stack([torch.zeros_like(ns), ns, torch.zeros_like(ns)], dim=1)
        ez = torch.stack([torch.zeros_like(ns), torch.zeros_like(ns), ns], dim=1)
        bank = torch.cat([ex, ey, ez], dim=0)  # [3*nmax, 3]
        return bank

    def _periodic_coord_embed(
        self,
        coords: torch.Tensor,                  # [N,3]  either x (Cartesian) or u (fractional)
        lattice_unconstrained: torch.Tensor,   # [B,6]
        batch: torch.Tensor,                   # [N]
        coords_are_fractional: bool = False,
    ) -> torch.Tensor:
        """
        Build lattice-aware sinusoidal features WITHOUT any ρ-based Top-K selection.
        Modes are axis-aligned only:
           N = {(n,0,0), (0,n,0), (0,0,n)} for n=1..nmax.

        Phi_i = [cos(2π n^T u_i), sin(2π n^T u_i)]_{n in N}, then a learned linear projection.

        Returns: [N, d_model]
        """
        assert self.freq_bank is not None and self.periodic_proj is not None
        device = coords.device
        dtype  = coords.dtype

        # 1) Get fractional coordinates u in [0,1)
        if coords_are_fractional:
            u = coords
        else:
            lat_con = self.lattice_transform.unconstrained_to_constrained(lattice_unconstrained)  # [B,6]
            H_all   = lattice_params_to_matrix_torch(lat_con)                                    # [B,3,3]
            # Convert cartesian to fractional using utility function
            H_nodes = H_all[batch]  # [N,3,3]
            u = cart_to_frac(coords, H_nodes)  # [N,3]
        u = u - torch.floor(u)  # wrap to [0,1)

        # 2) Fixed axis-aligned integer bank
        n = self.freq_bank.to(device=device, dtype=dtype)  # [K,3]

        # 3) Phases and features
        theta = 2.0 * math.pi * (u @ n.T)                  # [N,K]
        phi = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # [N, 2K]

        # Optional variance stabilization so K doesn't blow up magnitudes
        K = n.size(0)
        if K > 0:
            phi = phi / math.sqrt(K)

        # 4) Linear projection to model width
        return self.periodic_proj(phi)                     # [N, d_model]

    def forward(self, atom_types: torch.Tensor, cart_coords: torch.Tensor,
                lattice: torch.Tensor, batch: torch.Tensor,
                t_cart: torch.Tensor, t_lattice: torch.Tensor,
                edge_index: torch.Tensor,
                node_features: Optional[torch.Tensor] = None,
                bond_features: Optional[torch.Tensor] = None,
                token_idx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Always Cartesian embedding path now
        if self.periodic_coord_emb:
            x = self._periodic_coord_embed(cart_coords, lattice, batch, coords_are_fractional=False)
        else:
            x = self.cart_coords_embedder(cart_coords)

        if node_features is not None and node_features.numel() > 0:
            x += self._node_feat_mlp(node_features.to(cart_coords.device).float())

        # Optional fractional-coordinate features at time t (no wrapping)
        if self.use_fractional_coords:
            # lattice is in unconstrained 6D; convert to constrained [a,b,c,α,β,γ]
            lattice_constrained = self.lattice_transform.unconstrained_to_constrained(lattice)  # [B,6]
            # Build lattice matrices H(t) from constrained params
            H_t = lattice_params_to_matrix_torch(lattice_constrained)  # [B,3,3]
            # Gather per-node H based on batch indices
            H_nodes = H_t[batch]  # [N,3,3]
            # Convert cartesian to fractional using utility function (u are fractional coordinates, not wrapped)
            u = cart_to_frac(cart_coords, H_nodes)  # [N,3]
            # Embed fractional coords and add to node representation
            x += self.frac_coords_embedder(u)

        lattice_embed = self.lattice_embedder(lattice)
        lattice_embed_expanded = lattice_embed[batch]
        if not self.use_lattice_token:
            x += lattice_embed_expanded

        if self.use_positional_embeddings:
            if token_idx is None:
                token_idx = torch.arange(len(atom_types), device=atom_types.device)
            x += get_index_embedding(token_idx, self.d_model)

        t_cart_embed = self.time_cart_embedder(self.sinusoidal_embedding(t_cart))
        t_lattice_embed = self.time_lattice_embedder(self.sinusoidal_embedding(t_lattice))
        x += t_cart_embed[batch]
        if not self.use_lattice_token:
            x += t_lattice_embed[batch]

        # GNN pass for bond information injection (before dense batching)
        if self.use_gnn_for_bonds:
            if bond_features is None or bond_features.numel() == 0:
                raise ValueError("bond_features required for GNN pass. Provide per-edge features aligned with edge_index.")
            bf = bond_features.to(x.device).float()
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index, edge_attr=bf)

        x_dense, token_mask = to_dense_batch(x, batch)
        B, M, _ = x_dense.shape
        x_all = x_dense
        key_padding_mask = ~token_mask

        if self.use_lattice_token:
            lattice_tok = (lattice_embed + t_lattice_embed + self.lattice_token_type).unsqueeze(1)
            x_all = torch.cat([x_all, lattice_tok], dim=1)
            M_total = M + 1
            lat_valid = torch.zeros((B, 1), dtype=torch.bool, device=x_all.device)
            key_padding_mask = torch.cat([key_padding_mask, lat_valid], dim=1)
        else:
            M_total = M

        if self.use_attention_bias_from_graph:
            if bond_features is None or bond_features.numel() == 0:
                raise ValueError("RDKit bond_features are required but missing. Provide per-edge features aligned with edge_index.")
            nodes_dense, mask_dense = to_dense_batch(torch.arange(len(atom_types), device=atom_types.device), batch)
            nodes_dense = nodes_dense.long()
            attn_bias_graph = self.attn_bias_builder(
                edge_index=edge_index,
                edge_attr_scalar=bond_features.to(x.device).float(),
                batch=batch,
                nodes_dense=nodes_dense,
                mask=mask_dense,
                M_total=M_total,
            )
            attn_bias = self.attn_bias_scale * attn_bias_graph
            x_out = self.transformer(x_all, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        else:
            x_out = self.transformer.forward(x_all, src_key_padding_mask=key_padding_mask)

        x_atoms_out = x_out[:, :M, :]
        x_atoms_out = x_atoms_out[token_mask]
        cart_coords_field = self.cart_coords_output_head(x_atoms_out)

        if self.use_lattice_token:
            lattice_features = x_out[:, M, :]
        else:
            lattice_features = scatter(x_atoms_out, batch, dim=0, reduce='mean')

        lattice_field = self.lattice_output_head(lattice_features)
        return cart_coords_field, lattice_field


class CrystalFlowMatching:
    """Flow matching implementation for crystal structure prediction using Cartesian coordinates."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', lattice_loss_weight: float = 1.0, 
                 shared_time: bool = False, use_wandb: bool = False, wandb_prefix: str = "",
                 checkpoint_dir: str = None, save_checkpoint_every: int = 50,
                 use_logit_normal_resampling: bool = False,
                 logit_normal_m: float = -0.8,  # Mean in logit space (mirror of paper's m=0.8 for t=0 clean data)
                 logit_normal_s: float = 1.7,  # Std in logit space (same as paper)
                 logit_normal_mix: float = 0.02,  # Mix with uniform sampling
                 t_eps: float = 0.0,  # Epsilon to avoid sampling exactly at boundaries
                 fixed_time: Optional[float] = None,  # If set, use this fixed time instead of sampling
                 use_smooth_lddt_loss: bool = False,
                 smooth_lddt_loss_weight: float = 1.0,
                 lddt_cutoff: float = 15.0,
                 lddt_weight_schedule: bool = False,  # Time-dependent LDDT weighting
                 grad_log_every: int = 0,
                 use_bond_length_loss: bool = False,
                 bond_length_loss_weight: float = 1.0,
                 add_periodic_edges: bool = False,
                 periodic_edge_cutoff: float = 5.0,
                 periodic_edge_periodic: bool = True,
                 periodic_edge_time_cutoff: Optional[float] = 0.5,
                 # --- periodic lDDT aux loss ---
                 use_periodic_lddt_loss: bool = False,
                 periodic_lddt_loss_weight: float = 1.0,
                 periodic_lddt_cutoff: Optional[float] = None,
                 periodic_lddt_warmup_epochs: int = 0,
                 # --- k_basis representation for lattice flow ---
                 use_k_basis_representation: bool = False,
                 ):
        self.model = model.to(device)
        self.device = device
        self.lattice_loss_weight = lattice_loss_weight
        self.shared_time = shared_time
        self.lattice_transform = LatticeTransform()
        self.use_k_basis_representation = bool(use_k_basis_representation)
        self.lattice_rep = "k_basis" if self.use_k_basis_representation else "invariant"
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_prefix = wandb_prefix
        self.save_checkpoint_every = save_checkpoint_every
        self.best_val_loss = float('inf')  # Track best validation loss
        
        self.use_logit_normal_resampling = use_logit_normal_resampling
        self.logit_normal_m = logit_normal_m
        self.logit_normal_s = logit_normal_s
        self.logit_normal_mix = logit_normal_mix
        self.t_eps = t_eps
        
        self.fixed_time = fixed_time
        if fixed_time is not None:
            if not (0.0 <= fixed_time <= 1.0):
                raise ValueError(f"fixed_time must be in [0, 1], got {fixed_time}")
            print(f"🔧 FIXED TIME MODE: Using fixed interpolation time t={fixed_time} for all training steps")
        
        self.use_smooth_lddt_loss = use_smooth_lddt_loss
        self.smooth_lddt_loss_weight = smooth_lddt_loss_weight
        self.lddt_cutoff = lddt_cutoff
        self.lddt_weight_schedule = lddt_weight_schedule
        
        self.use_bond_length_loss = use_bond_length_loss
        self.bond_length_loss_weight = bond_length_loss_weight
        
        self.grad_log_every = int(grad_log_every)  # 0 disables logging

        self.add_periodic_edges = bool(add_periodic_edges)
        self.periodic_edge_cutoff = float(periodic_edge_cutoff)
        self.periodic_edge_periodic = bool(periodic_edge_periodic)
        self.periodic_edge_time_cutoff = periodic_edge_time_cutoff
        self.misc_bond_type_index = 4  # Index of "misc" in bond_features_list["bond_type"]
        
        # Periodic lDDT config
        self.use_periodic_lddt_loss = bool(use_periodic_lddt_loss)
        self.periodic_lddt_loss_weight = float(periodic_lddt_loss_weight)
        self.periodic_lddt_cutoff = float(periodic_lddt_cutoff) if periodic_lddt_cutoff is not None else None
        self.periodic_lddt_warmup_epochs = int(periodic_lddt_warmup_epochs)
        self.current_epoch = 0  # updated externally via set_current_epoch(...)
        
        # Generate run name for directory naming
        self.run_name = None
        if self.use_wandb and wandb is not None and wandb.run is not None:
            # Use wandb run name if available and not None
            if wandb.run.name is not None:
                self.run_name = wandb.run.name
            else:
                # Generate fallback name for offline mode
                self.run_name = _generate_run_name()
                print(f"🔄 Generated run name for offline wandb: {self.run_name}")
        
        # Set up checkpoint directory based on run name
        if checkpoint_dir is None and self.run_name is not None:
            self.checkpoint_dir = f"checkpoints/{self.run_name}"
        elif checkpoint_dir is None:
            self.checkpoint_dir = "checkpoints"
        else:
            self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if self.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Install with: pip install wandb")
    
    def set_current_epoch(self, epoch: int):
        """
        Set the current training epoch (for periodic lDDT warmup).
        Call this once per epoch from your training loop.
        """
        self.current_epoch = int(epoch)
    
    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps using either uniform or logit-normal distribution.
        If fixed_time is set, returns that fixed time for all samples.
        
        Args:
            batch_size: Number of samples to generate
        Returns:
            t: Timestep tensor [batch_size, 1] in [0,1]
        """
        # Fixed time mode (for debugging/testing)
        if self.fixed_time is not None:
            return torch.full((batch_size, 1), self.fixed_time, device=self.device, dtype=torch.float32)
        
        if self.use_logit_normal_resampling:
            # Logit-normal sampling with parameters - ensure device consistency
            u = torch.randn(batch_size, device=self.device) * self.logit_normal_s + self.logit_normal_m
            t_logit = 1 / (1 + torch.exp(-u))
            
            # Mix with uniform sampling
            t_uniform = torch.rand(batch_size, device=self.device)
            t = (1 - self.logit_normal_mix) * t_logit + self.logit_normal_mix * t_uniform
            
            # Apply epsilon adjustment to avoid boundaries
            t = t * (1 - 2 * self.t_eps) + self.t_eps
        else:
            # Original uniform sampling
            t = torch.rand(batch_size, device=self.device)
        
        return t.unsqueeze(-1)
    
    def _sample_timesteps_for_coords_and_lattice(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timesteps for both coordinates and lattice, handling shared/unshared logic.
        
        Args:
            batch_size: Number of samples to generate
        Returns:
            t_cart: Timestep tensor for coordinates [batch_size, 1]
            t_lattice: Timestep tensor for lattice [batch_size, 1]
        """
        if self.shared_time:
            # Use same timestep for both coordinates and lattice
            t_shared = self._sample_timesteps(batch_size)
            return t_shared, t_shared
        else:
            # Sample separate timesteps for coordinates and lattice
            t_cart = self._sample_timesteps(batch_size)
            t_lattice = self._sample_timesteps(batch_size)
            return t_cart, t_lattice
    
    def _reshape_to_batch(self, coords: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reshape coordinates from [total_atoms, 3] to [batch_size, max_atoms, 3].
        
        Args:
            coords: Coordinates [total_atoms, 3]
            batch_indices: Batch indices [total_atoms]
            batch_size: Number of batches
        Returns:
            coords_batch: Reshaped coordinates [batch_size, max_atoms, 3]
        """
        # Get max atoms per batch
        max_atoms = 0
        for i in range(batch_size):
            atoms_in_batch = (batch_indices == i).sum().item()
            max_atoms = max(max_atoms, atoms_in_batch)
        
        # Pad to max_atoms
        coords_batch = torch.zeros(batch_size, max_atoms, 3, device=coords.device)
        for i in range(batch_size):
            mask = (batch_indices == i)
            atoms_in_batch = mask.sum().item()
            if atoms_in_batch > 0:
                coords_batch[i, :atoms_in_batch] = coords[mask]
        
        return coords_batch
    
    def _build_periodic_neighbor_edges(
        self,
        coords_wrapped: torch.Tensor,
        batch_idx: torch.Tensor,
        lattice_mats: Optional[torch.Tensor],
        include_graph_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Build periodic neighbor edges (i, j, k) using vesin.NeighborList for each graph in the batch.
        Returns (edge_index, fractional_shifts, edge_batch) if any edges are found.
        include_graph_mask: optional [num_graphs] bool tensor; if provided, only graphs with True will be processed.
        """
        if (not self.add_periodic_edges) or lattice_mats is None:
            return None
        if self.periodic_edge_cutoff <= 0:
            return None
        
        nl = NeighborList(cutoff=self.periodic_edge_cutoff, full_list=True)
        coords_cpu = coords_wrapped.detach().cpu()
        lattice_cpu = lattice_mats.detach().cpu()
        edge_chunks: List[torch.Tensor] = []
        shift_chunks: List[torch.Tensor] = []
        batch_chunks: List[torch.Tensor] = []
        num_graphs = lattice_cpu.shape[0]
        
        for graph_idx in range(num_graphs):
            if include_graph_mask is not None and not bool(include_graph_mask[graph_idx]):
                continue
            node_indices = torch.nonzero(batch_idx == graph_idx, as_tuple=False).squeeze(-1)
            if node_indices.numel() == 0:
                continue
            node_indices_cpu = node_indices.cpu()
            positions = coords_cpu[node_indices_cpu].numpy()
            lattice = lattice_cpu[graph_idx].numpy()
            
            # Check volume to prevent OOM
            volume = np.abs(np.linalg.det(lattice))
            if volume < 5.0:
                # Skip very small lattices that would cause neighbor list explosion
                continue

            src, dst, shifts = nl.compute(
                quantities="ijS",
                points=positions,
                box=lattice,
                periodic=self.periodic_edge_periodic,
            )
            if src.size == 0:
                continue
            
            edge_attrs = np.hstack((src[:, None], dst[:, None], shifts))
            edge_attrs = np.unique(edge_attrs, axis=0)
            if edge_attrs.size == 0:
                continue
            edge_attrs_tensor = torch.from_numpy(edge_attrs).long()
            local_src = edge_attrs_tensor[:, 0]
            local_dst = edge_attrs_tensor[:, 1]
            frac_shifts = edge_attrs_tensor[:, 2:]
            keep_mask = ~((local_src == local_dst) & (frac_shifts == 0).all(dim=1))
            if not keep_mask.all():
                local_src = local_src[keep_mask]
                local_dst = local_dst[keep_mask]
                frac_shifts = frac_shifts[keep_mask]
            if local_src.numel() == 0:
                continue
            
            global_src = node_indices_cpu[local_src]
            global_dst = node_indices_cpu[local_dst]
            edge_chunks.append(torch.stack([global_src, global_dst], dim=0))
            shift_chunks.append(frac_shifts)
            batch_chunks.append(torch.full((local_src.numel(),), graph_idx, dtype=torch.long))
        
        if not edge_chunks:
            return None
        
        device = coords_wrapped.device
        edge_index = torch.cat(edge_chunks, dim=1).to(device)
        shifts_frac = torch.cat(shift_chunks, dim=0).to(device)
        edge_batch = torch.cat(batch_chunks, dim=0).to(device)
        return edge_index, shifts_frac, edge_batch

    @staticmethod
    def _fractional_shifts_to_cart(
        shifts_frac: torch.Tensor,
        edge_batch: torch.Tensor,
        lattice_mats: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if shifts_frac.numel() == 0:
            return shifts_frac.new_zeros((0, 3), dtype=dtype)
        if lattice_mats is None:
            return shifts_frac.new_zeros((shifts_frac.size(0), 3), dtype=dtype)
        lattice_sel = lattice_mats[edge_batch].to(shifts_frac.device)
        cart = torch.bmm(shifts_frac.to(dtype).unsqueeze(1), lattice_sel).squeeze(1)
        return cart

    @staticmethod
    def _compute_edge_distances(
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        shift_cart: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return coords.new_zeros((0, 1))
        src, dst = edge_index
        shifted_dst = coords[dst] + shift_cart
        diff = coords[src] - shifted_dst
        return diff.norm(dim=-1, keepdim=True)

    @staticmethod
    def _wrap_cartesian_to_unit_cell(
        coords: torch.Tensor,
        batch_idx: torch.Tensor,
        lattice_mats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if lattice_mats is None:
            return coords
        H_selected = lattice_mats[batch_idx]  # [N,3,3]
        # Convert cartesian to fractional using batched utility function
        frac = cart_to_frac_batched(coords, H_selected)  # [N, 3]
        frac = frac - torch.floor(frac)
        # Convert back to cartesian using batched utility function
        coords_wrapped = frac_to_cart_batched(frac, H_selected)  # [N, 3]
        return coords_wrapped

    def _augment_edges_with_periodic_bonds(
        self,
        coords_reference: torch.Tensor,
        batch_indices: torch.Tensor,
        lattice_mats: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        bond_features: torch.Tensor,
        t_cart: Optional[torch.Tensor] = None,
        t_lattice: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally append periodic non-covalent edges and a distance feature.

        Behaviors:
        - If self.add_periodic_edges == False:
            * Only covalent edges are returned.
            * bond_features are returned unchanged (no distance feature).
        - If self.add_periodic_edges == True:
            * A distance feature is always appended so bond feature dimensionality stays constant.
            * Non-covalent (periodic) edges are only added for graphs where BOTH
              t_cart and t_lattice are below self.periodic_edge_time_cutoff (if set).
            * Above cutoff, we keep only covalent edges but still append distances.
        """
        device = coords_reference.device
        coords_reference = coords_reference.detach()
        if lattice_mats is not None:
            lattice_mats = lattice_mats.detach()
        if bond_features.dim() == 1:
            bond_features = bond_features.unsqueeze(-1)
        bond_features = bond_features.to(device).float()
        edge_index = edge_index.to(device)

        if not self.add_periodic_edges:
            return edge_index, bond_features

        feature_dim = bond_features.size(-1)

        coords_wrapped = self._wrap_cartesian_to_unit_cell(coords_reference, batch_indices, lattice_mats)
        include_graph_mask: Optional[torch.Tensor] = None
        if (
            self.periodic_edge_time_cutoff is not None
            and t_cart is not None
            and t_lattice is not None
        ):
            t_cart_b = t_cart.squeeze(-1)
            t_lat_b = t_lattice.squeeze(-1)
            include_graph_mask = (t_cart_b < self.periodic_edge_time_cutoff) & (
                t_lat_b < self.periodic_edge_time_cutoff
            )

        base_shift = coords_reference.new_zeros((edge_index.size(1), 3))
        edge_chunks: List[torch.Tensor] = [edge_index]
        shift_chunks: List[torch.Tensor] = [base_shift]
        feature_chunks: List[torch.Tensor] = [bond_features]

        periodic_edges = self._build_periodic_neighbor_edges(
            coords_wrapped,
            batch_indices,
            lattice_mats,
            include_graph_mask=include_graph_mask,
        )
        if periodic_edges is not None:
            p_edge_index, p_shifts_frac, p_edge_batch = periodic_edges
            shift_cart = self._fractional_shifts_to_cart(
                p_shifts_frac, p_edge_batch, lattice_mats, coords_reference.dtype
            )
            edge_chunks.append(p_edge_index)
            shift_chunks.append(shift_cart)

            new_features = torch.zeros(
                (p_edge_index.size(1), feature_dim),
                dtype=bond_features.dtype,
                device=device,
            )
            if feature_dim > 0:
                new_features[:, 0] = float(self.misc_bond_type_index)
            if feature_dim > 1:
                new_features[:, 1] = 0.0
            feature_chunks.append(new_features)

        augmented_edge_index = (
            torch.cat(edge_chunks, dim=1) if len(edge_chunks) > 1 else edge_chunks[0]
        )
        shift_cart_all = torch.cat(shift_chunks, dim=0)
        bond_features_all = torch.cat(feature_chunks, dim=0)

        distances = self._compute_edge_distances(
            coords_wrapped, augmented_edge_index, shift_cart_all
        )
        bond_features_all = torch.cat([bond_features_all, distances], dim=1)
        return augmented_edge_index, bond_features_all
    
    @staticmethod
    def _mse_per_atom_per_dim(pred: torch.Tensor, target: torch.Tensor,
                              batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Mean-squared error, normalized per atom and per dimension.
        Returns scalar: mean over graphs of (1/N_g * 1/D * sum_{i in g} ||pred_i - target_i||^2 )
        """
        D = pred.size(-1)
        diff2 = (pred - target) ** 2                        # [N, D]
        per_node = diff2.sum(-1) / float(D)                 # [N]
        # average over atoms in each graph
        per_graph = scatter(per_node, batch_idx, dim=0, dim_size=batch_size, reduce="mean")  # [B]
        return per_graph.mean()
    
    @staticmethod
    def _mse_per_dim_batched(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        For lattice (shape [B, D]): return mean over batch of per-dimension MSE.
        """
        D = pred.size(-1)
        return ((pred - target) ** 2).sum(dim=-1).div(float(D)).mean()
    
    def smooth_lddt_loss(self, pred_coords, true_coords, coords_mask, t):
        """
        Compute smooth LDDT loss for crystal structure prediction.
        
        Args:
            pred_coords: Predicted atom coordinates [B, N, 3]
            true_coords: Ground truth atom coordinates [B, N, 3]
            coords_mask: Atom mask [B, N]
            t: Timestep tensor [B]
        Returns:
            smooth_lddt_loss: Scalar loss tensor
        """
        B, N, _ = true_coords.shape
        true_dists = torch.cdist(true_coords, true_coords)

        # Create mask for distances within cutoff
        mask = (true_dists < self.lddt_cutoff).float()
        mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
        mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

        # Compute predicted distances
        pred_dists = torch.cdist(pred_coords, pred_coords)
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute smooth LDDT using sigmoid functions
        eps = (
            (
                (
                    F.sigmoid(0.5 - dist_diff)
                    + F.sigmoid(1.0 - dist_diff)
                    + F.sigmoid(2.0 - dist_diff)
                    + F.sigmoid(4.0 - dist_diff)
                )
                / 4.0
            )
            .view(B, N, N)
            .mean(dim=0)
        )

        # Calculate masked averaging
        num = (eps * mask).sum(dim=(-1, -2))
        den = mask.sum(dim=(-1, -2)).clamp(min=1)
        lddt = num / den
        
        if self.lddt_weight_schedule:
            # Weight more heavily when t is close to 0 (clean data)
            t_weight = 1 + 8 * torch.relu(0.5 - t)
            lddt = (1.0 - lddt) * t_weight
            return lddt.mean()
        else:
            return (1.0 - lddt.mean()) * self.smooth_lddt_loss_weight
    
    def bond_length_loss(
        self,
        denoised_coords: torch.Tensor,   # [N, 3]
        true_coords: torch.Tensor,       # [N, 3]
        edge_index: torch.Tensor,        # [2, E]
        batch_idx: torch.Tensor,         # [N]
        batch_size: int,
    ) -> torch.Tensor:
        """
        Penalize errors in covalent bond lengths:
        mean over graphs of:
            1 / |E_g| * sum_{(i,j) in g} ( ||x̂_i - x̂_j|| - ||x_i - x_j|| )^2
        """
        # Handle no-bond case gracefully
        if edge_index.numel() == 0:
            return denoised_coords.new_tensor(0.0)

        src, dst = edge_index  # [E], [E]

        # True and predicted bond vectors
        true_vec = true_coords[src] - true_coords[dst]        # [E, 3]
        pred_vec = denoised_coords[src] - denoised_coords[dst]  # [E, 3]

        true_len = true_vec.norm(dim=-1)  # [E]
        pred_len = pred_vec.norm(dim=-1)  # [E]

        sq_err = (pred_len - true_len) ** 2  # [E]

        # Assign each edge to its graph via src node batch index
        edge_batch = batch_idx[src]  # [E]
        per_graph = scatter(
            sq_err,
            edge_batch,
            dim=0,
            dim_size=batch_size,
            reduce="mean",
        )  # [B]

        return per_graph.mean()
    
    def periodic_lddt_loss(
        self,
        denoised_coords: torch.Tensor,        # [N,3] from x_t - t_cart * v_cart
        true_coords: torch.Tensor,            # [N,3]
        batch_idx: torch.Tensor,              # [N]
        batch_size: int,
        lattice_true_constrained: torch.Tensor,   # [B,6] ground-truth constrained
        lattice_denoised_constrained: torch.Tensor,  # [B,6] denoised constrained
    ) -> torch.Tensor:
        """
        Periodic smooth-lDDT style loss:
        - Neighbors from ground-truth coords + lattice via NeighborList
        - Distance differences measured under (x, H_true) vs (x_hat, H_hat)
        - Sigmoid thresholds at 0.5, 1.0, 2.0, 4.0 Å, averaged over edges and graphs
        """
        if not self.use_periodic_lddt_loss:
            return denoised_coords.new_tensor(0.0)

        device = denoised_coords.device
        dtype = denoised_coords.dtype

        cutoff = self.periodic_lddt_cutoff if self.periodic_lddt_cutoff is not None else self.lddt_cutoff
        if cutoff <= 0:
            return denoised_coords.new_tensor(0.0)

        # Build lattice matrices
        H_true = lattice_params_to_matrix_torch(lattice_true_constrained)          # [B,3,3]
        H_pred = lattice_params_to_matrix_torch(lattice_denoised_constrained)      # [B,3,3]

        nl = NeighborList(cutoff=cutoff, full_list=True)

        per_graph_score = denoised_coords.new_zeros(batch_size)

        true_coords_cpu = true_coords.detach().cpu()
        H_true_cpu = H_true.detach().cpu()

        for g in range(batch_size):
            node_mask = (batch_idx == g)
            idx = torch.nonzero(node_mask, as_tuple=False).squeeze(-1).cpu()
            if idx.numel() == 0:
                # No atoms in this graph; treat as perfect
                per_graph_score[g] = 1.0
                continue

            # Positions and box for NeighborList (true structure)
            pos_true = true_coords_cpu[idx].numpy()        # [Ng,3]
            box_true = H_true_cpu[g].numpy()               # [3,3]

            src, dst, shifts = nl.compute(
                quantities="ijS",
                points=pos_true,
                box=box_true,
                periodic=True,
            )

            if src.size == 0:
                per_graph_score[g] = 1.0
                continue

            edge_attrs = np.hstack((src[:, None], dst[:, None], shifts))  # [E, 2+3]
            edge_attrs = np.unique(edge_attrs, axis=0)
            if edge_attrs.size == 0:
                per_graph_score[g] = 1.0
                continue

            edge_attrs_t = torch.from_numpy(edge_attrs).to(device=device)
            local_src = edge_attrs_t[:, 0].long()
            local_dst = edge_attrs_t[:, 1].long()
            frac_shifts = edge_attrs_t[:, 2:]  # integer/fractional cell shifts

            # Remove trivial self-edges with zero shift
            keep_mask = ~((local_src == local_dst) & (frac_shifts == 0).all(dim=1))
            if not keep_mask.any():
                per_graph_score[g] = 1.0
                continue

            local_src = local_src[keep_mask]
            local_dst = local_dst[keep_mask]
            frac_shifts = frac_shifts[keep_mask]

            # Map to global indices (idx is on CPU, so move local indices to CPU for indexing)
            global_src = idx[local_src.cpu()].to(device=device)
            global_dst = idx[local_dst.cpu()].to(device=device)

            tau_frac = frac_shifts.to(device=device, dtype=dtype)  # [E,3]

            H_true_g = H_true[g].to(device=device, dtype=dtype)    # [3,3]
            H_pred_g = H_pred[g].to(device=device, dtype=dtype)

            # Periodic displacements in Cartesian using utility function
            shift_true_cart = frac_to_cart(tau_frac, H_true_g)  # [E,3]
            shift_pred_cart = frac_to_cart(tau_frac, H_pred_g)  # [E,3]

            # True & predicted periodic distances
            vec_true = (true_coords[global_dst] + shift_true_cart) - true_coords[global_src]      # [E,3]
            vec_pred = (denoised_coords[global_dst] + shift_pred_cart) - denoised_coords[global_src]  # [E,3]

            d_true = vec_true.norm(dim=-1)  # [E]
            d_pred = vec_pred.norm(dim=-1)  # [E]
            diff = torch.abs(d_pred - d_true)  # [E]

            if diff.numel() == 0:
                per_graph_score[g] = 1.0
                continue

            # Smooth multi-threshold sigmoid average
            s = (
                torch.sigmoid(0.5 - diff)
                + torch.sigmoid(1.0 - diff)
                + torch.sigmoid(2.0 - diff)
                + torch.sigmoid(4.0 - diff)
            ) / 4.0  # [E]

            per_graph_score[g] = s.mean()

        # periodic lDDT ~ average over graphs, loss = 1 - score
        plddt = per_graph_score.mean()
        return (1.0 - plddt) * self.periodic_lddt_loss_weight
    
    def _schedule(self, t: torch.Tensor):
        """
        Return alpha(t), sigma(t), dalpha/dt, dsigma/dt for OT schedule.
        t: tensor [..., 1] with values in [0,1].
        """
        alpha = 1.0 - t
        sigma = t
        dalpha = -torch.ones_like(t)
        dsigma = torch.ones_like(t)
        return alpha, sigma, dalpha, dsigma

    
    def save_checkpoint(self, epoch: int, optimizer, loss: float, is_best: bool = False, 
                       model_config: dict = None, training_config: dict = None):
        """Save model checkpoint with training state."""
        import os
        import torch
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lattice_loss_weight': self.lattice_loss_weight,
            'shared_time': self.shared_time,
        }
        
        # Add optional configs
        if model_config:
            checkpoint['model_config'] = model_config
        if training_config:
            checkpoint['training_config'] = training_config
        
        # Save regular checkpoint
        if epoch % self.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}: {best_path}")
        
        return checkpoint
    
    def _maybe_log_grad_norms(self, step: Optional[int] = None):
        if self.grad_log_every <= 0:
            return
        if step is not None and step % self.grad_log_every != 0:
            return

        groups = compute_grad_norms_by_group(self.model)
        # Pretty print to console
        header = f"[GRAD] step={step}" if step is not None else "[GRAD]"
        print(header + "  (||g||2 / ||w||2 by layer group)")
        for k in sorted(groups.keys()):
            gn = groups[k]["grad_norm"]
            pn = groups[k]["param_norm"]
            ratio = (gn / (pn + 1e-12))
            print(f"   - {k:<28}  g={gn:10.4e}  w={pn:10.4e}  g/w={ratio:8.4e}")

        # Optional: wandb table logging
        if self.use_wandb and wandb is not None and wandb.run is not None:
            data = []
            for k in sorted(groups.keys()):
                data.append([k, groups[k]["grad_norm"], groups[k]["param_norm"]])
            wandb.log({
                f"{self.wandb_prefix}grad_norms": wandb.Table(
                    columns=["group", "grad_norm", "param_norm"], data=data
                )
            }, step=step)
        
    def compute_loss_batch(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss for a batch of crystal data using Cartesian coordinates only.
        """
        cart_coords_0 = batch.whole_cartesian_coords.to(self.device)  # [total_atoms, 3]
        atom_types = batch.atom_types.to(self.device)                 # [total_atoms]
        edge_index = batch.edge_index.to(self.device)                 # [2, total_edges]
        batch_indices = batch.batch.to(self.device)                   # [total_atoms]
        batch_size = batch.num_graphs

        node_features = getattr(batch, 'node_features', None)
        bond_features = getattr(batch, 'bond_features', None)
        if node_features is not None:
            node_features = node_features.to(self.device)
        if bond_features is not None:
            bond_features = bond_features.to(self.device)
        if bond_features is None or bond_features.numel() == 0:
            raise ValueError("bond_features are required: provide RDKit per-edge features aligned with edge_index.")
        
        covalent_edge_index = edge_index
        lat_params_con = batch.lattice_1.to(self.device, dtype=cart_coords_0.dtype).view(batch_size, 6)

        # Lattice: use k_basis or invariant 6D representation
        if self.use_k_basis_representation:
            # Use k_basis directly from batch (no transformation needed)
            if hasattr(batch, 'k_basis') and batch.k_basis is not None and batch.k_basis.numel() > 0:
                lattice_0 = batch.k_basis.to(self.device, dtype=cart_coords_0.dtype).view(batch_size, 6)
            else:
                # Fallback: compute k_basis from cell_1 if available, otherwise from lattice_1
                if hasattr(batch, 'cell_1') and batch.cell_1 is not None and batch.cell_1.numel() > 0:
                    cell_1 = batch.cell_1.to(self.device, dtype=cart_coords_0.dtype).view(batch_size, 3, 3)
                    lattice_0 = lattice_matrix_to_k_basis_torch(cell_1)
                else:
                    # Last resort: convert from lattice params
                    lattice_matrix_0 = lattice_params_to_matrix_torch(lat_params_con)  # [B, 3, 3]
                    lattice_0 = lattice_matrix_to_k_basis_torch(lattice_matrix_0)  # [B, 6]
        else:
            # Original invariant 6D with constrained to unconstrained transformation
            lattice_0 = self.lattice_transform.constrained_to_unconstrained(lat_params_con)

        # Sample times
        t_cart, t_lattice = self._sample_timesteps_for_coords_and_lattice(batch_size)

        # Lattice schedule and targets
        alpha_lat,  sigma_lat,  dalpha_lat,  dsigma_lat  = self._schedule(t_lattice)
        z_lattice      = torch.randn(batch_size, lattice_0.size(1), device=self.device, dtype=lattice_0.dtype)
        lattice_t      = alpha_lat * lattice_0 + sigma_lat * z_lattice
        target_lattice = dalpha_lat * lattice_0 + dsigma_lat * z_lattice

        # Cartesian schedule and targets
        z_cart = torch.randn_like(cart_coords_0)
        alpha_cart, sigma_cart, dalpha_cart, dsigma_cart = self._schedule(t_cart)
        alpha_cart_exp   = alpha_cart[batch_indices]
        sigma_cart_exp   = sigma_cart[batch_indices]
        dalpha_cart_exp  = dalpha_cart[batch_indices]
        dsigma_cart_exp  = dsigma_cart[batch_indices]

        cart_coords_t = alpha_cart_exp * cart_coords_0 + sigma_cart_exp * z_cart
        target_cart   = dalpha_cart_exp * cart_coords_0 + dsigma_cart_exp * z_cart

        # Convert lattice_t to lattice matrix for periodic edge building
        if self.use_k_basis_representation:
            # Convert k_basis back to lattice matrix
            lattice_matrix_t = k_basis_to_lattice_matrix_torch(lattice_t.detach())  # [B, 3, 3]
        else:
            # Convert unconstrained to constrained, then to matrix
            lattice_constrained_t = self.lattice_transform.unconstrained_to_constrained(lattice_t.detach())
            lattice_matrix_t = lattice_params_to_matrix_torch(lattice_constrained_t)
        edge_index, bond_features = self._augment_edges_with_periodic_bonds(
            coords_reference=cart_coords_t.detach(),
            batch_indices=batch_indices,
            lattice_mats=lattice_matrix_t,
            edge_index=edge_index,
            bond_features=bond_features,
            t_cart=t_cart,
            t_lattice=t_lattice,
        )

        # Model prediction (always Cartesian path now)
        predicted_cart_field, predicted_lattice_field = self.model(
            atom_types=atom_types,
            cart_coords=cart_coords_t,
            lattice=lattice_t,
            batch=batch_indices,
            t_cart=t_cart,
            t_lattice=t_lattice,
            edge_index=edge_index,
            node_features=node_features,
            bond_features=bond_features,
        )

        # Per-atom, per-dim MSE (coords) + per-dim MSE (lattice)
        loss_cart = self._mse_per_atom_per_dim(predicted_cart_field, target_cart, batch_indices, batch_size)
        loss_lattice = self._mse_per_dim_batched(predicted_lattice_field, target_lattice)
        total_loss = loss_cart + self.lattice_loss_weight * loss_lattice

        aux_losses: Dict[str, torch.Tensor] = {}

        # We need denoised coords whenever any aux loss depends on them
        need_denoised_coords = self.use_smooth_lddt_loss or self.use_bond_length_loss or self.use_periodic_lddt_loss
        if need_denoised_coords:
            # Same denoising heuristic as before
            denoised_coords = cart_coords_t - predicted_cart_field * t_cart[batch_indices]  # [N, 3]

        # For periodic lDDT we also need denoised lattice
        if self.use_periodic_lddt_loss:
            # Denoise lattice based on representation type
            lattice_denoised = lattice_t - t_lattice * predicted_lattice_field  # [B,6]
            if self.use_k_basis_representation:
                # k_basis -> lattice matrix -> lattice params
                lattice_denoised_matrix = k_basis_to_lattice_matrix_torch(lattice_denoised)  # [B, 3, 3]
                lattice_denoised_constrained = lattice_matrix_to_params_torch(lattice_denoised_matrix)  # [B, 6]
            else:
                # Unconstrained -> constrained
                lattice_denoised_constrained = self.lattice_transform.unconstrained_to_constrained(
                    lattice_denoised
                )  # [B,6]
        else:
            lattice_denoised_constrained = None  # unused

        if self.use_smooth_lddt_loss:
            denoised_coords_batch = self._reshape_to_batch(denoised_coords, batch_indices, batch_size)
            true_coords_batch = self._reshape_to_batch(cart_coords_0, batch_indices, batch_size)
            coords_mask = torch.ones(batch_size, denoised_coords_batch.shape[1], device=self.device)

            smooth_lddt_loss = self.smooth_lddt_loss(
                denoised_coords_batch,
                true_coords_batch,
                coords_mask,
                t_cart.squeeze(-1),
            )
            total_loss = total_loss + smooth_lddt_loss
            aux_losses['smooth_lddt_loss'] = smooth_lddt_loss

        if self.use_bond_length_loss:
            bond_len_loss = self.bond_length_loss(
                denoised_coords=denoised_coords,
                true_coords=cart_coords_0,
                edge_index=covalent_edge_index,
                batch_idx=batch_indices,
                batch_size=batch_size,
            )
            total_loss = total_loss + self.bond_length_loss_weight * bond_len_loss
            aux_losses['bond_length_loss'] = bond_len_loss

        # Periodic lDDT with epoch-based warmup
        if self.use_periodic_lddt_loss and (self.current_epoch >= self.periodic_lddt_warmup_epochs):
            periodic_lddt_loss = self.periodic_lddt_loss(
                denoised_coords=denoised_coords,
                true_coords=cart_coords_0,
                batch_idx=batch_indices,
                batch_size=batch_size,
                lattice_true_constrained=lat_params_con,
                lattice_denoised_constrained=lattice_denoised_constrained,
            )
            total_loss = total_loss + periodic_lddt_loss
            aux_losses['periodic_lddt_loss'] = periodic_lddt_loss
        elif self.use_periodic_lddt_loss and (self.current_epoch < self.periodic_lddt_warmup_epochs):
            # Loss is enabled but still in warmup period
            aux_losses['periodic_lddt_loss'] = denoised_coords.new_tensor(0.0)

        # Build output dict
        out = {
            'total_loss': total_loss,
            'cart_loss': loss_cart,
            'lattice_loss': loss_lattice,
        }
        out.update(aux_losses)
        
        # Include denoised structures for visualization when periodic lDDT is enabled
        if self.use_periodic_lddt_loss and need_denoised_coords:
            out['_denoised_coords'] = denoised_coords.detach()
            out['_denoised_lattice'] = lattice_denoised_constrained.detach() if lattice_denoised_constrained is not None else None
            out['_t_cart'] = t_cart.detach()
            out['_t_lattice'] = t_lattice.detach()
            out['_batch_indices'] = batch_indices.detach()
            out['_lat_params_con'] = lat_params_con.detach()
        
        return out

    def compute_loss(self, batch_or_crystal_data) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss for crystal data (unified batched approach).
        
        Args:
            batch_or_crystal_data: Either PyG Batch object or dictionary (converted to batch)
        Returns:
            loss_dict: Dictionary containing total loss and component losses
        """
        # Convert dictionary to batch if needed
        if isinstance(batch_or_crystal_data, dict):
            batch = self._dict_to_batch(batch_or_crystal_data)
        else:
            batch = batch_or_crystal_data
            
        return self.compute_loss_batch(batch)
    
    def _dict_to_batch(self, crystal_data: Dict[str, torch.Tensor]) -> Batch:
        """Convert a single crystal dictionary to a PyG Batch object."""
        from torch_geometric.data import Data, Batch
        
        # Create a Data object
        data = Data(
            whole_cartesian_coords=crystal_data['whole_cartesian_coords'],
            atom_types=crystal_data['atom_types'], 
            edge_index=crystal_data['edge_index'],
            lattice_1=crystal_data['lattice_1'],
            refcode=crystal_data.get('refcode', 'single_crystal')
        )
        
        # Convert to batch (batch_size=1)
        batch = Batch.from_data_list([data])
        return batch
    
    def train_step_batch(self, batch: Batch, optimizer: optim.Optimizer, step: int = None) -> Dict[str, float]:
        """Single training step with batched data."""
        optimizer.zero_grad()
        loss_dict = self.compute_loss_batch(batch)
        loss_dict['total_loss'].backward()
        
        self._maybe_log_grad_norms(step=step)
        
        optimizer.step()
        
        # Convert to float for logging (skip visualization tensors that start with '_')
        loss_dict_float = {k: v.item() for k, v in loss_dict.items() if not k.startswith('_')}
        
        # Log to wandb if enabled and initialized
        if self.use_wandb and wandb is not None and wandb.run is not None:
            wandb_dict = {}
            for k, v in loss_dict_float.items():
                wandb_dict[f"{self.wandb_prefix}{k}"] = v
            if step is not None:
                wandb.log(wandb_dict, step=step)
            else:
                wandb.log(wandb_dict)
        
        return loss_dict_float

    def train_step(self, batch_or_crystal_data, optimizer: optim.Optimizer, step: int = None) -> Dict[str, float]:
        """Single training step (unified approach - handles both batches and dicts)."""
        optimizer.zero_grad()
        loss_dict = self.compute_loss(batch_or_crystal_data)
        loss_dict['total_loss'].backward()
        
        self._maybe_log_grad_norms(step=step)
        
        optimizer.step()
        
        # Convert to float for logging (skip visualization tensors that start with '_')
        loss_dict_float = {k: v.item() for k, v in loss_dict.items() if not k.startswith('_')}
        
        # Log to wandb if enabled and initialized
        if self.use_wandb and wandb is not None and wandb.run is not None:
            wandb_dict = {}
            for k, v in loss_dict_float.items():
                wandb_dict[f"{self.wandb_prefix}{k}"] = v
            if step is not None:
                wandb.log(wandb_dict, step=step)
            else:
                wandb.log(wandb_dict)
        
        return loss_dict_float
    
    def validate_step(self, batch_or_crystal_data) -> Dict[str, float]:
        """Single validation step (no gradient updates)."""
        self.model.eval()
        with torch.no_grad():
            loss_dict = self.compute_loss(batch_or_crystal_data)
            # Convert to float for logging (skip visualization tensors that start with '_')
            loss_dict_float = {k: v.item() for k, v in loss_dict.items() if not k.startswith('_')}
        self.model.train()
        return loss_dict_float
    
    def validate_on_loader(self, dataloader, max_batches: int = None) -> Dict[str, float]:
        """Validate on a DataLoader and return average losses."""
        self.model.eval()
        val_losses = {'total': [], 'cart': [], 'lattice': []}
        
        # Track auxiliary losses if they exist
        aux_loss_keys = ['smooth_lddt_loss', 'bond_length_loss', 'periodic_lddt_loss']
        for key in aux_loss_keys:
            val_losses[key] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                loss_dict = self.compute_loss_batch(batch)
                val_losses['total'].append(loss_dict['total_loss'].item())
                val_losses['cart'].append(loss_dict['cart_loss'].item())
                val_losses['lattice'].append(loss_dict['lattice_loss'].item())
                
                # Track auxiliary losses if present
                for key in aux_loss_keys:
                    if key in loss_dict:
                        val_losses[key].append(loss_dict[key].item())
        
        self.model.train()
        
        # Return average losses
        if len(val_losses['total']) == 0:
            result = {'total_loss': 0.0, 'cart_loss': 0.0, 'lattice_loss': 0.0}
            for key in aux_loss_keys:
                result[key] = 0.0
            return result
        
        result = {
            'total_loss': sum(val_losses['total']) / len(val_losses['total']),
            'cart_loss': sum(val_losses['cart']) / len(val_losses['cart']),
            'lattice_loss': sum(val_losses['lattice']) / len(val_losses['lattice'])
        }
        
        # Add auxiliary losses if they were computed
        for key in aux_loss_keys:
            if len(val_losses[key]) > 0:
                result[key] = sum(val_losses[key]) / len(val_losses[key])
            else:
                result[key] = 0.0
        
        return result

    def temper_vector_field(self, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor,
                            lam: float, eps: float = 1e-4) -> torch.Tensor:
        """
        Low-temperature tempering for the vector field.
        For OT schedule (alpha=1-t), we use the exact conversion:
            v_λ = λ v + (λ-1) * x / alpha(t)
        """
        alpha, _, _, _ = self._schedule(t)    # [B,1] alpha(t)
        alpha = torch.clamp(alpha, min=eps)

        # OT schedule tempering
        return lam * v + (lam - 1.0) * (x / alpha if x.dim() == 2 else x / alpha)

    def sample(self, crystal_template: Dict[str, torch.Tensor], n_steps: int = 100, low_temperature_lambda: float = 1.0, 
               coords_time_grid_type: str = "linear", lattice_time_grid_type: str = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate crystal structure using ODE integration in Cartesian coordinates only.
        """
        self.model.eval()
        with torch.no_grad():
            atom_types = crystal_template['atom_types'].to(self.device)
            edge_index = crystal_template['edge_index'].to(self.device)
            n_atoms = len(atom_types)
            
            # Detect batch size
            if 'batch' in crystal_template:
                batch = crystal_template['batch'].to(self.device)
                batch_size = batch.max().item() + 1
            else:
                batch_size = 1
                batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

            node_features = crystal_template.get('node_features', None)
            bond_features = crystal_template.get('bond_features', None)
            if node_features is not None:
                node_features = node_features.to(self.device)
            if bond_features is not None:
                bond_features = bond_features.to(self.device)
            if bond_features is None or bond_features.numel() == 0:
                raise ValueError("bond_features are required: provide RDKit per-edge features aligned with edge_index.")

            covalent_edge_index = edge_index
            covalent_bond_features = bond_features

            # Initialize lattice state (k_basis or invariant representation)
            # Support batched sampling
            lattice_state = torch.randn(batch_size, 6, device=self.device)

            coords_time_grid, lattice_time_grid = create_dual_time_grids(
                n_steps, coords_time_grid_type, lattice_time_grid_type, self.device
            )

            # Cartesian init
            cart_coords = torch.randn(n_atoms, 3, device=self.device)

            for step in range(n_steps):
                t_cart_val = 1.0 - coords_time_grid[step]
                t_lattice_val = 1.0 - lattice_time_grid[step]

                if self.shared_time:
                    t_shared = torch.full((batch_size, 1), t_cart_val, device=self.device)
                    t_cart = t_shared
                    t_lattice = t_shared
                else:
                    t_cart = torch.full((batch_size, 1), t_cart_val, device=self.device)
                    t_lattice = torch.full((batch_size, 1), t_lattice_val, device=self.device)

                # Convert lattice_state to lattice matrix for periodic edge building
                if self.use_k_basis_representation:
                    # k_basis -> lattice matrix
                    lattice_mats = k_basis_to_lattice_matrix_torch(lattice_state.detach())  # [B, 3, 3]
                else:
                    # Unconstrained -> constrained -> lattice matrix
                    # lattice_state is [B, 6]
                    lattice_constrained = self.lattice_transform.unconstrained_to_constrained(lattice_state.detach())
                    lattice_mats = lattice_params_to_matrix_torch(lattice_constrained)
                
                edge_index_aug, bond_features_aug = self._augment_edges_with_periodic_bonds(
                    coords_reference=cart_coords.detach(),
                    batch_indices=batch,
                    lattice_mats=lattice_mats,
                    edge_index=covalent_edge_index,
                    bond_features=covalent_bond_features,
                    t_cart=t_cart,
                    t_lattice=t_lattice,
                )

                v_cart, v_lattice = self.model(
                    atom_types=atom_types,
                    cart_coords=cart_coords,
                    lattice=lattice_state,
                    batch=batch,
                    t_cart=t_cart,
                    t_lattice=t_lattice,
                    edge_index=edge_index_aug,
                    node_features=node_features,
                    bond_features=bond_features_aug,
                )

                if low_temperature_lambda != 1.0:
                    v_cart    = self.temper_vector_field(cart_coords, v_cart, t_cart, low_temperature_lambda)
                    v_lattice = self.temper_vector_field(lattice_state, v_lattice, t_lattice, low_temperature_lambda)

                if step < n_steps - 1:
                    dt_cart = coords_time_grid[step + 1] - coords_time_grid[step]
                    dt_lat  = lattice_time_grid[step + 1] - lattice_time_grid[step]
                else:
                    dt_cart = coords_time_grid[step] - coords_time_grid[step - 1] if step > 0 else 1.0 / n_steps
                    dt_lat  = lattice_time_grid[step] - lattice_time_grid[step - 1] if step > 0 else 1.0 / n_steps

                cart_coords   = cart_coords - dt_cart * v_cart
                lattice_state = lattice_state - dt_lat * v_lattice

            # Convert final lattice_state to lattice_params (constrained)
            if self.use_k_basis_representation:
                # k_basis -> lattice matrix -> lattice params
                final_lattice_matrix = k_basis_to_lattice_matrix_torch(lattice_state)  # [1, 3, 3]
                final_lattice_constrained = lattice_matrix_to_params_torch(final_lattice_matrix).squeeze(0)  # [6]
            else:
                # Unconstrained -> constrained
                final_lattice_constrained = self.lattice_transform.unconstrained_to_constrained(lattice_state).squeeze(0)
            sampled_cart_coords = cart_coords

        self.model.train()
        return sampled_cart_coords, final_lattice_constrained


def draw_unit_cell_box(ax, lattice_matrix, center=None, color='black', alpha=0.5):
    """Draw unit cell box in 3D space, optionally centered at a specific point."""
    import numpy as np
    
    # Define unit cell vertices in fractional coordinates
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])
    
    # Transform to Cartesian coordinates using utility function
    vertices_cart = frac_to_cart(vertices, lattice_matrix)
    
    # If center is provided, shift the lattice box to be centered around that point
    if center is not None:
        # Calculate the current center of the lattice box
        lattice_center = vertices_cart.mean(axis=0)
        # Shift vertices to center the box around the desired center
        vertices_cart = vertices_cart - lattice_center + center
    
    # Define edges of the unit cell
    edges = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], 
        [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
    ]
    
    # Draw edges
    for edge in edges:
        v1, v2 = edge
        ax.plot([vertices_cart[v1, 0], vertices_cart[v2, 0]],
                [vertices_cart[v1, 1], vertices_cart[v2, 1]],
                [vertices_cart[v1, 2], vertices_cart[v2, 2]], 
                color=color, alpha=alpha, linewidth=1.5)


def visualize_crystal_cartesian_advanced(crystal_data, generated_coords, crystal_name, epoch, structure_type="original", generated_lattice=None):
    """Advanced crystal visualization for cartesian coordinates with PBC filtering, centering, and molecule coloring."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data - use whole_cartesian_coords for cartesian model
    cart_coords = crystal_data['whole_cartesian_coords'].cpu().numpy() if structure_type == "original" else generated_coords.cpu().numpy()
    edge_index = crystal_data.get('edge_index', None)
    
    # Use generated lattice if provided, otherwise use ground truth lattice
    if structure_type == "generated" and generated_lattice is not None:
        lattice = generated_lattice.cpu().numpy() if hasattr(generated_lattice, 'cpu') else generated_lattice
    else:
        lattice = crystal_data['lattice_1'].cpu().numpy()
    
    # Convert lattice to matrix form for visualization
    from packflow.utils.crystal_utils import lattice_params_to_matrix
    lattice_matrix = lattice_params_to_matrix(
        lattice[0], lattice[1], lattice[2],
        lattice[3], lattice[4], lattice[5]
    )
    
    # Get atom types for coloring
    atom_types = crystal_data['atom_types'].cpu().numpy()
    
    # Create atom type to color mapping
    unique_atom_types = np.unique(atom_types)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_atom_types)))
    atom_color_map = {atom_type: colors[i] for i, atom_type in enumerate(unique_atom_types)}
    atom_colors = [atom_color_map[atom_type] for atom_type in atom_types]
    
    # Center the coordinates by subtracting mean
    cart_coords_centered = cart_coords - np.mean(cart_coords, axis=0)
    
    # Convert edge_index to numpy for processing
    edge_index_np = edge_index.cpu().numpy() if edge_index is not None else np.array([[], []])
    
    # Use all bonds without filtering
    valid_bond_indices = list(range(edge_index.shape[1])) if edge_index is not None and edge_index.shape[1] > 0 else []
    
    return {
        'cart_coords_centered': cart_coords_centered,
        'lattice_matrix': lattice_matrix,
        'colors': atom_colors,
        'edge_index_np': edge_index_np,
        'valid_bond_indices': valid_bond_indices,
        'lattice': lattice
    }


def visualize_crystal_lambda_comparison(crystal_data, generated_samples_dict, crystal_name, save_path=None):
    """
    Visualize crystal structure with different lambda values for comparison.
    
    Args:
        crystal_data: Original crystal data dictionary
        generated_samples_dict: Dictionary with lambda values as keys and (cart_coords, lattice) as values
        crystal_name: Name of the crystal
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate optimal grid layout for large number of lambda values
    n_samples = len(generated_samples_dict)
    n_total = 1 + n_samples  # original + generated samples
    
    # For many samples, use a grid layout instead of a single row
    if n_total <= 5:
        n_rows, n_cols = 1, n_total
        fig_height = 5
    elif n_total <= 10:
        n_rows, n_cols = 2, 5
        fig_height = 10
    else:
        n_rows = int(np.ceil(n_total / 5))
        n_cols = 5
        fig_height = 5 * n_rows
    
    fig = plt.figure(figsize=(5 * n_cols, fig_height))
    
    # Plot original crystal
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    orig_viz = visualize_crystal_cartesian_advanced(crystal_data, None, crystal_name, 0, "original")
    
    if orig_viz is not None:
        # Scatter plot of atoms
        ax1.scatter(orig_viz['cart_coords_centered'][:, 0], 
                   orig_viz['cart_coords_centered'][:, 1], 
                   orig_viz['cart_coords_centered'][:, 2], 
                   c=orig_viz['colors'], s=60, alpha=0.8)  # Smaller points for grid layout
        
        # Draw lattice box
        molecular_center = orig_viz['cart_coords_centered'].mean(axis=0)
        draw_unit_cell_box(ax1, orig_viz['lattice_matrix'], center=molecular_center, color='black', alpha=0.5)
        
        # Draw bonds
        for bond_idx in orig_viz['valid_bond_indices']:
            atom1, atom2 = orig_viz['edge_index_np'][:, bond_idx]
            if atom1 < len(orig_viz['cart_coords_centered']) and atom2 < len(orig_viz['cart_coords_centered']):
                ax1.plot([orig_viz['cart_coords_centered'][atom1, 0], orig_viz['cart_coords_centered'][atom2, 0]],
                        [orig_viz['cart_coords_centered'][atom1, 1], orig_viz['cart_coords_centered'][atom2, 1]],
                        [orig_viz['cart_coords_centered'][atom1, 2], orig_viz['cart_coords_centered'][atom2, 2]], 
                        'k-', alpha=0.6, linewidth=1.2)  # Thinner lines for grid layout
    
    ax1.set_xlabel('X (Å)', fontsize=8)
    ax1.set_ylabel('Y (Å)', fontsize=8)
    ax1.set_zlabel('Z (Å)', fontsize=8)
    ax1.set_title(f'Original\n{crystal_name}', fontsize=10)
    ax1.view_init(elev=20, azim=45)
    
    # Plot generated structures for each lambda
    for i, (lam, (generated_coords, generated_lattice)) in enumerate(sorted(generated_samples_dict.items())):
        subplot_idx = i + 2  # Start from position 2 (after original)
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection='3d')
        
        gen_viz = visualize_crystal_cartesian_advanced(crystal_data, generated_coords, crystal_name, 0, "generated", generated_lattice)
        
        if gen_viz is not None:
            # Scatter plot of atoms
            ax.scatter(gen_viz['cart_coords_centered'][:, 0], 
                      gen_viz['cart_coords_centered'][:, 1], 
                      gen_viz['cart_coords_centered'][:, 2], 
                      c=gen_viz['colors'], s=60, alpha=0.8)  # Smaller points for grid layout
            
            # Draw lattice box
            molecular_center = gen_viz['cart_coords_centered'].mean(axis=0)
            draw_unit_cell_box(ax, gen_viz['lattice_matrix'], center=molecular_center, color='black', alpha=0.5)
            
            # Draw bonds
            for bond_idx in gen_viz['valid_bond_indices']:
                atom1, atom2 = gen_viz['edge_index_np'][:, bond_idx]
                if atom1 < len(gen_viz['cart_coords_centered']) and atom2 < len(gen_viz['cart_coords_centered']):
                    ax.plot([gen_viz['cart_coords_centered'][atom1, 0], gen_viz['cart_coords_centered'][atom2, 0]],
                           [gen_viz['cart_coords_centered'][atom1, 1], gen_viz['cart_coords_centered'][atom2, 1]],
                           [gen_viz['cart_coords_centered'][atom1, 2], gen_viz['cart_coords_centered'][atom2, 2]], 
                           'k-', alpha=0.6, linewidth=1.2)  # Thinner lines for grid layout
        
        ax.set_xlabel('X (Å)', fontsize=8)
        ax.set_ylabel('Y (Å)', fontsize=8)
        ax.set_zlabel('Z (Å)', fontsize=8)
        ax.set_title(f'λ={lam}', fontsize=10)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lambda comparison visualization saved to: {save_path}")
    
    plt.close()


def sample_crystals_with_different_lambdas(flow_matching, crystal_data, lambda_values, n_steps=500, 
                                         coords_time_grid_type="linear", lattice_time_grid_type="linear"):
    """
    Sample a crystal with different lambda values for temperature comparison.
    
    Args:
        flow_matching: CrystalFlowMatching instance
        crystal_data: Crystal data dictionary
        lambda_values: List of lambda values to test
        n_steps: Number of sampling steps
        coords_time_grid_type: Time grid type for coordinates ("linear", "quadratic", "exponential")
        lattice_time_grid_type: Time grid type for lattice ("linear", "quadratic", "exponential")
        
    Returns:
        Dictionary with lambda values as keys and (cart_coords, lattice) tuples as values
    """
    generated_samples = {}
    
    for lam in lambda_values:
        try:
            print(f"Sampling with lambda={lam}...")
            cart_coords, lattice = flow_matching.sample(
                crystal_data, n_steps=n_steps, low_temperature_lambda=lam,
                coords_time_grid_type=coords_time_grid_type,
                lattice_time_grid_type=lattice_time_grid_type
            )
            generated_samples[lam] = (cart_coords, lattice)
        except Exception as e:
            print(f"Warning: Sampling failed for lambda={lam}: {e}")
            continue
    
    return generated_samples


def sample_and_visualize_lambda_comparison(flow_matching, data_loader, n_crystals=3, lambda_values=[1.0, 1.5, 2.0, 2.5], n_steps=500, save_dir='plots',
                                         coords_time_grid_type="linear", lattice_time_grid_type="linear"):
    """
    Sample crystals with different lambda values and create comparison visualizations.
    
    Args:
        flow_matching: CrystalFlowMatching instance
        data_loader: DataLoader for validation data
        n_crystals: Number of crystals to process
        lambda_values: List of lambda values to test
        n_steps: Number of sampling steps
        save_dir: Directory to save plots
        coords_time_grid_type: Time grid type for coordinates ("linear", "quadratic", "exponential")
        lattice_time_grid_type: Time grid type for lattice ("linear", "quadratic", "exponential")
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    for batch in data_loader:
        if count >= n_crystals:
            break
            
        # Process each crystal in the batch
        batch_split = batch.to_data_list()
        
        for crystal_data_obj in batch_split:
            if count >= n_crystals:
                break
                
            # Convert to dictionary format
            crystal_data = {
                'whole_cartesian_coords': crystal_data_obj.whole_cartesian_coords,
                'atom_types': crystal_data_obj.atom_types,
                'edge_index': crystal_data_obj.edge_index,
                'lattice_1': crystal_data_obj.lattice_1,
                'refcode': getattr(crystal_data_obj, 'refcode', f'Crystal_{count}')
            }
            
            crystal_name = crystal_data['refcode']
            print(f"\n🔬 Processing crystal {count + 1}/{n_crystals}: {crystal_name}")
            
            # Sample with different lambda values
            generated_samples_dict = sample_crystals_with_different_lambdas(
                flow_matching, crystal_data, lambda_values, n_steps,
                coords_time_grid_type, lattice_time_grid_type
            )
            
            if generated_samples_dict:
                # Create comparison visualization
                save_path = os.path.join(save_dir, f'lambda_comparison_{crystal_name}_{count}.png')
                visualize_crystal_lambda_comparison(
                    crystal_data, generated_samples_dict, crystal_name, save_path
                )
                
                count += 1
            else:
                print(f"Warning: No successful samples for crystal {crystal_name}")
    
    print(f"\nCompleted lambda comparison for {count} crystals")
    return count
