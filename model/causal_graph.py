import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphLearner(nn.Module):
    def __init__(self, slot_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(slot_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, N_slots, D_slot]
        Returns:
            adjacency: [B, N_slots, N_slots]
        """
        B, N, D = slots.shape
        
        # Compute edge features for all pairs
        slots_i = slots.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        slots_j = slots.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        
        edge_features = torch.cat([
            slots_i,
            slots_j,
            slots_i - slots_j,
            slots_i * slots_j
        ], dim=-1)  # [B, N, N, 4D]
        
        # Predict edge weights
        logits = self.edge_mlp(edge_features).squeeze(-1)  # [B, N, N]
        adjacency = torch.sigmoid(logits)
        
        return adjacency
    
    @staticmethod
    def acyclicity_constraint(adj: torch.Tensor) -> torch.Tensor:
        """
        Compute h(A) = tr[exp(A ∘ A)] - N
        which equals 0 iff A is acyclic
        """
        N = adj.size(-1)
        A_squared = adj * adj
        
        # Matrix exponential via eigendecomposition (more stable)
        # For small matrices, direct computation works
        exp_A = torch.matrix_exp(A_squared)
        h = torch.diagonal(exp_A, dim1=-2, dim2=-1).sum(-1) - N
        
        return h
    
    @staticmethod
    def sparsity_loss(adj: torch.Tensor) -> torch.Tensor:
        """L1 norm for sparsity"""
        return adj.abs().sum(dim=(-2, -1)).mean()
