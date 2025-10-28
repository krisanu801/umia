import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        num_heads: int = 4,
        eps: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.num_heads = num_heads
        self.eps = eps
        self.scale = (slot_dim // num_heads) ** -0.5
        
        # Slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        # Projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)
        
        # Slot update (GRU-like)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim)
        )
        
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, L, D_in] input features
        Returns:
            slots: [B, N_slots, D_slot]
            attn: [B, N_slots, L] attention weights
        """
        B, L, D_in = inputs.shape
        
        # Initialize slots
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, L, D_slot]
        v = self.project_v(inputs)  # [B, L, D_slot]
        
        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Compute attention
            q = self.project_q(slots)  # [B, N_slots, D_slot]
            
            # Multi-head attention over inputs
            attn_logits = torch.einsum('bnd,bld->bnl', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots
            
            # Weighted mean
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum('bnl,bld->bnd', attn_norm, v)
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)
            
            # MLP residual
            slots = slots + self.mlp(slots)
        
        return slots, attn


class SlotEncoder(nn.Module):
    """Complete encoder: input → features → slots"""
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.slot_attention = SlotAttention(
            num_slots=config.num_slots,
            slot_dim=config.slot_dim,
            input_dim=config.hidden_dim,
            num_iterations=config.slot_attn_iters,
            num_heads=config.slot_attn_heads
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, D_in]
        Returns:
            slots: [B, N_slots, D_slot]
            attn: [B, N_slots, L]
        """
        features = self.input_proj(x)
        slots, attn = self.slot_attention(features)
        return slots, attn
