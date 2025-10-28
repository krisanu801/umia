import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DynamicsMechanism(nn.Module):
    """Physics-like continuous dynamics over causal graph"""
    def __init__(self, slot_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        self.node_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim)
        )
    
    def forward(
        self,
        slots: torch.Tensor,
        adjacency: torch.Tensor,
        working_mem: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = slots.shape
        
        # Message passing
        messages = []
        for i in range(N):
            msg_i = torch.zeros(B, D, device=slots.device)
            for j in range(N):
                if i != j:
                    edge_feat = torch.cat([slots[:, i], slots[:, j]], dim=-1)
                    msg_i += adjacency[:, i, j:j+1] * self.edge_net(edge_feat)
            messages.append(msg_i)
        
        messages = torch.stack(messages, dim=1)  # [B, N, D]
        
        # Node update
        node_input = torch.cat([slots, messages], dim=-1)
        delta = self.node_net(node_input)
        
        return slots + delta


class ReasoningMechanism(nn.Module):
    """Graph neural network for relational reasoning"""
    def __init__(self, slot_dim: int, hidden_dim: int):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim)
        )
    
    def forward(self, slots: torch.Tensor, adjacency: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, D = slots.shape
        
        # Compute messages
        slots_i = slots.unsqueeze(2).expand(B, N, N, D)
        slots_j = slots.unsqueeze(1).expand(B, N, N, D)
        
        edge_input = torch.cat([slots_i, slots_j], dim=-1)
        messages = self.message_net(edge_input)  # [B, N, N, D]
        
        # Aggregate with adjacency
        adjacency_exp = adjacency.unsqueeze(-1)  # [B, N, N, 1]
        aggregated = (adjacency_exp * messages).sum(dim=2)  # [B, N, D]
        
        # Update slots
        update_input = torch.cat([slots, aggregated], dim=-1)
        delta = self.update_net(update_input)
        
        return slots + delta


class AttentionMechanism(nn.Module):
    """Transformer-style self-attention"""
    def __init__(self, slot_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = slot_dim // num_heads
        
        self.qkv = nn.Linear(slot_dim, slot_dim * 3)
        self.proj = nn.Linear(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim)
        )
        self.norm2 = nn.LayerNorm(slot_dim)
    
    def forward(self, slots: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, D = slots.shape
        
        # Multi-head attention
        slots_norm = self.norm(slots)
        qkv = self.qkv(slots_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, H, N, D_head]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [B, H, N, D_head]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        slots = slots + out
        
        # FFN
        slots = slots + self.ffn(self.norm2(slots))
        
        return slots


class MemoryMechanism(nn.Module):
    """Modulate slots with retrieved memory"""
    def __init__(self, slot_dim: int, memory_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(slot_dim + memory_dim, slot_dim),
            nn.Sigmoid()
        )
        
        self.modulation_net = nn.Sequential(
            nn.Linear(slot_dim + memory_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )
    
    def forward(self, slots: torch.Tensor, memory_readout: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, D = slots.shape
        
        # Expand memory to all slots
        memory_exp = memory_readout.unsqueeze(1).expand(B, N, -1)
        
        # Compute gate
        gate_input = torch.cat([slots, memory_exp], dim=-1)
        gate = self.gate_net(gate_input)
        
        # Modulate
        modulation = self.modulation_net(gate_input)
        
        return slots + gate * modulation
