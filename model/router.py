import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        working_mem_dim: int,
        num_mechanisms: int,
        budget: int,
        gumbel_temp: float = 1.0
    ):
        super().__init__()
        self.num_mechanisms = num_mechanisms
        self.budget = budget
        self.gumbel_temp = gumbel_temp
        
        self.routing_net = nn.Sequential(
            nn.Linear(slot_dim + working_mem_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_mechanisms)
        )
    
    def forward(self, slots: torch.Tensor, working_mem: torch.Tensor, training: bool = True):
        """
        Args:
            slots: [B, N_slots, D_slot]
            working_mem: [B, D_working]
            training: whether to use Gumbel-Softmax
        Returns:
            gates: [B, M] mechanism activation
        """
        # Pool slots
        slots_pooled = slots.mean(dim=1)  # [B, D_slot]
        
        # Routing scores
        routing_input = torch.cat([slots_pooled, working_mem], dim=-1)
        logits = self.routing_net(routing_input)  # [B, M]
        
        if training:
            # Gumbel-Softmax for differentiable sampling
            gates = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=False)
        else:
            # Hard top-K selection
            _, topk_indices = torch.topk(logits, self.budget, dim=-1)
            gates = torch.zeros_like(logits)
            gates.scatter_(1, topk_indices, 1.0)
        
        return gates
    
    def update_temperature(self, decay: float):
        """Anneal Gumbel temperature"""
        self.gumbel_temp = max(0.1, self.gumbel_temp * decay)
