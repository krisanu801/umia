import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .slot_attention import SlotEncoder
from .causal_graph import CausalGraphLearner
from .memory import ExternalMemory
from .mechanisms import DynamicsMechanism, ReasoningMechanism, AttentionMechanism, MemoryMechanism
from .router import Router

class UMIA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = SlotEncoder(config)
        self.causal_graph = CausalGraphLearner(config.slot_dim)
        self.external_memory = ExternalMemory(
            config.memory_size,
            config.memory_key_dim,
            config.memory_val_dim,
            config.working_mem_dim
        )
        
        # Working memory (GRU controller)
        self.working_memory = nn.GRUCell(
            config.slot_dim + config.memory_val_dim,
            config.working_mem_dim
        )
        
        # Mechanism library
        self.mechanisms = nn.ModuleList([
            DynamicsMechanism(config.slot_dim, config.hidden_dim),
            ReasoningMechanism(config.slot_dim, config.hidden_dim),
            AttentionMechanism(config.slot_dim),
            MemoryMechanism(config.slot_dim, config.memory_val_dim),
        ])
        
        # Pad to num_mechanisms if needed
        while len(self.mechanisms) < config.num_mechanisms:
            self.mechanisms.append(
                DynamicsMechanism(config.slot_dim, config.hidden_dim)
            )
        
        # Router
        self.router = Router(
            config.slot_dim,
            config.working_mem_dim,
            config.num_mechanisms,
            config.routing_budget
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.slot_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Initialize working memory
        self.register_buffer(
            'h_0',
            torch.zeros(1, config.working_mem_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        working_mem: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, D_in] input sequence
            working_mem: [B, D_working] or None
            intervention_mask: [B, N_slots] binary mask for interventions
            intervention_values: [B, N_slots, D_slot] values for intervened slots
        Returns:
            Dictionary with all outputs
        """
        B = x.size(0)
        
        # Initialize working memory if needed
        if working_mem is None:
            working_mem = self.h_0.expand(B, -1)
        
        # 1. Perception: encode to slots
        slots, slot_attn = self.encoder(x)  # [B, N_slots, D_slot]
        
        # Apply interventions if provided
        if intervention_mask is not None and intervention_values is not None:
            mask_expanded = intervention_mask.unsqueeze(-1)  # [B, N_slots, 1]
            slots = slots * (1 - mask_expanded) + intervention_values * mask_expanded
        
        # 2. Learn causal graph
        adjacency = self.causal_graph(slots)  # [B, N_slots, N_slots]
        
        # 3. Memory operations
        memory_readout = self.external_memory.read(working_mem)  # [B, D_val]
        
        # 4. Update working memory
        slots_pooled = slots.mean(dim=1)
        working_mem_input = torch.cat([slots_pooled, memory_readout], dim=-1)
        working_mem_new = self.working_memory(working_mem_input, working_mem)
        
        # 5. Route mechanisms
        gates = self.router(slots, working_mem_new, training=self.training)  # [B, M]
        
        # 6. Apply mechanisms
        slots_new = slots.clone()
        for m_idx, mechanism in enumerate(self.mechanisms):
            gate = gates[:, m_idx:m_idx+1].unsqueeze(-1)  # [B, 1, 1]
            
            # Forward through mechanism
            slots_updated = mechanism(
                slots_new,
                adjacency=adjacency,
                working_mem=working_mem_new,
                memory_readout=memory_readout
            )
            
            # Weighted combination
            slots_new = slots_new + gate * (slots_updated - slots_new)
        
        # 7. Write to memory (side effect)
        self.external_memory.write(working_mem_new)
        
        # 8. Decode
        output = self.decoder(slots_new.mean(dim=1))  # [B, D_out]
        
        return {
            'output': output,
            'slots': slots,
            'slots_new': slots_new,
            'adjacency': adjacency,
            'gates': gates,
            'working_mem': working_mem_new,
            'slot_attn': slot_attn
        }
