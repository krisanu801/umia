from dataclasses import dataclass
from typing import Optional

@dataclass
class UMIAConfig:
    # Architecture
    num_slots: int = 32
    slot_dim: int = 256
    working_mem_dim: int = 512
    
    # Memory
    memory_size: int = 1024
    memory_key_dim: int = 128
    memory_val_dim: int = 256
    
    # Mechanisms
    num_mechanisms: int = 8
    routing_budget: int = 3
    
    # Input/Output
    input_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 512
    
    # Slot attention
    slot_attn_iters: int = 3
    slot_attn_heads: int = 4
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    
    # Loss weights
    lambda_do: float = 0.1
    lambda_graph_sparsity: float = 0.01
    lambda_graph_dag: float = 0.01  # Annealed
    lambda_energy: float = 0.001
    lambda_memory: float = 0.01
    
    # Annealing
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.1
    gumbel_temp_decay: float = 0.9999
    dag_anneal_rate: float = 1.01
    
    # Optimization
    use_amp: bool = True  # Mixed precision
    device: str = 'cuda'
    num_workers: int = 2
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 2000

config = UMIAConfig()
