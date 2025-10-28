import torch
import torch.nn as nn
import torch.nn.functional as F

class ExternalMemory(nn.Module):
    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        value_dim: int,
        controller_dim: int
    ):
        super().__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Controllers for read
        self.read_key_net = nn.Linear(controller_dim, key_dim)
        self.read_sharp_net = nn.Linear(controller_dim, 1)
        
        # Controllers for write
        self.write_key_net = nn.Linear(controller_dim, key_dim)
        self.write_sharp_net = nn.Linear(controller_dim, 1)
        self.erase_net = nn.Linear(controller_dim, value_dim)
        self.add_net = nn.Linear(controller_dim, value_dim)
        
        # Initialize memory
        self.register_buffer('keys', torch.randn(memory_size, key_dim))
        self.register_buffer('values', torch.zeros(memory_size, value_dim))
        
    def cosine_similarity(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Normalized cosine similarity"""
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        return torch.matmul(q_norm, k_norm.T)
    
    def read(self, controller_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            controller_state: [B, D_controller]
        Returns:
            read_value: [B, D_value]
        """
        B = controller_state.size(0)
        
        # Generate read query
        read_key = self.read_key_net(controller_state)  # [B, D_key]
        read_sharp = F.softplus(self.read_sharp_net(controller_state)) + 1.0
        
        # Content-based addressing
        similarity = self.cosine_similarity(read_key, self.keys)  # [B, M]
        read_weights = F.softmax(read_sharp * similarity, dim=-1)
        
        # Read from memory
        read_value = torch.matmul(read_weights, self.values)  # [B, D_value]
        
        return read_value
    
    def write(self, controller_state: torch.Tensor):
        """
        In-place memory update
        Args:
            controller_state: [B, D_controller]
        """
        B = controller_state.size(0)
        
        # Generate write parameters
        write_key = self.write_key_net(controller_state)
        write_sharp = F.softplus(self.write_sharp_net(controller_state)) + 1.0
        erase_vector = torch.sigmoid(self.erase_net(controller_state))  # [B, D_value]
        add_vector = torch.tanh(self.add_net(controller_state))  # [B, D_value]
        
        # Content-based addressing
        similarity = self.cosine_similarity(write_key, self.keys)
        write_weights = F.softmax(write_sharp * similarity, dim=-1)  # [B, M]
        
        # Batch average for memory update
        write_weights = write_weights.mean(dim=0, keepdim=True)  # [1, M]
        erase_vector = erase_vector.mean(dim=0, keepdim=True)  # [1, D_value]
        add_vector = add_vector.mean(dim=0, keepdim=True)  # [1, D_value]
        
        # Erase then add
        erase_matrix = torch.matmul(write_weights.T, erase_vector)  # [M, D_value]
        add_matrix = torch.matmul(write_weights.T, add_vector)  # [M, D_value]
        
        self.values = self.values * (1 - erase_matrix) + add_matrix
        
        # Optional: update keys
        self.keys = F.normalize(
            self.keys + 0.01 * torch.matmul(write_weights.T, write_key.mean(0, keepdim=True)),
            dim=-1
        )
