import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple

class CausalStoryDataset(Dataset):
    """Generate synthetic causal reasoning tasks"""
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int,
        max_entities: int = 8,
        max_actions: int = 5,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.max_entities = max_entities
        self.max_actions = max_actions
        
        np.random.seed(seed)
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict]:
        samples = []
        
        for _ in range(self.num_samples):
            # Generate a simple causal chain
            num_entities = np.random.randint(3, self.max_entities + 1)
            num_actions = np.random.randint(2, self.max_actions + 1)
            
            # State: value for each entity
            state = np.random.randint(0, 10, size=num_entities)
            
            # Actions: (action_type, entity_idx, value)
            actions = []
            causal_graph = np.zeros((num_entities, num_entities))
            
            for _ in range(num_actions):
                action_type = np.random.choice(['add', 'sub', 'transfer'])
                
                if action_type in ['add', 'sub']:
                    entity = np.random.randint(0, num_entities)
                    value = np.random.randint(1, 5)
                    actions.append((action_type, entity, value))
                    
                    # Update state
                    if action_type == 'add':
                        state[entity] += value
                    else:
                        state[entity] = max(0, state[entity] - value)
                
                elif action_type == 'transfer':
                    if num_entities < 2:
                        continue
                    from_idx, to_idx = np.random.choice(num_entities, 2, replace=False)
                    value = np.random.randint(1, min(state[from_idx] + 1, 5))
                    actions.append((action_type, from_idx, to_idx, value))
                    
                    # Update state
                    state[from_idx] = max(0, state[from_idx] - value)
                    state[to_idx] += value
                    
                    # Causal edge
                    causal_graph[to_idx, from_idx] = 1
            
            # Encode as sequence
            sequence = self._encode_story(num_entities, actions)
            
            samples.append({
                'sequence': sequence,
                'final_state': state.copy(),
                'causal_graph': causal_graph,
                'num_entities': num_entities
            })
        
        return samples
    
    def _encode_story(self, num_entities: int, actions: List) -> np.ndarray:
        """Encode story as token sequence"""
        sequence = []
        
        # Start token
        sequence.append(1)
        
        # Num entities
        sequence.append(num_entities + 10)
        
        # Actions
        for action in actions:
            if len(action) == 3:  # add/sub
                action_type, entity, value = action
                sequence.extend([
                    20 if action_type == 'add' else 21,
                    entity + 30,
                    value + 40
                ])
            else:  # transfer
                _, from_idx, to_idx, value = action
                sequence.extend([
                    22,
                    from_idx + 30,
                    to_idx + 30,
                    value + 40
                ])
        
        # End token
        sequence.append(2)
        
        # Pad to seq_length
        sequence = sequence[:self.seq_length]
        sequence = sequence + [0] * (self.seq_length - len(sequence))
        
        return np.array(sequence, dtype=np.int64)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        sequence = torch.tensor(sample['sequence'], dtype=torch.long)
        final_state = torch.tensor(sample['final_state'], dtype=torch.float32)
        causal_graph = torch.tensor(sample['causal_graph'], dtype=torch.float32)
        
        # Embed sequence (simple embedding)
        embedded = torch.nn.functional.one_hot(sequence, num_classes=self.vocab_size).float()
        
        return {
            'input': embedded,
            'target': final_state,
            'causal_graph': causal_graph,
            'num_entities': sample['num_entities']
        }
