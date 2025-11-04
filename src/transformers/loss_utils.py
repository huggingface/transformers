"""Memory-efficient loss computation."""
import torch
import torch.nn.functional as F

class TiledFusedLogitsLoss:
    def __init__(self, chunk_size=4096, ignore_index=-100):
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
    
    def __call__(self, hidden_states, lm_head_weight, labels, lm_head_bias=None):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        labels = labels.view(-1)
        total_loss = 0.0
        num_chunks = 0
        
        for i in range(0, hidden_states.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, hidden_states.shape[0])
            chunk_hidden = hidden_states[i:end_idx]
            chunk_labels = labels[i:end_idx]
            
            if (chunk_labels != self.ignore_index).sum() == 0:
                continue
            
            chunk_logits = F.linear(chunk_hidden, lm_head_weight, lm_head_bias)
            chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, ignore_index=self.ignore_index, reduction='sum')
            total_loss += chunk_loss
            num_chunks += (chunk_labels != self.ignore_index).sum()
        
        return total_loss / max(num_chunks, 1)
