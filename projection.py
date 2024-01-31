import torch
import torch.nn as nn
import math 

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = x.to(torch.float32)
        return torch.log_softmax(self.linear(x), dim=-1)