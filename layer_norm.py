import torch
import torch.nn as nn
import math

class LayerNormalisation(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        #define the learnable parameters alpha and beta
        self.alpha = nn.Parameter(torch.ones(features)) #multipliable parameters
        self.beta = nn.Parameter(torch.zeros(features)) #addable parameters
        
    def forward(self, x):
        x = x.type(torch.float64)
        #calculate the mean and variance of the input tensor x
        mean = x.mean(-1, keepdim=True) #calculate the mean along the last dimension
        std = x.std(-1, keepdim=True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
    