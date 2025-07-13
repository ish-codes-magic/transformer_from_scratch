import torch
import torch.nn as nn
import math
from layer_norm import LayerNormalisation

class ResidualConnections(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalisation(features)
        
    def forward(self, x, sublayer):
        #apply layer normalisation to the input tensor x
        norm = self.layer_norm(x)
        #apply sublayer to the output tensor x
        layered = sublayer(norm)
        #apply dropout to the output tensor x
        out = self.dropout(layered)
        return x + out  #add the input tensor x to the output tensor x