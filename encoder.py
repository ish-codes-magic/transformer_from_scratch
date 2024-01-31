import torch 
import torch.nn as nn
import math
from mha import MultiHeadAttention
from layer_norm import LayerNormalisation
from residual_connection import ResidualConnections
from feed_forward import FeedForwardBlock


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnections(dropout) for _ in range(2)])   #2 residual connections for encoder blocK(makes it easier to pass a function like the self attention block function in the next step)
        
    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))     #calculating self attention of encoder input with encoder mask
        x = self.residual_connection[1](x, self.feed_forward_block)                                #calculating feed forward block of encoder input
        #both these will be the output of the encoder block
        return x
        
        
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalisation()
        
    def forward(self, x, mask):
        #loop through all N encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)       #return the output of the encoder after passing it through layer normalisation