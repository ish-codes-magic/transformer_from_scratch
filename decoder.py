import torch
import torch.nn as nn
import math
from mha import MultiHeadAttention
from layer_norm import LayerNormalisation
from residual_connection import ResidualConnections
from feed_forward import FeedForwardBlock

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnections(self_attention_block.d_model, dropout) for _ in range(3)])    #3 residual connections for decoder block(makes it easier to pass a function like the self attention block function in the next step)
        
    def forward(self, x, encoder_output, self_mask, cross_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, self_mask))    #calculating self attention of decoder input with decoder mask
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, cross_mask))  #calculating cross attention of decoder input with encoder output and cross mask
        x = self.residual_connection[2](x, self.feed_forward_block)                              #calculating feed forward block of decoder input
        #after all these steps, we get the output of one decoder block
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalisation(layers[0].self_attention_block.d_model)
        
    def forward(self, x, encoder_output, self_mask, cross_mask):
        #loop through all N decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return self.layer_norm(x)                               #return the output of the encoder after passing it through layer normalisation