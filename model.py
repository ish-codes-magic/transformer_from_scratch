import torch
import torch.nn as nn
import math

from embeddings import InputEmbeddings
from encoder import Encoder
from decoder import Decoder
from pos_enc import PositionalEncoding
from projection import ProjectionLayer
from mha import MultiHeadAttention
from feed_forward import FeedForwardBlock
from layer_norm import LayerNormalisation
from encoder import EncoderBlock
from decoder import DecoderBlock

#to define the various methods that the transformer model will be performing i.e. encoding, decoding and then the final projection layer
class TransformerBlock(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, tgt_embeddings: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_pos(self.src_embeddings(src))     #convert the encoder input from tokeniser to embedding vector and add positional encoding
        return self.encoder(src, src_mask)               #pass it to the encoder class
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_pos(self.tgt_embeddings(tgt))                    #convert the decoder input from tokeniser to embedding vector and add positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    #pass it to the encoder class
    
    def project(self, x):
        return self.projection_layer(x)
    
#function to build the transformer model
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> TransformerBlock:
    
    #defining the embedding classes
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeddings = InputEmbeddings(d_model, tgt_vocab_size)
    
    #defining the pos encoding classes
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    #having multiple encoder blocks so that the inpput can be passed through multiple encoder blocks iteratively N times
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)   #one encoder block consists of self attention and feed forward block
        encoder_blocks.append(encoder_block)
        
    #having multiple decoder blocks so that the inpput can be passed through multiple decoder blocks iteratively N times
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)     #one decoder block consists of self attention, cross attention and feed forward block
        decoder_blocks.append(decoder_block)
        
    encoder = Encoder(nn.ModuleList(encoder_blocks))             #final encoder class
    decoder = Decoder(nn.ModuleList(decoder_blocks))             #final decoder class
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)   #final projection layer
    
    transformer = TransformerBlock(encoder, decoder, src_embeddings, tgt_embeddings, src_pos, tgt_pos, projection_layer)   #final transformer class
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer