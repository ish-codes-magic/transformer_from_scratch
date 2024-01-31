import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create an empty tensor of size (seq_len, d_model) to store the positional encodings
        pe = torch.zeros(seq_len, d_model)
        
        #defining positions in terms of tensor indices
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        #calculate the term inside the sine and cosine functions i.e. the div term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        #div_term_2 = 1/(10000**(torch.arange(0, d_model, 2).float() / d_model))       #this is the same as the above div_term, so why its not used?
        
        #to get the positional encodings, we need to calculate the sine of even indices of the div_term and cosine of odd indices of the div_term
        pe[:, 0::2] = torch.sin(pos * div_term)       #pe[:, 0::2] means all the rows and all the columns with even indices
        pe[:, 1::2] = torch.cos(pos * div_term)       #pe[:, 1::2] means all the rows and all the columns with odd indices
        
        pe = pe.unsqueeze(0)   #unsqueeze the positional encodings tensor to add a batch dimension to create a tensor of shape (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)      #register the positional encodings tensor as a buffer so that it is not updated as a model parameter
        
    def forward(self, x):
        
        #add the positional encodings to the input embeddings and return the result
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)      #x.size(1) is the length of the input sequence
        return self.dropout(x)      #apply dropout to the input embeddings + positional encodings and return the result