import torch
import torch.nn as nn
import math


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float) -> None:
        super().__init__()
        #detailing how the paper defines the feed forward block, see section 3.3 in the paper in page 5
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = x.to(torch.float32)
        #first linear layer(converts batch,seq_len,d_model ->batch,seq_len,d_ff) -> relu -> dropout -> second linear layer(converts batch,seq_len,d_ff ->batch,seq_len,d_model)
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
        