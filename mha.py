import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h
        #d_model = d_k * h
        self.w_q = nn.Linear(d_model, d_model)  #Wq: we need to convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        self.w_k = nn.Linear(d_model, d_model)  #Wk: we need to convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        self.w_v = nn.Linear(d_model, d_model)  #Wv: we need to convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        
        self.w_o = nn.Linear(d_model, d_model)  #Wo: we need to convert the output tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        
    @staticmethod
    def attention(q, k, v, d_k, mask, dropout):
        
        attention_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)  #calculate the attention scores for each head (matrix multiplication of q and k transpose)
        #so basically it says that for the positions that are 0 in the mask, replace the attention scores with -1e9
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  #mask the attention scores for each head
            
        attention_scores = torch.softmax(attention_scores, dim=-1)  #apply softmax to the attention scores for each head
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return torch.matmul(attention_scores, v), attention_scores
    
    def forward(self, q, k, v, mask):
        
        #convert Longtensor to float tensor
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        
        query = self.w_q(q)     #convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        key = self.w_k(k)       #convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        value = self.w_v(v)     #convert the input tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        
        #split the query, key and value into h heads -> (batch, seq_len, d_k*h) to (batch, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)
        
        #transpose the query, key and value to (batch, h, seq_len, d_k)
        #we transpose them because we want to perform matrix multiplication for each head for a particular batch, with the (seq_len,d_k) matrix for the attention calculation for each head, essentially we want to perform the attention calculation for each head in parallel
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        
        #calculate the attention scores for each head
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.d_k, mask, self.dropout)
        
        #now transpose back the transposed tensor to the original shape -> (batch, h, seq_len, d_k) to (batch, seq_len, h, d_k)
        x = x.transpose(1,2)
        
        #concatenate the h heads -> (batch, seq_len, h, d_k) to (batch, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.d_model)
        
        output = self.w_o(x)  #convert the output tensor to a tensor of shape (batch, seq_len, d_model) to (batch, seq_len, d_k*h)
        
        return output
        