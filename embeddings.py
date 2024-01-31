import torch
import torch.nn as nn
import math

#define the embeddings class module
class InputEmbeddings(nn.Module): #InputEmbeddings inherits from nn.Module which is a class provided by PyTorch
    
    #define the constructor and pass the parameters d_model(dimension of the embeddings) and vocab_size
    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__()          #ensures all the methods and parameters are called from the parent class of nn.Module
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)     #define the embedding layer with vocab_size and d_model as parameters
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)     #return the embedding layer with the input x and multiply it with sqrt(d_model)[according to the og transformer paper]