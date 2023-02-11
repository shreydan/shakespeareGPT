import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, Config):
        super().__init__()

        self.block_size = Config.block_size
        self.n_embed = Config.n_embed
        self.head_size = Config.head_size
        
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False)
        
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(self.block_size,self.block_size))
        )

    def forward(self, x):
        """
        k: (B,T,head_size)
        q: (B,T,head_size)

        last two dims of k have to be swapped
        k: (B,head_size,T)

        therefore

        q:   (B,T,head_size)
                    @
        k:   (B,head_size,T)
            =
        q@k: (B,T,T) # weights

        v:   (B,T,head_size)

        out: weights @ v
             (B,T,C)
        """

        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q@k.transpose(-2,-1) * (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))

        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        
        return out
        