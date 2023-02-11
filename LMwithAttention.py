import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from AttentionBlock import AttentionHead


torch.manual_seed(1357)


with open('./dataset/shakespeare.txt','r',encoding='utf-8') as f:
    data = f.read()


class CharacterLevelTokenizer:
    def __init__(self,data):
        self.data = data
        self.vocab = sorted(list(set(self.data)))
        self.VOCAB_SIZE = len(self.vocab)
        
        self.i_s = {i:s for i,s in enumerate(self.vocab)}
        self.s_i = {s:i for i,s in self.i_s.items()}
        
    def encode(self,s):
        return torch.tensor([self.s_i[c] for c in s],dtype=torch.long)

    def decode(self,s):
        return ''.join([self.i_s[i.item()] for i in s])



tokenizer = CharacterLevelTokenizer(data)


class ShakespeareDataset:
    def __init__(self,block_size:int, is_test=False) -> None:
        self.tokenizer = CharacterLevelTokenizer(data)
        self.is_test = is_test
        self.full_data = self.tokenizer.encode(self.tokenizer.data)
        if self.is_test:
            self.data = self.full_data[int(0.9*len(self.full_data)):]
        else:
            self.data = self.full_data[:int(0.9*len(self.full_data))]
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data)

    def get_block_size(self) -> int:
        return self.block_size

    def get_vocab_size(self) -> int:
        return self.tokenizer.VOCAB_SIZE

    def __getitem__(self,idx):
        item = self.data[idx:idx+self.block_size+1]
        x = item[:-1]
        y = item[1:]
        return x,y


@dataclass
class Config:
    block_size = 8 # context-length
    batch_size = 32 # mini-batch size
    vocab_size = tokenizer.VOCAB_SIZE
    n_embed = 16
    head_size = 16

train_ds = ShakespeareDataset(Config.block_size)
val_ds = ShakespeareDataset(Config.block_size,is_test=True)

class BigramLM(nn.Module):
    def __init__(self,Config):
        super(BigramLM,self).__init__()
        
        self.n_embed = Config.n_embed # number of embedding dims
        self.block_size = Config.block_size
        
        self.token_embedding_table = nn.Embedding(Config.vocab_size,self.n_embed)
        
        self.pos_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        
        self.self_attention = AttentionHead(Config)
        
        self.lm_head = nn.Linear(self.n_embed,Config.vocab_size)
        
    def forward(self,idx,targets=None):
        
        B,T = idx.shape
        
        token_embs = self.token_embedding_table(idx) # (B,T,n_embed)
        pos_embs = self.pos_embedding_table(torch.arange(T)) # (T,n_embed)
        
        """
        token_embs: B,T,n_embed
        pos_embs:  ,T,n_embed
               +: B,T,n_embed (broadcasted)
               
        so at this point, x knows the token affinities and importance of position
        """
        x = token_embs + pos_embs # (B,T,n_embed)
        x = self.self_attention(x) # (B,T,head_size)
        
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            # torch cross entropy expects B,C,T instead of B,T,C
            # and for targets, we need B*T instead of B,T
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
            
        return logits,loss

        
    def generate(self,idx,total):
        # idx (B,T) in current context
        for _ in range(total):
            idx_condition = idx[:,-self.block_size:]
            logits,loss = self(idx_condition)
            # since the last element is the next character, we pluck out -1 from T
            logits = logits[:,-1,:] # (B*T,C) -> (B,C)
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx,idx_next],dim=1) # (B, T+=1)
            
        return idx



bglm = BigramLM(Config)


optim = torch.optim.AdamW(bglm.parameters(),lr=1e-3)
bglm_dl = torch.utils.data.DataLoader(train_ds,shuffle=False,batch_size=Config.batch_size)

it = iter(bglm_dl)
for steps in range(10_000):
    inputs,targets = next(it)
    logits,loss=bglm(inputs,targets)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if steps%1000==0:
        print(f'step: {steps} loss: {loss.item()}')

generated = bglm.generate(
    torch.zeros((1,1),dtype=torch.long), # initial context 0
    total=500
)
generated = tokenizer.decode(generated[0])
print('generated (500 tokens) >>>\n',generated)

