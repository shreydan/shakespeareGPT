import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from tokenizers import Tokenizer
from train import Config

from dataset import ShakespeareDataset

from gpt import ShakespeareGPT

from tqdm.auto import tqdm
from pathlib import Path


tokenizer = Tokenizer.from_file('./tokenizer/shakespeare.json')

@dataclass
class Config:
    
    block_size = 256 # context-length
    batch_size = 64 # mini-batch size
    
    vocab_size = tokenizer.get_vocab_size()
    
    train_size = 0.8 
    
    n_embed = 384
    n_heads = 12
    head_size = n_embed // n_heads # computes to 384/12=32
    
    n_layers = 4
    
    train_iters = 5000 # no. of batches to train on
    val_iters = 500 # no. of batches to validate on every eval_intervals
    
    eval_interval = 500 # validate after every eval_interval iterations while training
    
    lr = 6e-4 # also used by the GPT 3 Small, quite a lot more stable than 1e-3
    
    attn_dropout = 0.1
    block_dropout = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    



lm = ShakespeareGPT(Config)
lm = lm.to(device=Config.device)

train_ds = ShakespeareDataset(Config)
val_ds = ShakespeareDataset(Config,is_test=True)


optim = torch.optim.AdamW(lm.parameters(), lr=Config.lr)

def loss_fn(logits, targets):
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits,targets)
    return loss




@torch.no_grad()
def valid_N_iters():
    val_step_losses = []
    for batch in tqdm(range(Config.val_iters)):
        inputs, targets = next(val_ds)
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        val_step_losses.append(loss.item())
        
        del inputs, targets, loss, logits
    
    val_loss = torch.tensor(val_step_losses).mean()
    print(f'val loss: {val_loss}')
    return val_loss


def train_N_iters():
    lm.train()
    train_step_losses = []
    val_losses = []
    for batch in tqdm(range(Config.train_iters)):
        optim.zero_grad()
        inputs, targets = next(train_ds)
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        loss.backward()
        optim.step()
        train_step_losses.append(loss.item())
        
        if batch%(Config.train_iters//10)==0 or batch==Config.train_iters-1:
            print(f"\n{'-'*50}\nbatch {batch} train step loss: {loss.item()}")
            print(f"train loss so far: {torch.tensor(train_step_losses).mean()}\n{'-'*50}\n")
            
        if batch%Config.eval_interval==0 or batch==Config.train_iters-1:
            lm.eval()
            val_loss = valid_N_iters()
            lm.train()
            val_losses.append(val_loss.item())
            
            del val_loss
            
        del inputs, targets, loss, logits
        
    return train_step_losses, val_losses


def save_lm():
    state_dict = lm.state_dict()
    save_path = Path('./').resolve() / 'shakespeareGPT'
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f'shakespeareGPT.pth'
    torch.save(state_dict, model_path)


def train_lm():
    train_step_losses,val_losses = train_N_iters()
    save_lm()
    return train_step_losses,val_losses


tsl,vl=train_lm()
tsl_mean = torch.tensor(tsl).mean()
print('Train Loss:',tsl_mean.item())
print('Validation Loss:',vl[-1])