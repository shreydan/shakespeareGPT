import torch
from gpt import ShakespeareGPT
from tokenizers import Tokenizer
from dataclasses import dataclass

model_path = ''

tokenizer = Tokenizer.from_file('./tokenizer/shakespeare.json')

@dataclass
class Config:
    
    block_size = 256 # context-length
    batch_size = 64 # mini-batch size
    
    vocab_size = tokenizer.get_vocab_size()
    
    n_embed = 384
    n_heads = 12
    head_size = n_embed // n_heads # computes to 384/12=32
    
    n_layers = 4
    
    attn_dropout = 0.1
    block_dropout = 0.1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


print(Config.device)

lm = ShakespeareGPT(Config)
state_dict = torch.load(model_path, map_location='cpu')
lm.load_state_dict(state_dict)

generated_texts = []
for length in [100]:
    generated = lm.generate(
    torch.zeros((1,1),dtype=torch.long,device='cpu'), # initial context 0
    total=length
)
    generated = tokenizer.decode(generated[0].cpu().numpy())
    text=f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
    generated_texts.append(text)

with open('generated.txt','w') as f:
    for text in generated_texts:
        f.write(text)