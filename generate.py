import torch
from gpt import ShakespeareGPT
from train import Config
from tokenizers import Tokenizer

model_path = ''

tokenizer = Tokenizer.from_file('./tokenizer/shakespeare.json')


lm = ShakespeareGPT(Config)
state_dict = torch.load(model_path, map_location='cpu')
lm.load_state_dict(state_dict)

generated_texts = []
for length in [100,300,500,700,1000]:
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