from tokenizers import Tokenizer
import torch
from pathlib import Path
from dataclasses import dataclass


class ShakespeareDataset:
    def __init__(self, Config, is_test=False) -> None:
        self.file_path = Path('./data/shakespeare.txt')
        self.tokenizer_path = Path('./tokenizer/shakespeare.json')
        with open(self.file_path,'r',encoding='utf-8') as f:
            self.data = f.read()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        
        self.full_data = torch.tensor(self.tokenizer.encode(self.data).ids, dtype=torch.long)

        self.is_test = is_test
        if self.is_test:
            self.data = self.full_data[int(0.9*len(self.full_data)):]
        else:
            self.data = self.full_data[:int(0.9*len(self.full_data))]

        self.block_size = Config.block_size
        self.batch_size = Config.batch_size

    def __len__(self) -> int:
        return len(self.data)

    def get_block_size(self) -> int:
        return self.block_size

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __next__(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x,y



    
if __name__ == '__main__':

    @dataclass
    class Config:
        batch_size = 4
        block_size = 8
        
    ds = ShakespeareDataset(Config)
    for i in range(4):
        x,y = next(ds)
        print(x.shape,y.shape)
    
