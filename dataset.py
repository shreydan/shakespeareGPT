from tokenizers import Tokenizer
import torch
from pathlib import Path

class ShakespeareDataset:
    def __init__(self,block_size:int) -> None:
        self.file_path = Path('./data/shakespeare.txt')
        self.tokenizer_path = Path('./tokenizer/shakespeare.json')
        with open(self.file_path,'r',encoding='utf-8') as f:
            self.data = f.read()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.data = self.tokenizer.encode(self.data).ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data)

    def get_block_size(self) -> int:
        return self.block_size

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __getitem__(self,idx):
        item = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(item[:-1],dtype=torch.long)
        y = torch.tensor(item[1:],dtype=torch.long)
        return x,y

    
if __name__ == '__main__':
    dataset = ShakespeareDataset(block_size=8)
    print(len(dataset))
    print(dataset[0])
    print(dataset.get_block_size())
    print(dataset.get_vocab_size())
    dl = torch.utils.data.DataLoader(dataset,shuffle=False,batch_size=4)
    print(next(iter(dl)))
