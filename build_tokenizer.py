from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

if __name__ == '__main__':

    tokenizer_path = Path('./tokenizer/')
    tokenizer_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=['[UNK]'])
    tokenizer.train(['./dataset/shakespeare.txt'],trainer)
    tokenizer.save(str(tokenizer_path / 'shakespeare.json'))