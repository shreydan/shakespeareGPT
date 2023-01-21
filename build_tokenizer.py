from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
import tokenizers.processors as processors
import tokenizers.decoders as decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC

if __name__ == '__main__':

    tokenizer_path = Path('./tokenizer/')
    tokenizer_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token='<|unknown|>'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.normalizer = NFKC()
    trainer = BpeTrainer(special_tokens=['<|unknown|>'], min_frequency=2)
    tokenizer.train(['./dataset/shakespeare.txt'],trainer)
    tokenizer.save(str(tokenizer_path / 'shakespeare.json'))