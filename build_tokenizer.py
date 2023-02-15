from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
import tokenizers.pre_tokenizers as pre_tokenizers
import tokenizers.processors as processors
import tokenizers.decoders as decoders
from tokenizers.trainers import BpeTrainer

if __name__ == '__main__':

    tokenizer_path = Path('./tokenizer/')
    tokenizer_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(special_tokens=['<|endoftext|>'], min_frequency=2)

    tokenizer.train(['./data/shakespeare.txt'],trainer)
    tokenizer.save(str(tokenizer_path / 'shakespeare.json'))