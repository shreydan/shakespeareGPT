from tokenizers import Tokenizer



if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('./tokenizer/shakespeare.json')
    print(tokenizer.get_vocab_size())
    ex = tokenizer.encode("""MENENIUS:
Be gone;
Put not your worthy rage into your tongue;
One time will owe another.""")
    print(ex.ids)
    print(ex.tokens)
    print(len(ex.ids))
    print(tokenizer.decode(ex.ids))