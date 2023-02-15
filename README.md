# ShakespeareGPT

building & training GPT from scratch based off of [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial

#### dataset [tiny-shakespeare](./data/shakespeare.txt) : [original](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) with slight modifications.

### tutorialGPT (following the video)

- [basic_bigramLM.py](./tutorialGPT/basic_bigramLM.py) : built a basic bigram model with generate to get things rolling.
- [tutorial.ipynb](./tutorialGPT/tutorial.ipynb) : understood basic attention mechanism, using tril, masked_fill, softmax + notes on attention.
- [LMwithAttention.py](./tutorialGPT/LMwithAttention.py) : continued the model but now with single attention head, token embeddings, positional embeddings.
- [AttentionBlock.py](./tutorialGPT/AttentionBlock.py) : built a single attention head
- [LM_multihead_attention_ffwd.ipynb](./tutorialGPT/LM_multihead_attention_ffwd.ipynb) : continued the model to now have multiple attention heads concantenated, and a separate feed forward layer before lm_head.
- [tutorialGPT.ipynb](./tutorialGPT/tutorialGPT.ipynb) : created the transformer block, layering, residual connections, better loss evaluation, dropout, layernorm.

### Character Level GPT

> used a character level tokenizer. Trained two versions with different configurations to better understand the impact of the hyperparameters such as n_embeds, num_heads.

- [v1](./character_level_GPT/v1/):
  - [notebook](./character_level_GPT/v1/GPT_character_level_v1_trained.ipynb)
  - [saved model](./character_level_GPT/v1/shakespareGPT)
  - [results](./character_level_GPT/v1/generated.txt)

- [v2](./character_level_GPT/v2/):
  - [notebook](./character_level_GPT/v2/GPT_character_level_v2_trained.ipynb)
  - [saved model](./character_level_GPT/v2/shakespareGPT)
  - [results](./character_level_GPT/v2/generated.txt)

### ShakespeareGPT

> used a byte-pair encoding tokenizer.

- [gpt.py](./gpt.py) : the full GPT model
- [dataset.py](./dataset.py) : torch dataset
- [build_tokenizer.py](./build_tokenizer.py) : BPE tokenizer using `huggingface tokenizers` from scratch similar to GPT-2 saved at [tokenizer](./tokenizer/)
- [train.py](./train.py) : training script contains optimizer, config, loss function, train loop, validation loop, model saving
- [generate.py](./generate.py) : generate text by loading the model on CPU.

#### **Versions**

- ``` 
    V1
    n_embed = 384
    n_heads = 12
    head_size = 32
    n_layers = 4
    lr = 6e-4
    attn_dropout = 0.1
    block_dropout = 0.1

    Train Loss: 4.020419597625732
    Valid Loss: 6.213085174560547
  ```
  - [notebook](./saved/v1/shakespearegpt_v1.ipynb)
  - [saved model](./saved/v1/shakespeareGPT/)
  - [results](./saved/v1/generated.txt)

- ``` 
    V2
    n_embed = 384
    n_heads = 6
    head_size = 64
    n_layers = 3
    lr = 5e-4
    attn_dropout = 0.2
    block_dropout = 0.2

    Train Loss: 3.933095216751099 
    Valid Loss: 5.970513820648193
  ```
  - [notebook](./saved/v2/shakespearegpt_v2.ipynb)
  - [saved model](./saved/v2/shakespeareGPT/)
  - [results](./saved/v2/generated.txt)



---

as always, an incredible tutorial by Andrej!