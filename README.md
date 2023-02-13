# ShakespeareGPT

building & training GPT from scratch based off of [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial

### dataset [tiny-shakespeare](dataset/shakespeare.txt) : [src](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) with slight modifications.

### tutorialGPT

- [basic_bigramLM.py](./tutorialGPT/basic_bigramLM.py) : built a basic bigram model with generate to get things rolling.
- [tutorial.ipynb](./tutorialGPT/tutorial.ipynb) : understood basic attention mechanism, using tril, masked_fill, softmax
- [LMwithAttention.py](./tutorialGPT/LMwithAttention.py) : continued the model but now with single attention head, token embeddings, positional embeddings.
- [AttentionBlock.py](./tutorialGPT/AttentionBlock.py) : built a single attention head
- [LM_multihead_attention_ffwd.ipynb](./tutorialGPT/LM_multihead_attention_ffwd.ipynb) : continued the model to now have multiple attention heads concantenated, and a separate feed forward layer before lm_head.
- [tutorialGPT.ipynb](./tutorialGPT/tutorialGPT.ipynb) : created the transformer block, layering, residual connections, better loss evaluation, dropout, layernorm.
