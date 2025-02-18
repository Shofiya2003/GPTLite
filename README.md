# GPTLite
A nano transformer model
I used [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and Andrej Karpathy's videos to build a small GPT-inspired model from scratch and trained it on the Shakespeare dataset.


# Validation Data Loss with Successive Additions to Transformer Architecture

| Added Component                | Validation Loss |
|--------------------------------|----------------|
| `batch_size=32`, `block_size=8`, `n_layer=4`, `n_head=4`, `N_emb=64` |
| Multi-head Attention + Computation | 2.2533         |
| + Blocks & Residual Connections  | 2.0960         |
| + Layer Normalization            | 2.0760         |
| `batch_size=64`, `block_size=1024`, `n_layer=12`, `n_head=12`, `N_emb=768` | 1.2335 |


# To Do
- [ ] Use `tiktoken` to encode the data.  
- [ ] Increase the hyperparameter values and train the model (when I am not GPU-scarce).
