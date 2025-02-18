from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from data.prepare import vocab_size


@dataclass
class GPTConfig:
  batch_size:int = 64  # how many independent sequences will we process in parallel?
  block_size:int = 1024  # what is the maximum context length for predictions?
  n_embd = 768
  n_head = 12
  n_layers = 12
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dropout = 0.2
  vocab_size = vocab_size


class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=False)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.n_embd = config.n_embd
    self.n_heads = config.n_head
    self.attn_dropout = nn.Dropout(config.dropout)
    self.res_dropout = nn.Dropout(config.dropout)
    self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):

    B,T,C = x.size()

    # query, key, values
    q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, T, n_heads, head_size)

    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(C)) # (B, nh, T, T)
    att = att.masked_fill(self.tril[:,:,:T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    out = att @ v
    out = out.transpose(1,2).contiguous().view(B, T, C)
    out = self.res_dropout(self.c_proj(out))
    return out


class Head(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B,T,C = x.shape

    k = self.key(x)   # (B, T, 16)
    q = self.query(x) # (B, T, 16)
    wei =  q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    print(wei[0])
    return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config.head_size) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.fw = nn.Sequential(
        nn.Linear(config.n_embd, 4*config.n_embd),
        nn.ReLU(),
        nn.Linear(4*config.n_embd, config.n_embd),
        nn.Dropout(config.dropout)
    )

  def forward(self, x):
    return self.fw(x)


class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.head_size = config.n_embd//config.n_head
    # self.sa_heads = MultiHeadAttention(head_size, n_heads)
    self.attn = CausalSelfAttention(config)
    self.fw = FeedForward(config)
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.fw(self.ln2(x))
    return x


class LayerNorm1d:
  def __init__(self, dim, eps=-5):
    self.training = True
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    xmean = x.mean(1, keepdim=True)
    xvar = x.var(1, keepdim=True)
    xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class GPTLite(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
    self.layernorm = nn.LayerNorm(config.n_embd)
    self.ln_head = nn.Linear(config.n_embd, config.vocab_size)
    self.config = config

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embds = self.token_embedding_table(idx) # (B, T, C) (here C = 32)
    pos_embds = self.position_embedding(torch.arange(T, device=self.config.device)) # (T, C)
    x = token_embds + pos_embds
    x = self.blocks(x)
    x = self.layernorm(x)
    logits = self.ln_head(x)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(-1)
      loss = F.cross_entropy(logits, targets)
    return logits, loss


  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.config.block_size:]
      logits, loss  = self(self.config) # (B, T, C) (T=1)
      logits = logits[:, -1, :] # (B, C)
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
