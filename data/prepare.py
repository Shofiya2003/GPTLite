import torch

with open("input.txt", "r") as f:
  input = f.read()

# calculating the vocab size
input_chars = sorted(list(set(input)))
vocab_size = len(input_chars)

# poor man's encoder and decoder
stoi = {ch:i for i, ch in enumerate(input_chars)}
itos = {i:ch for i, ch in enumerate(input_chars)}
encode = lambda s : [stoi[ch] for ch in s] # takes str as input and returns list of integer
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(input), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]