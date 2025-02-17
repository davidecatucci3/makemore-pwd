import torch

from hyperparameters import hyperparams
from network_ds import vocab_size

# hyperparameters
block_size = hyperparams['block size']

# create params
emb_dim = 20
fan_in, fan_out = block_size * emb_dim, 400

# embedding tensor
C = torch.randn((vocab_size, emb_dim))

# avoid vanishing gradient using kaiming init
W1 = torch.randn((fan_in, fan_out)) * (5/3) / (fan_in ** 0.5)
b1 = torch.randn((fan_out)) * 0.01

# init weights in some way that that the loss start from a low number, thislow  number should be around: -torch.tensor([1 / vocab_size]).log(), in this case is ~3.2958
W2 = torch.randn((fan_out, vocab_size)) * 0.01
b2 = torch.randn((vocab_size)) * 0.1

# batch normalization
bngain = torch.ones((1, fan_out))
bnbias = torch.zeros((1, fan_out))
bnmean_running = torch.ones((1, fan_out))
bnstd_running = torch.zeros((1, fan_out))

params = [C, W1, b1, W2, b2, bngain, bnbias]
num_params = 0

# count tot number of params and activate grad tracking
for p in params:
  p.requires_grad = True

  num_params += p.numel()