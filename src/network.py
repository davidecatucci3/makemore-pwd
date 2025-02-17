import torch.nn as nn
import torch

from network_ds import vocab_size, Xtr, Ytr
from init_params import C, W1, b1, W2, b2, bngain, bnbias, params
from hyperparameters import hyperparams

# hyperparameters
steps = hyperparams['steps']
block_size = hyperparams['block size']
batch_size = hyperparams['batch size']
emb_dim = hyperparams['embedding dimension']
fan_out = hyperparams['dimension hidden layer']

# train network
for i in range(steps):
  # B: batch_size
  # V: vocab_size
  # F: fan_out

  # network input
  fan_in = vocab_size * block_size
  idxs = torch.randint(0, Xtr.shape[0], size=(batch_size,))
  x_emb = C[Xtr[idxs]].view(-1, fan_in) # (B, V)

  # forward pass
  z = x_emb @ W1 + b1 # (B, F)

  bnmeani = z.mean(0, keepdim=True)
  bnstdi = z.std(0, keepdim=True)
  z = bngain * (z - bnmeani) / bnstdi + bnbias # (B, F), batch normalization

  a = torch.tanh(z) # (B, F)

  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  
  logits = a @ W2 + b2 # (B, V)
  probs = nn.functional.softmax(logits, dim=1) # (B, V)

  # calculate loss
  loss = -probs[torch.arange(batch_size), Ytr[idxs]].log().mean()

  # set grad to None
  for p in params:
    p.grad = None # more efficenly to put None then 0

  # backward pass
  loss.backward()

  # gradient steps
  lr = 0.1 if i < 100000 else 0.01

  for p in params:
    p.data += lr * -p.grad

  # train stats
  if i % (steps * 0.1) == 0:
    print(f'step {i} / {steps} | loss: {loss.item():.3f}')

print(f'step {steps} / {steps} | loss: {loss.item():.3f}')