import torch.nn as nn
import torch
import time

from hyperparameters import hyperparams
from network_ds import Xtr, Ytr
from network import Network

def train():
  net = Network()
  
  # keep track of latency
  t0 = time.time()
  k = 0

  # network data
  print(f'Layers structure: {net.layers}')
  print(f'Number of parameters: {net.num_params} \n')
  
  # hyperparameters
  steps = hyperparams['steps']
  batch_size = hyperparams['batch size']

  # train
  for i in range(steps):
    # network input
    idxs = torch.randint(0, Xtr.shape[0], size=(batch_size,))
    x_emb = net.C[Xtr[idxs]].view(-1, net.fan_in)

    # forward pass
    probs = net.forward(x_emb)

    # calc loss
    loss = -probs[torch.arange(batch_size), Ytr[idxs]].log().mean()

    # set grad to None
    for p in net.params.values():
      if isinstance(p, nn.ParameterList):
          for j in p:
              j.grad = None
      else:
        p.grad = None

    # backward pass
    loss.backward()

    # gradient steps
    lr = 0.1 if i < (steps // 2) else 0.01

    for p in net.params.values():
      if isinstance(p, nn.ParameterList):
        for j in p:
          j.data += lr * -j.grad
      else:
        p.data += lr * -p.grad

    # stats
    if i % (steps * 0.1) == 0:
      if i != 0 and (k + steps * 0.1) % i == 0:
        t1 = time.time()
        
        dt = t1 - t0

        t0 = t1

        k += int(steps * 0.1)
      else:
        dt = 0

      print(f'step {i} / {steps} | loss: {loss.item():.3f} | latency: {dt:.1f}s')

  print(f'step {steps} / {steps} | loss: {loss.item():.3f} | latency: {dt:.1f}s')

  # save network params
  net.save()

train()