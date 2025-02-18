import torch.nn as nn
import torch

from network_ds import Xtr, Ytr
from hyperparameters import hyperparams
from network import Network

lossi = []

def train():
  net = Network()

  # network data
  print(f'Number of parameters: {net.num_params}')

  # hyperparameters
  steps = hyperparams['steps']
  batch_size = hyperparams['batch size']

  # train
  for i in range(steps):
    # network input
    idxs = torch.randint(0, Xtr.shape[0], size=(batch_size,))
    x = net.C[Xtr[idxs]].view(-1, net.fan_in)

    # forward pass
    probs = net.forward(x)

    if i == 0:
      sprobs = probs

    # calc loss
    loss = -probs[torch.arange(batch_size), Ytr[idxs]].log().mean()

    # set grad to None
    for p in net.params.values():
      if isinstance(p, nn.ParameterList):
          for j in p:
              j.grad = None # more efficenly to put None then 0
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
      print(f'step {i} / {steps} | loss: {loss.item():.3f}')
    
    lossi.append(loss.log10().item())

  print(f'step {steps} / {steps} | loss: {loss.item():.3f}')

  # save network params
  net.save()

  return lossi, probs, sprobs