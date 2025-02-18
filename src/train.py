import torch.nn as nn
import torch

from network_ds import Xtr, Ytr
from hyperparameters import hyperparams
from network import Network

def train():
  net = Network()
  
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
    lr = 0.1 if i < 100000 else 0.01

    with torch.no_grad():
      for p in net.params.values():
          if isinstance(p, nn.ParameterList):
            for j in p:
                  if j.grad is not None:
                      j -= lr * j.grad

                      j.grad.zero_()
          else:
              if p.grad is not None:
                  p -= lr * p.grad

                  p.grad.zero_()

    # stats
    if i % (steps * 0.1) == 0:
      print(f'step {i} / {steps} | loss: {loss.item():.3f}')

  print(f'step {steps} / {steps} | loss: {loss.item():.3f}')

  # save network params
  net.save()