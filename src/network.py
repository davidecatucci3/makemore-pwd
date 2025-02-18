import torch.nn as nn
import random
import torch
import json
import os

from network_ds import vocab_size
from hyperparameters import hyperparams

# hyperparameters
steps = hyperparams['steps']
block_size = hyperparams['block size']
batch_size = hyperparams['batch size']
emb_dim = hyperparams['embedding dimension']
num_layers = hyperparams['number hidden layers']
fan_out = hyperparams['dimension hidden layers']

# network
class Network:
  def __init__(self, load=False):
    if not load:
      self.init_params()

      self.train = True
    else:
      self.load()

      self.train = False
  
  def init_params(self):
    self.fan_in = block_size * emb_dim
    layers = [self.fan_in] + [fan_out] * num_layers + [vocab_size]

    # embedding tensor
    self.C = nn.Parameter(torch.randn(size=(vocab_size, emb_dim)))

    # weights (avoid vanishing gradient and high loss at the start using kaiming init)
    self.w = nn.ParameterList([
            nn.Parameter(torch.randn(_in, _out) * ((5/3) / (_in ** 0.5)))
            for _in, _out in zip(layers, layers[1:])
    ])
    
    # biases (avoid high loss at the start using kaiming init)
    self.b = nn.ParameterList([
            nn.Parameter(torch.randn(_out) * 0.1)
            for _out in layers[1:]
    ])

    # batch normalization
    self.bngain = nn.Parameter(torch.ones(size=(1, fan_out)))
    self.bnbias = nn.Parameter(torch.zeros(size=(1, fan_out)))

    self.bnmean_running = torch.ones((1, fan_out))
    self.bnstd_running = torch.zeros((1, fan_out))

    self.params = {'C': self.C, 'w': self.w, 'b': self.b} # 'bngain': self.bngain, 'bnbias': self.bnbias}

    self.num_params = self.C.numel() + sum(w.numel() for w in self.w) + sum(b.numel() for b in self.b) + self.bngain.numel() + self.bnbias.numel()

  def forward(self, x):
    '''
    x: input tensor of dimension (B, F), B = batch_size and F = fan_in
    '''

    for w, b in zip(self.w, self.b):
      z = x @ w + b 
       
      # batch normalization
      '''
      if self.train:
        if self.bngain.shape[-1] == z.shape[-1]: # skip batch normalization after the last layer
            bnmeani = z.mean(0, keepdim=True)
            bnstdi = z.std(0, keepdim=True)

            xhat = (z - bnmeani) / torch.sqrt(bnstdi + 1e-5)
            z = self.bngain * xhat + self.bnbias 
            
        with torch.no_grad():
          self.sbnmean_running = 0.999 * self.bnmean_running + 0.001 * bnmeani
          self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * bnstdi
      else:
        if self.bngain.shape[-1] == z.shape[-1]:
          z = self.bngain * (z - self.bnmean_running) / (self.bnstd_running + 1e-5) + self.bnbias 
      '''
      a = torch.tanh(z)
      x = a

    probs = nn.functional.softmax(z, dim=1) # (B, V), V = vocab_size

    return probs

  def save(self):
    path = 'data/'

    for n, p in self.params.items():
       if isinstance(p, nn.ParameterList):
            d = {}

            for i in p:
                if len(i.data.shape) == 2:
                    r, c = i.data.shape
                else:
                    r = i.data.shape[0]
                    c = 1

                k = f'{r}x{c}_{random.random()}'

                d[k] = i.tolist()
       
            with open(path + n + '.json', 'w') as f:
                json.dump(d, f, indent=4)
       else:
          with open(path + n + '.json', 'w') as f:
            json.dump(p.tolist(), f, indent=4)

  def load(self):
    self.fan_in = block_size * emb_dim

    path = 'data/'
    l_path = os.listdir(path)

    for filename in l_path:
      with open(path + filename) as f:
        jp = json.load(f)

      if isinstance(jp, dict):
        if filename == 'w.json': 
          self.w = nn.ParameterList([
             nn.Parameter(
                  torch.tensor(i, requires_grad=True)
                ) for i in jp.values()
          ])
        elif filename =='b.json':
          self.b = nn.ParameterList([
             nn.Parameter(
                  torch.tensor(i, requires_grad=True) 
                ) for i in jp.values()
          ])
      else:
        if filename == 'C.json':
          self.C = torch.tensor(jp, requires_grad=True)
        elif filename == 'bngain.json':
          self.bngain = torch.tensor(jp, requires_grad=True)
        elif filename == 'bnbias.json':
          self.bnbias = torch.tensor(jp, requires_grad=True)

    self.bnmean_running = torch.ones((1, fan_out))
    self.bnstd_running = torch.zeros((1, fan_out))

    self.params = {'C': self.C, 'w': self.w, 'b': self.b, 'bngain': self.bngain, 'bnbias': self.bnbias}

    self.num_params = self.C.numel() + sum([w.numel() for w in self.w]) + sum([b.numel() for b in self.b]) + self.bngain.numel() + self.bnbias.numel()



