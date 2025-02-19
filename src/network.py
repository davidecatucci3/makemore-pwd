import torch.nn as nn
import random
import torch
import json
import os

from hyperparameters import hyperparams
from network_ds import vocab_size

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
    '''
    load: if load is True that means that the parameters are loaded from the data folder (network has already been trained),load for default is False
    '''

    if not load:
      self.init_params()
    else:
      self.load()
  
  def init_params(self):
    self.fan_in = block_size * emb_dim
    
    self.layers = [self.fan_in] + [fan_out] * num_layers + [vocab_size]

    # embedding tensor
    self.C = nn.Parameter(torch.randn(size=(vocab_size, emb_dim)))

    # weights (avoid saturing gradient and high loss at the start using kaiming init)
    self.w = nn.ParameterList([
      nn.Parameter(torch.randn(_in, _out) * ((5/3) / (_in ** 0.5)))
      for _in, _out in zip(self.layers, self.layers[1:])
    ])
    
    # biases (avoid high loss at the start using kaiming init)
    self.b = nn.ParameterList([
      nn.Parameter(torch.randn(_out) * 0.01)
      for _out in self.layers[1:]
    ])

    self.params = {'C': self.C, 'w': self.w, 'b': self.b} 

    self.num_params = self.C.numel() + sum(w.numel() for w in self.w) + sum(b.numel() for b in self.b)

  def forward(self, x):
    '''
    x: input tensor of dimension (B, F), B = batch_size and F = fan_in
    '''

    for w, b in zip(self.w, self.b):
      z = x @ w + b 
  
      a = torch.tanh(z)
      
      x = a

    probs = nn.functional.softmax(z, dim=1) # (B, V), V = vocab_size

    return probs

  def save(self):
    params_path = 'data/'

    for p_name, p in self.params.items():
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
       
            with open(params_path + p_name + '.json', 'w') as f:
                json.dump(d, f, indent=4)
       else:
          with open(params_path + p_name + '.json', 'w') as f:
            json.dump(p.tolist(), f, indent=4)

  def load(self):
    self.fan_in = block_size * emb_dim

    params_path = 'data/'
    params = os.listdir(params_path)

    for p_name in params:
      with open(params_path + p_name) as f:
        jp = json.load(f)

      if isinstance(jp, dict):
        if p_name == 'w.json': 
          self.w = nn.ParameterList([
             nn.Parameter(
                  torch.tensor(i, requires_grad=True)
                ) for i in jp.values()
          ])
        elif p_name =='b.json':
          self.b = nn.ParameterList([
             nn.Parameter(
                  torch.tensor(i, requires_grad=True) 
                ) for i in jp.values()
          ])
      else:
        if p_name == 'C.json':
          self.C = torch.tensor(jp, requires_grad=True)
          
    self.params = {'C': self.C, 'w': self.w, 'b': self.b}

    self.num_params = self.C.numel() + sum([w.numel() for w in self.w]) + sum([b.numel() for b in self.b])