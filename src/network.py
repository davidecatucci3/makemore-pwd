import torch.nn as nn
import torch

from network_ds import vocab_size, Xtr
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
  def __init__(self):
    self.init_params()
  
  def init_params(self):
    self.fan_in = block_size * emb_dim
    layers = [self.fan_in] + [fan_out] * num_layers + [vocab_size]

    # embedding tensor
    self.C = torch.randn(size=(vocab_size, emb_dim), requires_grad=True)

    # weights (avoid vanishing gradient and high loss at the start using kaiming init)
    self.w = nn.ParameterList([
            nn.Parameter(torch.randn(_in, _out) * (5/3) / (self.fan_in ** 0.5))
            for _in, _out in zip(layers, layers[1:])
    ])
    
    # biases (avoid high loss at the start using kaiming init)
    self.b = nn.ParameterList([
            nn.Parameter(torch.randn(_out) * 0.1)
            for _out in layers[1:]
    ])

    # batch normalization
    self.bngain = torch.ones(size=(1, fan_out), requires_grad=True)
    self.bnbias = torch.zeros(size=(1, fan_out), requires_grad=True)

    self.bnmean_running = torch.ones((1, fan_out))
    self.bnstd_running = torch.zeros((1, fan_out))

    self.params = [self.C, self.w, self.b, self.bngain, self.bnbias]

    self.num_params = self.C.numel() + sum([w.numel() for w in self.w]) + sum([b.numel() for b in self.b]) + self.bngain.numel() + self.bnbias.numel()

  def forward(self, x):
    '''
    x: input tensor of dimension (B, F), B = batch_size and F = fan_in
    '''

    for w, b in zip(self.w, self.b):
        z = x @ w + b 

        # batch normalization
        if self.bngain.shape[-1] == z.shape[-1]: # skip batch normlizarion after the last layer
            bnmeani = z.mean(0, keepdim=True)
            bnstdi = z.std(0, keepdim=True)
    
            z = self.bngain * (z - bnmeani) / bnstdi + self.bnbias 

        a = torch.tanh(z)

        with torch.no_grad():
            self.sbnmean_running = 0.999 * self.bnmean_running + 0.001 * bnmeani
            self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * bnstdi

        x = a
    
    probs = nn.functional.softmax(x, dim=1) # (B, V), V = vocab_size

    return probs
  