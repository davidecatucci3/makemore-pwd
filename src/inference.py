import torch

from hyperparameters import hyperparams
from network_ds import stoi, itos
from network import Network

# hyperparameters
block_size = hyperparams['block size']

# network
net = Network(load=True)

# inference (generate passwords)
num_pwd_generated = 20

for i in range(num_pwd_generated):
  pwd = ''
  
  x = [stoi['Σ']] * block_size
  
  idx = None

  with torch.no_grad():
    while idx != stoi['ε']:
      # create network input
      x_emb = net.C[x].view(-1, net.fan_in)

      # forward pass
      probs = net.forward(x_emb)

      # idx to chr
      idx = torch.multinomial(probs, num_samples=1).item()

      pwd += itos[idx]

      x = x[1:] + [idx]

  print(f'pwd {i + 1:02d}:', pwd)