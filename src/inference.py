import matplotlib.pyplot as plt
import torch

from hyperparameters import hyperparams
from network_ds import stoi, itos
from network import Network
from train import train

# hyperparams
block_size = hyperparams['block size']

# network
lossi, probs, sprobs = train()

'''plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()'''

# Create a figure with 1 row and 2 columns for side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # (1 row, 2 columns)

# Plot the first histogram on the first axis
axes[0].hist(sprobs[34].detach(), bins=50)
axes[0].set_title('Histogram of sprobs')

# Plot the second histogram on the second axis
axes[1].hist(probs[34].detach(), bins=50)
axes[1].set_title('Histogram of probs')

# Show the plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()

net = Network(load=True)

# inference (generate passwords)
for i in range(20):
  pwd = ''

  x = [stoi['Σ']] * block_size
  idx = None

  with torch.no_grad():
    while idx != stoi['ε']:
      # create ntwork input
      x_emb = net.C[x].view(-1, net.fan_in)

      # forward pass
      probs = net.forward(x_emb)

      # idx to chr
      idx = torch.multinomial(probs, num_samples=1).item()

      pwd += itos[idx]

      x = x[1:] + [idx]

  print(f'pwd {i + 1:02d}:', pwd)