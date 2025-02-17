import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from datasets import load_dataset

# load dataset
ds = load_dataset('andrew98450/PasswordCaption')

# create vocab
data = ds['train']['password']
str_ds =  ''.join(data) # ['break1', 'maria12', ...] -> 'break1maria12 ...'

special_tokens = ['Σ', 'ε'] # Σ: start, ε: end, i am using greek letters because many of the special chars that i want to use are already in the vocab
vocab = sorted(set(str_ds)) + special_tokens
vocab_size = len(vocab)

# chars and int converter dicts
stoi = {vocab[i]: i for i in range(vocab_size)} # str to int
itos = {i: vocab[i] for i in range(vocab_size)} # int to str

# create model ds inputs
block_size = 3 # trigram, 3 input chars to predict the next one
X, Y = [], []

for text in data:
  text += 'ε'

  x = [stoi['Σ']] * block_size
  #print(text)
  for chr in text:
    y = stoi[chr]

    X.append(x)
    Y.append(y)
    #print(''.join([itos[i] for i in x]), '->', itos[y])
    x = x[1:] + [stoi[chr]]
  #print('-------')

# split ds
a = int(0.8 * len(X)) # 80% train 
b = int(0.9 * len(X)) # 10% val and test 

Xtr, Ytr = torch.tensor(X[:a]), torch.tensor(Y[:a])        # train ds
Xval, Yval = torch.tensor(X[a:b]), torch.tensor(Y[a:b])    # validation  ds
Xte, Yte = torch.tensor(X[b:]), torch.tensor(Y[b:])        # test ds

# create params
emb_dim = 20
fan_in, fan_out = block_size * emb_dim, 400

C = torch.randn((vocab_size, emb_dim))

W1 = torch.randn((fan_in, fan_out))
b1 = torch.randn((fan_out))

W2 = torch.randn((fan_out, vocab_size)) * 0.01 # * 0.1 for init W2 in some way that the loss does not go to inf
b2 = torch.randn((vocab_size)) * 0.01 # * 0.1 for init W2 in some way that the loss does not go to inf

params = [C, W1, b1, W2, b2]
num_params = 0

# count tot number of params and # activate grad tracking
for p in params:
  p.requires_grad = True

  num_params += p.numel()

# keep track of loss
lsteps = []
lloss = []

# train network
batch_size = 64
steps = 20000

for i in range(steps):
  # create network input
  idxs = torch.randint(0, Xtr.shape[0], size=(batch_size,))
  x_emb = C[Xtr[idxs]].view(-1, fan_in)

  # forward pass
  z = x_emb @ W1 + b1
  a = torch.tanh(z)

  logits = a @ W2 + b2
  probs = nn.functional.softmax(logits, dim=1)

  # calc loss
  loss = nn.functional.cross_entropy(logits, Ytr[idxs])

  lsteps.append(i)
  lloss.append(loss.item())

  # set grat to None
  for p in params:
    p.grad = None # more efficenly to put None then 0

  # backward pass
  loss.backward()

  # gradient steps
  lr = 0.01

  for p in params:
    p.data += lr * -p.grad

  # stats
  if i % (steps * 0.1) == 0:
    print(f'step {i} / {steps} | loss: {loss.item()}')
  
print(f'step {steps} / {steps} | loss: {loss.item()}')

# test on train set
x_emb = C[Xtr].view(-1, fan_in)

z = x_emb @ W1 + b1
a = torch.tanh(z)

logits = a @ W2 + b2

loss = nn.functional.cross_entropy(logits, Ytr)

print(f'train loss: {loss.item()}')

# test on val set
x_emb = C[Xval].view(-1, fan_in)

z = x_emb @ W1 + b1
a = torch.tanh(z)

logits = a @ W2 + b2

loss = nn.functional.cross_entropy(logits, Yval)

print(f'test loss: {loss.item()}')

# inference (generate passwords)
for i in range(20):
  pwd = ''

  x = [stoi['Σ']] * block_size
  idx = None

  with torch.no_grad():
    while idx != stoi['ε']:
      # create ntwork input
      x_emb = C[x].view(-1, fan_in)

      # forward pass
      z = x_emb @ W1 + b1
      a = torch.tanh(z)

      logits = a @ W2 + b2
      probs = nn.functional.softmax(logits, dim=1)

      # idx to chr
      idx = torch.multinomial(probs, num_samples=1).item()

      pwd += itos[idx]

      x = x[1:] + [idx]

  print(f'pwd {i + 1:02d}:', pwd)