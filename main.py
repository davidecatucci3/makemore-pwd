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
fan_in, fan_out = block_size * emb_dim, 300

C = torch.randn((vocab_size, emb_dim))

# avoid vanishing gradient using kaiming init
W1 = torch.randn((fan_in, fan_out)) * (5/3) / (fan_in ** 0.5) 
b1 = torch.randn((fan_out)) * 0.01 

# init weights in some way that that the loss start from a low number -> this number should be around: -torch.tensor([1 / vocab_size]).log(), in this case is ~3.2958
W2 = torch.randn((fan_out, vocab_size)) * 0.01
b2 = torch.randn((vocab_size)) * 0.01 

params = [C, W1, b1, W2, b2]
num_params = 0

# count tot number of params and activate grad tracking
for p in params:
  p.requires_grad = True

  num_params += p.numel()

# keep track of network data
lsteps = []
ltr_loss = []
lval_loss = []

# calc val loss
def calc_val_loss():
  # test on val set
  idxs2 = torch.randint(0, Xval.shape[0], size=(batch_size,))

  a = C[Xval[idxs2]].view(-1, fan_in)

  b = a @ W1 + b1
  c = torch.tanh(b)

  l = c @ W2 + b2

  return nn.functional.cross_entropy(l, Yval[idxs2]).log10().item()

# train network
batch_size = 32
steps = 200000

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
  loss = -probs[torch.arange(batch_size), Ytr[idxs]].log().mean()

  lsteps.append(i)
  
  ltr_loss.append(loss.log10().item())

  val_loss = calc_val_loss()
  lval_loss.append(val_loss)

  # set grat to None
  for p in params:
    p.grad = None # more efficenly to put None then 0

  # backward pass
  loss.backward()

  # gradient steps
  lr = 0.1 if i < 100000 else 0.01

  for p in params:
    p.data += lr * -p.grad

  # stats
  if i % (steps * 0.1) == 0:
    print(f'step {i} / {steps} | loss: {loss.item():.3f}')

print(f'step {steps} / {steps} | loss: {loss.item():.3f}')

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