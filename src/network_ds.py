import torch

from hyperparameters import hyperparams
from datasets import load_dataset

# load dataset
ds = load_dataset('andrew98450/PasswordCaption')

# create vocab
data = ds['train']['password']
str_ds =  ''.join(data)

special_tks = ['Σ', 'ε'] # Σ: start sequence, ε sequence: end, I am using greek letters because many of the special chars that i want to use are already in the vocab
vocab = sorted(set(str_ds)) + special_tks
vocab_size = len(vocab)

# chars and int converter dicts
stoi = {vocab[i]: i for i in range(vocab_size)} # str to int
itos = {i: vocab[i] for i in range(vocab_size)} # int to str

# create network input data
block_size = hyperparams['block size'] # trigram, 3 input chars to predict the next one
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

# split dataset
a = int(0.8 * len(X)) # 80% train 
b = int(0.9 * len(X)) # 10% validation and 10% test 

Xtr, Ytr = torch.tensor(X[:a]), torch.tensor(Y[:a])        # 80% train dataset
Xval, Yval = torch.tensor(X[a:b]), torch.tensor(Y[a:b])    # 10% validation dataset
Xte, Yte = torch.tensor(X[b:]), torch.tensor(Y[b:])        # 10% test dataset