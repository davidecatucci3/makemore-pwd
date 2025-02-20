# makemore-pwd
makemore is ANN autoregressive model based on this paper: https://dl.acm.org/doi/pdf/10.5555/944919.944966, this project is inspired by Andrej Karpathy makemore project series,
the aim of this model is to create memorable passwords (passwords that are not randomized like: sd34sfdf, !!jsdi33z but more like: Katia12, rock!, 1234, etc...)

## Model architecture
I will briefly explain you how the model works and its architecture, if something is not clear or you want to know more you can read the full paper visiting the link above

<img width="526" alt="Screenshot 2025-02-19 at 16 30 19" src="https://github.com/user-attachments/assets/be1a46a7-2294-4b34-b495-0f91460993f6" />

This above is the network scheme and explain almost everyhing, the network has three layers (input, hidden and output), it takes 3 characters in input and predict the fourth one

### Input layer
The input layer takes as input three characters (trigram), C is the embedding table, a tensor of shape (vocab_size, emb_size), each of the three characters is converted in a row tensor of shape (emb_size), so now we have three row tensor of shape (3, emb_size) that are flattened so transormed from a shape of (3, emd_size) to (3 * emb_size), this final tensor is called X and feed it into the network

### Hidden layer
The hidden layer takes as input the tensor of shape (3 * emb_size) and perform the operation tanh(X@W1 + b1), where @ stands for matrix multiplication and W1 and b1 are some of the network parameters, this gives as output a tensor of shape (fan_out), this tensor is called a, fan_out is equal to the number of neurons in the hidden layer

### Output layer
The output layer takes the tensor of shape (fan_out) and perform the operation softmax(a@W2 + b2), so the output tensor is of shape (vocab_size) that we call probs where each element represent the probaiblity to be the next character of the trigram, the element with the higher probability will be selected as next character

## Files
### doc
In the data folder there are the plots that i imported after training the network

### data
In the data folder there are three files: C.json, w.json and b.json, where inside there are the parameters of the network saved after the trainig

### src
In the src folder there is the code:
  - hyperparamters.py: dictionary with the hyperparameters of the network
  - load_ds.py: this file load the dataset (list of passwords) and create train, validation and test data to feed into the network
  - network.py: code of the network 
  - train.py: code of the train process
  - inference.py: if you want to try to generate some passwords you can do it with this file

## How to use it

## Train data

> [!IMPORTANT]  
> My aim it was not to create the most accurate results and the most efficeint network, i just did it for persona exercise, so I paid more attention at the code rather then the loss and the other data

### steps-train
![step-loss](https://github.com/user-attachments/assets/b0ff7d02-6af3-425a-8aa0-1baaab24c134)

This plot shows on the x axis the steps and on the y axis the loss of both train and validation data, the plot is a logarithmic base 10, I trained the network for 200.000 steps with a lr of 0.1 that decays after half of the steps to 0.01 and 400 neurons for the hidden layer, the plot shows that the training went quit well there is not too much overfitting or underfitting and after half of the steps there is a steep descent of the loss due to the lr decay, the final train loss and validation loss on the entire dataset if 2.48 and 2.51

### histogram probs
![hist probs](https://github.com/user-attachments/assets/7f3d6871-6f29-4d08-9731-e27a8fdbee4c)

This graph is  something that i really like analyze and that helped me a lot for understand if the network is workig well, on the left there is probs tensor at the first step and on the right the probs tensor after all the steps, the right one is better because there are a lot of probabilities that are 0 and the higher probs so the one that makes the decision are in the range 0.62 to 0.82 so that means that the network is quite sure about is prediction, on the left there are always a lot of zeros but the higher probs are in the range 0.23 - 0.26 so the network is completely random, its not sure about what character comes nex
