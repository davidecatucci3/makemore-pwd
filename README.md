# makemore-pwd
makemore is ANN autoregressive model based on this paper: https://dl.acm.org/doi/pdf/10.5555/944919.944966, this project is inspired by the Andrej Karapthy makemore project series,
the aim of this model is to create memorable passwords (passwords that are not randomized but that they inside names, words, punctuaion and numbers that have a meaning)

## Model architecture
I will briefly explain you how the model works and its architecture, if something is not clear or you want to knwo more you can read the full paper visiting the link above.

This is the nework and this picture explains everything
<img width="526" alt="Screenshot 2025-02-19 at 16 30 19" src="https://github.com/user-attachments/assets/be1a46a7-2294-4b34-b495-0f91460993f6" />

The network is divided into three layers (input, hidden and output), it takes 3 characters in input and predict the fourth one

### Input layer
The input layer takes as input three characters (trigram), C is the embedding table, each of this three characters is converted in a row tensor of dimension emb_size,
so now we have three row tensor that are flattened so transormed from a shape of (3, emd_size) to (3 * emb_size), this row tensor is called X and feed it into the network

### Hidden layer
The hidden layer takes as input the (3 * emb_size) input and perform the operation tanh(XW1 + b1), where W1 and b1 are the network parameters, this gives as output a tensor of dimension (fan_out), this tensor is called a, fan_out is equal to the number of neurons in the hidden layer

### Output layer
The output layer takes the tensor of dimenison (fan_out) and perform the operaton softmax(aW2 + b2), so the output is (vocab_size) that we call probs where each element is the probability that the character will be the next of the trigram

## Files

## How to use it

## Train data
