# makemore-pwd
makemore is ANN autoregressive model based on this paper: https://dl.acm.org/doi/pdf/10.5555/944919.944966, this project is inspired by the Andrej Karapthy makemore project series,
the aim of this model is to create memorable passwords (passwords that are not randomized but that they inside names, words, punctuaion and numbers that have a meaning)

## Model architecture
I will briefly explain you how the model works and its architecture, if something is not clear or you want to knwo more you can read the full paper visiting the link above.

<img width="526" alt="Screenshot 2025-02-19 at 16 30 19" src="https://github.com/user-attachments/assets/be1a46a7-2294-4b34-b495-0f91460993f6" />

This above is the network scheme and explain almost everyhing, it has three layers (input, hidden and output), it takes 3 characters in input and predict the fourth one

### Input layer
The input layer takes as input three characters (trigram), C is the embedding table of size (vocab_size, emb_size), each of this three characters is converted in a row tensor of dimension emb_size, so now we have three row tensor that are flattened so transormed from a shape of (3, emd_size) to (3 * emb_size), this row tensor is called X and feed it into the network

### Hidden layer
The hidden layer takes as input the (3 * emb_size) input and perform the operation tanh(X@W1 + b1), where @ stands for matrix multiplication and W1 and b1 are the network parameters, this gives as output a tensor of dimension (fan_out), this tensor is called a, fan_out is equal to the number of neurons in the hidden layer

### Output layer
The output layer takes the tensor of dimenison (fan_out) and perform the operaton softmax(a@W2 + b2), so the output is (vocab_size) that we call probs where each element represent the probaiblity to be the next character of the trigram, the element with the higher probability will be selected as next character

## Files
### doc
In the data folder there are the plots that i imported after the training of the network

### data
In the data folder there are three files: C.json, w.json and b.json, that are the parameters of the network saved after the trainig

### src
In the src folder there is the code:
  - hyperparamters.py: all the hyperparamters of the newtork are there
  - load_ds.py: this file load the dataset (passwords) and create the input dataset for the network
  - network.py: code of the network
  - train.py: code of the train process
  - inference.py: if you wanna tru to generate some passwords you can do it there

## How to use it

## Train data

> [!IMPORTANT]  
> My aim it was not to create the most accurate result form this network but just for a personal exercise, so i am showing the main data plot that i looked at during the
> training to see if the newtork was working well but this values can be make it much lower

### steps-train
https://github.com/davidecatucci3/makemore-pwd/blob/main/doc/step-loss.png?raw=true![image](https://github.com/user-attachments/assets/190e88e9-2f61-4347-8f14-ec5c274f78c1)

This plot shows on the x axis the steps and on the y axis the loss of both train and validation data, the plot is a logarithmic base 10, I trained the network for 200.000 steps with a lr of 0.1 that decays after half of the steps to 0.01 and 400 neurons for the hidden layer, the plot shows that the train went quit well there is not too much overfitting or underfitting, and after half of the steps there is a steep descent of the loss due to the lr decay, the final train loss and validation loss on the entire dataset if 2.48 and 2.51

### histogram probs
https://github.com/davidecatucci3/makemore-pwd/blob/main/doc/hist%20probs.png?raw=true![image](https://github.com/user-attachments/assets/13fd380b-5b3f-407c-ac78-66cecec3d8bd)

This graph is  something that i really like and that helps me a lot for understand if the network is workig well, on the left there is probs tensor at the first step and on the right the prob tensor after all the steps, the right one is better because there are a lot of probabilities that are 0 and few from 0.5 more so that means that the network is quite sure about is prediction, on the right ther are more zeros that good but tht max prob values is less then 0.2 so the network is very confued and random
