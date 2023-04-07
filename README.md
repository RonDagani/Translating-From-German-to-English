# Translating-From-German-to-English-
In this project we used T5 model in order to translating sentences from German to English.

1. [General](#General)
    - [Background](#background)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
    - [Visualtions](#Visualtions)
5. [Installation](#Installation)

## General
The goal is to bulid deep learning based translation model from German to English.

### Background
The algorithm that was used to train the model on the training set is T5-Text-to-Text-Transfer-Transformer. Which was presented in [Exploring the 
Limits of Transfer Learning with a Unified Text-to-Text Transformer by 
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan 
Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu](https://arxiv.org/pdf/1910.10683.pdf) on 2020 

## Program Structure
* main_model.py - The main file where the model is defined and trained
* generate_comp_tagged.py - Uses the trained model to translate val and comp files.
* preprocess.py - Defnies a dataset used later out of file.
* project_evaluate.py - Defnies functions used to evalute the model

### Network-Structure
We found that using ada-factor optimization helped the performance of the model.
The parameters that were chosen were:
* Batch size = 2
* learning_rate = 0.0005
* num_train_epochs = 10 
* generation_max_length = 210
* gradient_accumulation_steps = 16

<img src="https://i.imgur.com/WJJmZuh.png" width = 50% height=50%>

The dims of the convolution are 3-32-64, padding is 1 and stride is 2 and every convolution layer also includes Leaky relu function in both encoder and decoder. 

The linear layers dims encoder: 
* The linear layer dims are 64 * 4 * 4 to hidden dim. 
* The linear discrete layers dims are hidden dim to discrete dim. 
* The linear continuous layers dims are hidden dim to continuous dim. 

The linear layers dims decoder: 
* The linear layers dims are latent dim to hidden dim and hidden dim to 64 * 4 * 4 .
* The linear layers also include leaky relu function. 
* We used reparameterization trick for continuous variables and Gumbel SoftMax for the discrete variables.  

### Visualtions
Having a look about the Visualtions of the latent spaces:
* Gender:

    <img src="https://i.imgur.com/JQWq0KF.png" width = 30% height=30%>

* Hair Style:

    <img src="https://i.imgur.com/2xnlPmn.png" width = 30% height=30%>

We can see that there is a speration but not a perfect one and in the hair which indcluds large number of categoires the model strugles with black hair for example.

## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Generative-Learning-With-Discrete-and-Continuous-VAE.git
```
3. Run the main.py file by:
```
    $ python main.py
```
