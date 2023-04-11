# Translating-From-German-to-English-
In this project we used T5 model in order to translating sentences from German to English.

1. [General](#General)
    - [Background](#background)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
    - [Improvement Trails](#Improvement-Trails)
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

<img src="https://i.imgur.com/WJJmZuh.png" width = 50% height=30%>

The dims of the convolution are 3-32-64, padding is 1 and stride is 2 and every convolution layer also includes Leaky relu function in both encoder and decoder. 

The linear layers dims encoder: 
* The linear layer dims are 64 * 4 * 4 to hidden dim. 
* The linear discrete layers dims are hidden dim to discrete dim. 
* The linear continuous layers dims are hidden dim to continuous dim. 

The linear layers dims decoder: 
* The linear layers dims are latent dim to hidden dim and hidden dim to 64 * 4 * 4 .
* The linear layers also include leaky relu function. 
* We used reparameterization trick for continuous variables and Gumbel SoftMax for the discrete variables.  

## Improvement-Trails

In order to improve the blue score, we thought that because blue biggest weakness is that it can't understand that 2 words are synonyms. 
Therefore, getting the roots and modifiers exact words, can help us increase the blue score if we'll replace the predicted roots and modifiers with the real one. This way, we can ensure that the those will get a positive blue score and won’t be considered wrong while having a very similar meaning. 
So in order to implement it, we tried finding the best method to switch the given and predicted roots and modifiers. We were inspired by a lot of known concepts as we'll elaborate now.

 

First, Bert-score: that as known, refers to that exact same problem. Bert-score uses cosine similarity between the embedded tokens of the word in the sentences in order to calculate its score, based on the idea that similar words would get a bigger cosine similarity after pretrained embedding. 
We also read about other methods of similarity score other then cosine similarity for text evaluation and made sure it is the best method to use. We used those ideas to find the similarity between the given modifiers\roots and predicted modifiers\roots.
After finding how similar each pair of predicted and given word is, we faced the problem of how to pick the replacement strategy. As expected, we want to take the max similar for each one, but what about the cases where 2 words got the same argmax? inspired by Active learning's uncertainty method of smallest margin calculation, where we can give a certainty score by calculating how much the model is confident in its 
prediction we implemented the method and understood this solution is not 
good enough because each replacement may cause a new conflict over and 
over again. When, we understood we're facing a "Perfect Match Problem" 
where each given modifiers\roots could be replaced with one predicted 
modifiers\roots and via versa and we want the best combination. We 
explored algorithms that solves this known problem. After trying those 
ideas we found out that in our case, it is better to able replacement 2 
predicted words with the same given one because it increases the changes to get a positive blue score (match) on at list on of them, rather of taking 
the risk of have a mistake in both of them and have zero matches at the n-gram comprehension in the blue calculation. 
In addition, we tried giving the roots and modifiers to the model, from the unlabeled version for the val and comp and using spacy for the training set which we did not have it's roots and modifiers.
 
## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Translating-From-German-to-English-.git
```
3. Train the mdeol with main_model.py file by:
```
    $ python main_model.py
```
4. Use the trained model to translate:
```
    $ python generate_comp_tagged.py
```
