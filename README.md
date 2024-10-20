# ML Assignment 1
The main goal of this project is to build two types of neural networks: a feedforward neural network (FFNN) and a recurrent neural network (RNN) for analyzing the tone of Yelp reviews. The objective is to predict the sentiment score (ranging from 1 to 5) of the review text. We observe both approaches by evaluating their performance on the validation and training sets.

**FFNN**
``python ffnn.py --hidden_dim 32 --epochs 50 --train_data ./training.json --val_data ./validation.json``


**RNN**
``python rnn.py --hidden_dim 32 --epochs 10 --train_data training.json --val_data validation.json``

