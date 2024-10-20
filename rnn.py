import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):

        #Initialising the hidden state to zero, before processing the sequence.
        hidden_state = torch.zeros(self.numOfLayer, inputs.size(1), self.h)

        #The pre-defined RNN layer processes the input sequence. Also updates the hidden state.
        output, hidden_state = self.rnn(inputs, hidden_state)

        #The code gets the last hidden state and stores to the 'previous_hidden_state'.
        previous_hidden_state = hidden_state[-1]

        #To change the hidden state to 5 sentiment classes ouput.
        output_layer = self.W(previous_hidden_state)

        #Applying softmax to convert them into a probability distribution
        predicted_vector = self.softmax(output_layer)

        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    return tra, val

def plot_learning_curve(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', markersize=5)
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy', markersize=5)
    plt.title('Training Loss and Validation Accuracy (RNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve_rnn.png')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(input_dim=50, h=args.hidden_dim)  
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Ensure <UNK> has an embedding
    if unk not in word_embedding:
        word_embedding[unk] = np.zeros((50,))  # Assuming 50 is the embedding dimension

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    # To track errors
    error_distribution = []
    train_predicted_labels = []
    train_actual_labels = []
    val_predicted_labels = []
    val_actual_labels = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 32
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = 0
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding.get(i.lower(), word_embedding[unk]) for i in input_words]

                # Convert to tensor and ensure it's of type float32
                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                loss += example_loss

                # Store predicted and actual labels for training
                train_predicted_labels.append(predicted_label.item())
                train_actual_labels.append(gold_label)

            loss /= minibatch_size
            loss_total += loss.item()  # Use item() to extract the value
            loss_count += 1
            loss.backward()
            optimizer.step()
        
        train_loss = loss_total / loss_count
        train_losses.append(train_loss)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        training_accuracy = correct / total
        val_accuracies.append(training_accuracy)

        # Validation
        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding.get(i.lower(), word_embedding[unk]) for i in input_words]
            vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1)  # Ensure float32 here
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

            # Store predicted and actual labels for validation
            val_predicted_labels.append(predicted_label.item())
            val_actual_labels.append(gold_label)
            
            # Store errors
            if predicted_label != gold_label:
                error_distribution.append({
                    "input": " ".join(input_words),
                    "predicted": int(predicted_label),
                    "actual": gold_label
                })

        validation_accuracy = correct / total
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1

    # To get training losses and validation accuracies graph
    with open('results_rnn.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_accuracies': val_accuracies}, f)

    # To get error distribution graph
    with open('error_distribution.json', 'w') as f:
        json.dump(error_distribution, f)

    with open('train_predictions.json', 'w') as f:
        json.dump({'predicted': train_predicted_labels, 'actual': train_actual_labels}, f)
    
    with open('val_predictions.json', 'w') as f:
        json.dump({'predicted': val_predicted_labels, 'actual': val_actual_labels}, f)

    # Save learning curves
    epochs = list(range(1, epoch + 1))
    plot_learning_curve(epochs, train_losses, val_accuracies)

    print("Training and validation results saved to results_rnn.json, error distribution saved to error_distribution.json, and learning curve saved to learning_curve_rnn.png")
