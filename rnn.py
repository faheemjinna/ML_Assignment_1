import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import string
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.rnn = nn.RNN(input_dim, hidden_dim, self.num_layers, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss()

    def compute_loss(self, predicted_output, true_label):
        return self.loss_function(predicted_output, true_label)

    def forward(self, inputs):
        hidden_state = torch.zeros(self.num_layers, inputs.size(1), self.hidden_dim)
        rnn_output, hidden_state = self.rnn(inputs, hidden_state)
        output = self.fc(hidden_state[-1])
        predicted_output = self.softmax(output)
        return predicted_output

def load_data(train_path, val_path):
    with open(train_path) as train_file:
        train_data = json.load(train_file)
    with open(val_path) as val_file:
        val_data = json.load(val_file)
    
    train = [(entry['text'].split(), int(entry['stars']) - 1) for entry in train_data]
    val = [(entry['text'].split(), int(entry['stars']) - 1) for entry in val_data]
    return train, val

def plot_learning_curve(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_rnn.png')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, required=True, help="Hidden dimension size")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--train_data', required=True, help="Path to training data")
    parser.add_argument('--val_data', required=True, help="Path to validation data")
    args = parser.parse_args()

    print("Loading data...")
    train_data, val_data = load_data(args.train_data, args.val_data)

    print("Initializing model...")
    input_dim = 50  # Assuming input word embeddings are of size 50
    model = RNN(input_dim=input_dim, hidden_dim=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    word_embeddings = pickle.load(open('./word_embedding.pkl', 'rb'))

    if unk not in word_embeddings:
        word_embeddings[unk] = np.zeros(input_dim)

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_val_accuracy = 0
    train_losses = []
    val_accuracies = []

    error_distribution = []
    train_predicted_labels = []
    train_actual_labels = []
    val_predicted_labels = []
    val_actual_labels = []

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        correct, total = 0, 0
        loss_total, loss_count = 0, 0

        print(f"Training epoch {epoch + 1}")
        for minibatch_index in tqdm(range(0, len(train_data), 32)):
            optimizer.zero_grad()
            batch_loss = 0

            for i in range(32):
                if minibatch_index + i >= len(train_data):
                    break
                input_words, label = train_data[minibatch_index + i]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                input_vectors = [word_embeddings.get(word.lower(), word_embeddings[unk]) for word in input_words]
                input_tensor = torch.tensor(input_vectors, dtype=torch.float32).unsqueeze(1)

                output = model(input_tensor)
                predicted_label = torch.argmax(output)

                batch_loss += model.compute_loss(output, torch.tensor([label]))
                correct += int(predicted_label == label)
                total += 1

                train_predicted_labels.append(predicted_label.item())
                train_actual_labels.append(label)

            batch_loss.backward()
            optimizer.step()

            loss_total += batch_loss.item()
            loss_count += 1

        train_loss = loss_total / loss_count
        train_losses.append(train_loss)
        train_accuracy = correct / total

        print(f"Training accuracy: {train_accuracy}")

        model.eval()
        correct, total = 0, 0
        print(f"Validating epoch {epoch + 1}")
        with torch.no_grad():
            for input_words, label in tqdm(val_data):
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                input_vectors = [word_embeddings.get(word.lower(), word_embeddings[unk]) for word in input_words]
                input_tensor = torch.tensor(input_vectors, dtype=torch.float32).unsqueeze(1)

                output = model(input_tensor)
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == label)
                total += 1

                val_predicted_labels.append(predicted_label.item())
                val_actual_labels.append(label)

                if predicted_label != label:
                    error_distribution.append({
                        "input": " ".join(input_words),
                        "predicted": predicted_label.item(),
                        "actual": label
                    })

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Validation accuracy: {val_accuracy}")

        if val_accuracy < last_val_accuracy and train_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Stopping early to avoid overfitting!")
        else:
            last_val_accuracy = val_accuracy
            last_train_accuracy = train_accuracy

        epoch += 1

    results = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies
    }
    with open('results_rnn.json', 'w') as f:
        json.dump(results, f)

    with open('error_distribution.json', 'w') as f:
        json.dump(error_distribution, f)

    with open('train_predictions.json', 'w') as f:
        json.dump({'predicted': train_predicted_labels, 'actual': train_actual_labels}, f)

    with open('val_predictions.json', 'w') as f:
        json.dump({'predicted': val_predicted_labels, 'actual': val_actual_labels}, f)

    epochs_range = list(range(1, epoch + 1))
    plot_learning_curve(epochs_range, train_losses, val_accuracies)

    print("Results and predictions saved. Learning curve plotted.")
