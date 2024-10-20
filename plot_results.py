import json
import matplotlib.pyplot as plt

# Load training losses and validation accuracies from the JSON file
with open('results.json', 'r') as f:
    results = json.load(f)

train_losses = results['train_losses']
val_accuracies = results['val_accuracies']

# Prepare x-axis as epoch numbers
epochs = range(1, len(train_losses) + 1)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, 'b', label='Training Loss')
plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
plt.title('Training Loss and Validation Accuracy (FFNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid()
plt.show()
