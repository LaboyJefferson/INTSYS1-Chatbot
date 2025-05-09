import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# You can adjust the num_epoch to handle overfitting

import nltk
nltk.download('punkt_tab')

# Create the directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

train_data_dir = "train_data/"
all_words = []
tags = []
xy = []

# Read all .txt files from train_data/
for filename in os.listdir(train_data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(train_data_dir, filename), "r", encoding='windows-1252') as file:
            lines = file.readlines()

        tag = ""
        patterns_section = False

        patterns = []
        for line in lines:
            line = line.strip()
            if line.startswith("intent:"):
                tag = line.split("intent:")[1].strip()
                tags.append(tag)
            elif line.startswith("patterns:"):
                patterns_section = True
                responses_section = False
            elif line.startswith("responses:"):
                patterns_section = False
            elif line.startswith("-") and patterns_section:
                pattern = line[1:].strip()
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"{len(xy)} patterns")
print(f"{len(tags)} tags:", tags)
print(f"{len(all_words)} unique stemmed words:", all_words)

# Training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
num_epochs = 200
batch_size = 8
learning_rate = 0.001
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

# DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, loss, optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
print(f'Final loss: {loss.item():.4f}')
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data/data.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')