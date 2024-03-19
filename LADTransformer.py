# Install packages
%pip install Bio
%pip install wandb

# Import required libraries and modules
import os
import pandas as pd
from Bio import SeqIO
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from tqdm import tqdm
import wandb

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import tensorflow as tf
tf.test.gpu_device_name()

# Mount Google Drive 
from google.colab import drive
drive.mount('/content/gdrive')

# Login to Weights and Biases (wandb) for experiment tracking
wandb.login()

# Define the LADTransformer model
class LADTransformer(nn.Module):
    def __init__(self, num_conv_layers, conv_hidden_size, num_transformer_layers, transformer_hidden_size, num_heads, dropout):
        super(LADTransformer, self).__init__()

        # Define convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_classes if i == 0 else conv_hidden_size, conv_hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            for i in range(num_conv_layers)
        ])

        # Define transformer layers
        self.pos_encoding = nn.Parameter(torch.zeros(2499, 1, transformer_hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(transformer_hidden_size, num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # Define fully connected layers
        self.fc1 = nn.Linear(conv_hidden_size, transformer_hidden_size)
        self.fc2 = nn.Linear(transformer_hidden_size, 1)

    def forward(self, x):
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
          
        # Reshape and linear projection
        x = x.permute(2, 0, 1)  # Reshape to (sequence_length, batch_size, conv_hidden_size)
        x = self.fc1(x)

        # Transformer layers
        x = x + self.pos_encoding
        x = self.transformer_encoder(x)

        # Fully connected layers
        x = x.mean(dim=0)  # Average over the sequence length
        x = self.fc2(x)
        return x.squeeze()

# Function to load data from a file with retry mechanism
def load_data_from_file(file_path, max_retries=3, retry_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            data = np.load(file_path)
            return data['sequences'], data['lad_percentages']
        except (ConnectionAbortedError, OSError) as e:
            retries += 1
            print(f"Error occurred while reading file {file_path}: {str(e)}. Retrying in {retry_delay} seconds... (Retry {retries}/{max_retries})")
            time.sleep(retry_delay)

    print(f"Max retries reached for file {file_path}. Skipping this file.")
    return None, None

# Function to load data one file at a time
def load_data_one_file_at_a_time(file_list):
    # Iterate over the files one by one
    for file in file_list:
        file_path = os.path.join("/content/gdrive/MyDrive/encoded_sequences", file)
        sequences, lad_percentages = load_data_from_file(file_path)

        if sequences is not None and lad_percentages is not None:
            yield sequences, lad_percentages

# Get the list of files in the encoded_sequences directory
file_list = [file for file in os.listdir("/content/gdrive/MyDrive/encoded_sequences") if file.endswith(".npz")]

# Split file list into training, validation and test sets
train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=100)
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=100)

# Function to load data from Google Drive and chunk it
def get_item_from_gdrive(idx: int, file_list, chunk_size):
    file_path = os.path.join("/content/gdrive/MyDrive/encoded_sequences", file_list[idx])
    data = np.load(file_path)
    sequences = data['sequences']
    lad_percentages = data['lad_percentages']

    num_chunks = (sequences.shape[0] + chunk_size - 1) // chunk_size
    chunked_sequences = []
    chunked_lad_percentages = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, sequences.shape[0])
        chunk_sequences = sequences[start_idx:end_idx]
        chunk_lad_percentages = lad_percentages[start_idx:end_idx]

        chunked_sequences.append(torch.from_numpy(chunk_sequences).float())
        chunked_lad_percentages.append(torch.from_numpy(chunk_lad_percentages).float())

    return chunked_sequences, chunked_lad_percentages

# Define custom dataset class for loading sequences
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, chunk_size):
        self.file_list = file_list
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
      return get_item_from_gdrive(idx, self.file_lit, self.chunk_size)

# Set the hyperparameters
num_conv_layers = 3
conv_hidden_size = 32
num_transformer_layers = 4
transformer_hidden_size = 128
num_heads = 4
dropout = 0.1
learning_rate = 0.001
num_classes = 5 # For A, G, T, C & N
chunk_size = 100 # Smaller batch size to reduce memory usage

# Create datasets
train_dataset = SequenceDataset(train_files, chunk_size)
val_dataset = SequenceDataset(val_files, chunk_size)
test_dataset = SequenceDataset(test_files, chunk_size)

# Initialize the model, optimizer, and loss function
model = LADTransformer(num_conv_layers,
                       conv_hidden_size,
                       num_transformer_layers,
                       transformer_hidden_size,
                       num_heads,
                       dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []
val_interval = 100 # Interval for calculating validation loss (in steps)

# Load the first item from the training dataset
seqs, labels = get_item_from_gdrive(0, train_files, chunk_size)

# Training and validation loop
model.train()
train_loss = 0.0
train_steps = 0
absolute_step_counter = 0

for outer_loop_step in tqdm(range(len(train_dataset))):
      chunked_sequences, chunked_labels = get_item_from_gdrive(outer_loop_step, train_dataset.file_list, chunk_size)
      for sequences, labels in zip(chunked_sequences, chunked_labels):
          sequences = sequences.squeeze(0)
          labels = labels.squeeze(0)

          sequences = torch.nn.functional.one_hot(sequences.to(torch.int64), num_classes=num_classes).float()
          sequences = sequences.permute(0, 2, 1).to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          outputs = model(sequences)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          train_loss += loss.item() * sequences.size(0)
          train_steps += sequences.size(0)

          # Log the train loss at each step
          wandb.log({'train_loss': loss.item()}, step=absolute_step_counter)
          absolute_step_counter += 1

      # Calculate validation loss every 10 outer steps steps
      if (outer_loop_step) % 10 == 0:
        validation_subset_proportion = 0.01 # Use 1% of validation set for each validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
          # Create a validation subset
          val_subset = random.sample(val_files, int(len(val_files) * validation_subset_proportion))
          val_dataset = SequenceDataset(val_subset, chunk_size)
          
          # Iterate over the validation subset
          for val_seq in range(len(val_dataset)):
            
            # Load the specified example
            chunked_sequences, chunked_labels = get_item_from_gdrive(val_seq, val_subset, chunk_size)
            for sequences, labels in zip(chunked_sequences, chunked_labels):
                sequences = sequences.squeeze(0)
                labels = labels.squeeze(0)

                sequences = torch.nn.functional.one_hot(sequences.to(torch.int64), num_classes=num_classes).float()
                sequences = sequences.permute(0, 2, 1).to(device)
                labels = labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * sequences.size(0)
                val_steps += sequences.size(0)

          val_loss /= val_steps
          val_losses.append(val_loss)
          
          # Only log the validation loss once per validation subset
        wandb.log({'val_loss': loss.item()}, step=absolute_step_counter)

        print(f"Step [{outer_loop_step}], Train Loss: {train_loss/train_steps:.4f}, Val Loss: {val_loss:.4f}")

train_loss /= train_steps
train_losses.append(train_loss)

# Evaluation on the test set
model.eval()
test_loss = 0.0
test_predictions = []
test_labels_list = []
test_steps = 0

with torch.no_grad():
    for test_seq in range(len(test_dataset)):
        chunked_sequences, chunked_labels = get_item_from_gdrive(test_seq, test_dataset.file_list, chunk_size)
        for sequences, labels in zip(chunked_sequences, chunked_labels):
            sequences = sequences.squeeze(0)
            labels = labels.squeeze(0)

            sequences = torch.nn.functional.one_hot(sequences.to(torch.int64), num_classes=num_classes).float()
            sequences = sequences.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * sequences.size(0)
            test_predictions.extend(outputs.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
            test_steps += sequences.size(0)

test_loss /= test_steps
test_mse = mean_squared_error(test_labels_list, test_predictions)
print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}")

# Close the wandb run
wandb.finish()
