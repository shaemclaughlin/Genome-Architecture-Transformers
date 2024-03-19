# Install packages
%pip install Bio

# Import packages
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
import psutil
import humanize
import GPUtil as GPU

# Mount Google Drive 
from google.colab import drive
drive.mount('/content/gdrive')

# Download reference human genome
!wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz

# Unzip the reference human genome
!gunzip GRCh38.primary_assembly.genome.fa.gz

# Read in CSV files and clean the data
path = '/content/gdrive/MyDrive/lad_files'

files = os.listdir(path)

sample_names = []
dataframes = []

for file in files:
    # Create file path
    file_path = os.path.join(path, file)

    # Read the file into a dataframe
    df = pd.read_csv(file_path, header=None, usecols=[0,1,2], skiprows=1)

    # Rename the columns
    df.columns = ['Chromosome', 'Start', 'End']

    # Get rid of '\' in 'End' column
    df['End'] = df['End'].str.replace('\\','')

    # Change strings to integers
    df['Start'] = pd.to_numeric(df['Start'], errors = 'coerce').astype(int)
    df['End'] = pd.to_numeric(df['End'], errors = 'coerce').astype(int)

    # Add length column
    df['Length'] = df['End'] - df['Start']

    # Add LAD column
    df['LAD'] = 'LAD'

    # Remove file extensions to get a sample name
    sample = os.path.splitext(file)[0]
    df['Sample']= sample

    # Append the df to the list
    dataframes.append(df)

# Chromosome lengths from the human genome fasta
chrom_lengths = {
    'chr1': 248956422,
    'chr2': 242193529,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chr10': 133797422,
    'chr11': 135086622,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr19': 58617616,
    'chr20': 64444167,
    'chr21': 46709983,
    'chr22': 50818468,
    'chrX': 156040895,
    'chrY': 57227415
}

def generate_bins(chrom, chrom_length, bin_size):
    bins = []
    for start in range(0, chrom_length, bin_size):
        end = min(start + bin_size - 1, chrom_length)
        bins.append((chrom, start, end))
    return pd.DataFrame(bins, columns=['Chromosome', 'Start', 'End'])

bin_size = 20000  # 20kb bin size

# Generate bins for each chromosome
bin_dfs = []
for chrom, length in chrom_lengths.items():
    chrom_bins = generate_bins(chrom, length, bin_size)
    bin_dfs.append(chrom_bins)

bins_df = pd.concat(bin_dfs, ignore_index=True)

sample_name = 'astrocyte_49yo'  # Replace with the desired sample name

# Select a single sample for testing
sample_names = [df['Sample'].unique()[0] for df in dataframes]

# Find index of sample name in sample_names list
sample_index = sample_names.index(sample_name)

# Select the sample_df from the dataframes list using the found index
sample_df = dataframes[sample_index]

# Calculate LAD percentage for each bin
def calculate_lad_percentage(row):
    bin_chrom, bin_start, bin_end = row['Chromosome'], row['Start'], row['End']
    lad_percentage = 0.0

    for _,lad in sample_df[sample_df['Chromosome'] == bin_chrom].iterrows():
        lad_start, lad_end = lad['Start'], lad['End']

        overlap_start = max(bin_start, lad_start)
        overlap_end = min(bin_end, lad_end)

        if overlap_start < overlap_end: # there is an overlap
            overlap_length = overlap_end - overlap_start
            lad_percentage += overlap_length / bin_size

    # In case there are multiple LADs overlapped with the bin and the lad_percentage exceeds 1
    return min(lad_percentage, 1)

bins_df['LAD_Percentage'] = bins_df.apply(calculate_lad_percentage, axis=1)

print(f"Number of rows in the bins DataFrame for sample {sample_name}: {len(bins_df)}")
print(bins_df.head())
print(bins_df['LAD_Percentage'].describe())

astrocyte_49yo_bins = bins_df
path = "/content/gdrive/MyDrive/lads/astrocyte_49yo_bins.csv"
astrocyte_49yo_bins.to_csv(path, index=False)

new_path = "/content/gdrive/MyDrive/lads"

new_files = os.listdir(new_path)

all_dfs = []

for file in new_files:
  # Create file path
  file_path = os.path.join(new_path, file)

  # Read the file into a dataframe
  df = pd.read_csv(file_path)

  # Remove file extension to get the sample name
  sample = os.path.splitext(file)[0]

  df['Sample']=sample

  # Add sample column
  all_dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)

genome_path = '/content/GRCh38.primary_assembly.genome.fa'
genome_sequences = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

# Iterate over the dataframe rows
sequences_data = []
for _, row in combined_df.iterrows():
    chromosome = row['Chromosome']
    start = row['Start']
    end = row['End']
    lad_percentage = row['LAD_Percentage']
    sample = row['Sample']

    # Extract the corresponding sequence from the genome
    sequence = genome_sequences[chromosome].seq[start:end]

    # Store the extracted sequence and other information
    sequences_data.append({
        'Chromosome': chromosome,
        'Start': start,
        'End': end,
        'LAD_Percentage': lad_percentage,
        'Sample': sample,
        'Sequence': str(sequence)
    })

# Create a new dataframe with the extracted sequences
sequences_df = pd.DataFrame(sequences_data)

# Print the sample dataframe
print(sequences_df.head())

path = "/content/gdrive/MyDrive/sequences_df.csv"
sequences_df.to_csv(path, index=False)

path = "/content/gdrive/MyDrive/"

# Set the batch size
batch_size = 1000

# Create directory to store the encoded sequences
output_dir = "/content/gdrive/MyDrive/encoded_sequences"
os.makedirs(output_dir, exist_ok=True)

# Load dataframe
df = pd.read_csv("/content/gdrive/MyDrive/sequences_df.csv")

# Get the DNA sequences and LAD percentages
sequences = df['Sequence'].values
lad_percentages = df['LAD_Percentage'].values

# Determine the maximum sequence length
max_length = 19999

# Integer encoding the DNA sequences
def integer_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded_sequence = np.array([mapping.get(base, 4) for base in sequence])
    return encoded_sequence

# Process the data in batches
num_batches = len(df) // batch_size + 1

for batch_idx in range(num_batches):
  start_idx = batch_idx * batch_size
  end_idx = min((batch_idx + 1) * batch_size, len(df))

  batch_df = df.iloc[start_idx:end_idx]
  batch_sequences = batch_df['Sequence'].values
  batch_lad_percentages = batch_df['LAD_Percentage'].values

  # Pad the sequences to maximum length
  batch_padded_sequences = [seq.ljust(max_length, 'N') for seq in batch_sequences]

  # Integer encode the batch of sequences
  batch_integer_sequences = np.array([integer_encode(seq) for seq in batch_padded_sequences])

  # Save the encoded sequences and LAD percentages for the current batch
  output_file = os.path.join(output_dir, f"encoded_sequences_batch_{batch_idx}.npz")
  np.savez(output_file, sequences=batch_integer_sequences, lad_percentages=batch_lad_percentages)

  print(f"Processed batch {batch_idx + 1}/{num_batches}")

print("Data processing completed.")
