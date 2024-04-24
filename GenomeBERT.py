#!/usr/bin/env python
# coding: utf-8

# In[40]:


# Import required libraries and modules
import os
import pandas as pd
from Bio import SeqIO
import gzip
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import wandb
import re
from pathlib import Path
import sentencepiece as spm
import linecache
import requests
import math


# In[41]:


# Set the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)
random.seed(10)
torch.cuda.manual_seed_all(10)

# Login to wandb for tracking
wandb.login()


# In[25]:


torch.cuda.empty_cache()


# In[ ]:


#!wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz


# In[4]:


# Define function to preprocess the raw fasta file
def preprocess_dna_sequence(sequence, chunk_size=512):
    # Remove unwanted characters and convert to uppercase
    sequence = re.sub(r"[^ACGT]", "", sequence.upper())
    chunks = [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
    return chunks


# In[5]:


# make this only chroms 1 to 22, X, Y
fasta_file = "GRCh38.primary_assembly.genome.fa.gz"
sequences = []
with gzip.open(fasta_file, "rt") as file:
    fasta_data = file.read()
fasta_records = fasta_data.split(">")[1:]
for record in tqdm(fasta_records):
    if record.strip():
        lines = record.strip().split("\n")
        sequence = "".join(lines[1:])
        preprocessed_chunks = preprocess_dna_sequence(sequence)
        sequences.extend(preprocessed_chunks)


# In[6]:


random.shuffle(sequences)


# In[7]:


# Write sequences to a new text file
sequence_file = "preprocessed_sequences.txt"
with open(sequence_file, "w") as file:
    for sequence in tqdm(sequences, desc="Writing sequences"):
        file.write(sequence + "\n") # New line for each sequence


# In[ ]:


# Write the first 100 sequences to a new text file
output_file = "samplesequences.txt"

with open(sequence_file, "r") as input_file, open(output_file, "w") as output:
    for i, line in enumerate(input_file):
        if i >= 100000:
            break
        output.write(line)


# In[8]:


# shuffle lines
with open(sequence_file, "r") as file:
    num_lines = sum(1 for _ in file)
print(num_lines)


# In[26]:


train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_lines = int(train_ratio * num_lines)
print(train_lines)
val_lines = int(val_ratio * num_lines)
print(val_lines)
test_lines = num_lines - train_lines - val_lines
print(test_lines)


# In[27]:


chars = sorted(list(set('ACGT')))
vocab_size = len(chars)

offset = 1

stoi = {ch: i + offset for i, ch in enumerate(chars)}
itos = {i + offset: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# In[28]:


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 100
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.01
vocab_size = 5


# In[29]:


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    if split == 'train':
        start_line = torch.randint(train_lines, (1,)).item()
        end_line = min(start_line + batch_size, train_lines)
    elif split == 'val':
        start_line = train_lines + torch.randint(val_lines, (1,)).item()
        end_line = min(start_line + batch_size, train_lines + val_lines)
    elif split == 'test':
        start_line = train_lines + val_lines + torch.randint(test_lines, (1,)).item()
        end_line = min(start_line + batch_size, num_lines)

    with open(sequence_file, "r") as file:
        # Move the file pointer to the starting line
        for _ in range(start_line):
            file.readline()

        # Read the required number of lines
        lines = [file.readline().strip() for _ in range(end_line - start_line)]

        # Encode the lines
        encoded_lines = [encode(line) for line in lines]

        # Pad sequences shorter than block_size
        padded_lines = [line + [0] * (block_size - len(line)) for line in encoded_lines]

        # Create full context data tensor y
        y = torch.tensor(padded_lines, dtype=torch.long)

        # Create input tensor x by masking a fraction of the tokens in y
        x = y.clone()
        
        num_masked = int(0.15 * y.numel()) # Mask 15% of tokens
        masked_indices = torch.randperm(y.numel())[:num_masked]
        x.view(-1)[masked_indices] = 0 # Set masked tokens to 0

        # Create a tensor to store positions of masked tokens
        masked_positions = torch.zeros_like(x, dtype=torch.bool)
        masked_positions.view(-1)[masked_indices] = True

        # Create a mask tensor for padding tokens
        padding_mask = torch.ones_like(x, dtype=torch.bool)
        padding_mask[y == 0] = False
        #print("x shape:", x.shape, "x dtype:", x.dtype)
        #print("y shape:", y.shape, "y dtype:", y.dtype)
        #print("masked_positions shape:", masked_positions.shape, "masked_positions dtype:", masked_positions.dtype)
        #print("padding_mask shape:", padding_mask.shape, "padding_mask dtype:", padding_mask.dtype)
        #print("x[:5]:", x[:5])
        #dummy_tensor = torch.ones_like(x)
        #dummy_tensor = dummy_tensor.to(device)
        x = x.to(device)
        y = y.to(device)
        masked_positions = masked_positions.to(device)
        padding_mask = padding_mask.to(device)
        return x, y, masked_positions, padding_mask


# In[ ]:


# Test the get_batch function
x, y, masked_positions, padding_mask = get_batch('train')
print("Train batch:")
print("Input tensor with 15% masked:")
print(x.shape)
print(x)
print("Target tensor with none masked:")
print(y.shape)
print(y)
print("Masked positions tensor:")
print(masked_positions.shape)
print(masked_positions)
print("Padding positions tensor:")
print(padding_mask.shape)
print(padding_mask)


# In[30]:


@torch.no_grad()
def estimate_loss():
    # initialize empty dictionary called out to store evaluation results
    out = {}
    # sets model to evaluation mode to ensure that any layers that behave different during training and eval use eval behavior
    model.eval()
    # loop iterates over two dataset splits: 'train' and 'val'
    for split in ['val']:
        # creates tensor losses of shape (eval_iters,) filled with zeros
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        # loop iterates eval_iters times
        for k in range(eval_iters):
            # retrieves batch of input data X and corresponding labels Y
            X, Y, masked_positions, padding_mask = get_batch(split)
            # passes input data X and labels Y to model's forward pass and returns logits and computed loss
            logits, _ = model(X, targets=Y, padding_mask=padding_mask)
            # stores scalar value of computed loss in losses tensor at index k
            #print(f"logits shape: {logits.shape}")
            #print(f"Y shape: {Y.shape}")
            
            # Compute loss only on masked tokens
            masked_logits = logits[masked_positions]
            #print(masked_logits.shape)
            masked_targets = Y[masked_positions]
            #print(masked_targets.shape)
            loss = F.cross_entropy(masked_logits, masked_targets)
            losses[k] = loss.item()
            
            # Calculate accuracy using targets
            targs = Y[masked_positions]
            preds = torch.argmax(masked_logits, dim=-1)
            acc = (preds == masked_targets).float().mean()
            accs[k] = acc.item()
            
        # computes mean of losses tensor and stores in out dictionary with corresponding split key
        out[split] = {
            'loss': losses.mean(),
            'acc': accs.mean()
        }
    model.train()
    return out


# In[31]:


# head class represents a single head of self-attention in the transformer model
# inherits from nn.Module to define a custom PyTorch module
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # initializes three linear layers (key, query, value) that project the input embeddings (n_embd) to the attention head space (head_size)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # registers buffer tril which is a lower triangular matrix of ones with shape (block_size, block_size)
        # matrix is used for masking future positions in the self-attention mechanism
        #self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # initializes dropout layer 
        self.dropout = nn.Dropout(dropout)

    # forward pass of the attention head
    def forward(self, x, padding_mask):
        #print("x")
        #print(x)
        B,T,C = x.shape # B = batch size, T = sequence length, C = embedding dimension
        k = self.key(x) # (B, T, C)
        #print("k")
        #print(k)
        q = self.query(x) # (B, T, C)
        #print("q")
        #print(q)
        # compute attention scores ("affinities")
        # performs matrix multiplication between query and transposed key and scaling by C**-0.5
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        #print("wei")
        #print(wei)
        # apply padding mask to attention scores
        #print("padding mask")
        #print(padding_mask.shape)
        #print(padding_mask)
        wei = wei.masked_fill(~padding_mask.unsqueeze(1), float('-inf')) # (B, T, T)
        # applies softmax to masked attention scores to obtain attention weights
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # applies dropout to weights
        wei = self.dropout(wei)
        # perform weighted aggregation of the values by matrix multiplication between attention weights (wei) and the value tensor v
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


# In[32]:


# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # initializes linear projection layer that maps the concatenated output of the attention heads back to the original embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        # initializes dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask):
        # takes input tensor x and applies each attention head in parallel using list comprehension
        # outputs of all heads are concatenated along the last dimension using torch.cat()
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)
        # applies projection layer and dropout to concatenated output
        out = self.dropout(self.proj(out))
        return out


# In[33]:


# simple linear layer followed by a non-linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # creates sequential neural network consisting of linear layer, relu activation function, linear layer, dropout
        self.net = nn.Sequential(
            # linear layer maps input from n_embd to 4 * n_embd dimensions
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # maps intermediate output back to n_embd dimensions
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


# In[34]:


# Transformer block: communication followed by computation
# Multi-head self-attention followed by feedforward neural network
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, padding_mask):
        # applies layer normalization to input tensor x
        # passes normalized input through multi-head self-attention module sa
        # adds output of self_attention module to the input
        x = x + self.sa(self.ln1(x), padding_mask)
        # applies layer normalization to output
        # passes normalized output through feedforward network
        # adds output of feedforward network to previous output (residual connection)
        x = x + self.ffwd(self.ln2(x))
        #returns output tensor 
        return x


# In[35]:


# simple model
class GenomicLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # creates embedding layer token_embedding_table to map token indices to token embeddings
        # embedding size is vocab_size and dimension n_embd
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # position_embedding_table maps position indices to position embeddings
        # embedding size is block size (maximum sequence length) and the embedding dimensionality is n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # creates sequential module (blocks) that contains n_layer instances of block class
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # final normalization layer
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # linear layer that maps output embeddings to logits over the vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None, padding_mask=None):
        # (idx, mask) == xs
        B, T = idx.shape # B = batch size, T = sequence length
        #print("B, T")
        #print(B, T)
        #print("idx")
        #print(idx)

        # idx and targets are both (B, T) tensor of integers
        # retrieves token embeddings for input idx resulting in tensor (B, T, C) where C is embedding dimensionality
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # generates position embeddings and a range tensor of length T resulting in tensor of shape (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # adds token embeddings and position embeddings element-wise to obtain input embeddings x
        x = tok_emb + pos_emb # (B, T, C)
        
        # pass padding mask to blocks
        for block in self.blocks:
            x = block(x, padding_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        
        else:
            # create a mask tensor for masked tokens
            masked_indices = idx == 0
            masked_logits = logits[masked_indices]
            masked_targets = targets[masked_indices]
            
            # replace this with nn.functional.binary_cross_entropy, but set weights as the mask
            loss = F.cross_entropy(masked_logits, masked_targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# In[36]:


model = GenomicLanguageModel()
m = model.to(device)
# print the number of parameters in the model
#print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# In[37]:


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# In[38]:


# Initialize wandb to log loss and accuracy
wandb.init(project='genome-BERT', name='model_test')


# In[ ]:


model.train()
x, y, masked_positions, padding_mask = get_batch('train')
print("Train batch:")
print("Input tensor with 15% masked:")
print(x.shape)
print(x)
print("Target tensor with none masked:")
print(y.shape)
print(y)
print("Masked positions tensor:")
print(masked_positions.shape)
print(masked_positions)
print("Padding positions tensor:")
print(padding_mask.shape)
print(padding_mask)


# In[ ]:


logits, loss = model(x, targets=y, padding_mask=padding_mask)


# In[ ]:


print(logits.shape)
print(logits)
print(loss)


# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[39]:


for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        metrics = estimate_loss()
        wandb.log({
            "val_loss": metrics['val']['loss'],
            "val_acc": metrics['val']['acc']
        }, step=iter)
        #print(f"step {iter}: train loss {metrics['train']['loss']:.4f}, train acc {metrics['train']['acc']:.4f}, val loss {metrics['val']['loss']:.4f}, val acc {metrics['val']['acc']:.4f}")
    # sample a batch of data
    
    model.train()
    
    xb, yb, masked_positions, padding_mask = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, targets=yb, padding_mask=padding_mask)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy using targets
    masked_logits = logits[masked_positions]
    masked_targets = yb[masked_positions]
    preds = torch.argmax(masked_logits, dim=-1)
    acc = (preds == masked_targets).float().mean()
    
    if iter % 10 == 0:
        wandb.log({"train_loss": loss.item(), "train_accuracy": acc.item()}, step=iter)

