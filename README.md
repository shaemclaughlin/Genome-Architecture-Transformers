### DATASCI223 Final Project


## LADTransformer
This repository contains code for training and evaluating a LADTransformer model on DNA sequence data. The LADTransformer is a deep learning model that combines convolutional layers and transformer layers to predict lamina-associated domain (LAD) percentages from DNA sequences.  

### Requirements
To run this code, you need the following dependencies:

* Python 3.x
* Bio
* wandb
* numpy
* pandas
* scikit-learn
* torch
* matplotlib
* tqdm
* tensorflow
  
You can install the required packages using pip:

`pip install Bio wandb numpy pandas scikit-learn torch matplotlib tqdm tensorflow`  

### Setup
1. Clone the repository:

`git clone https://github.com/shaemclaughlin/Genome-Architecture-Transformers.git
cd lad-transformer`

2. Ensure that you have the encoded sequence data stored in a Google Drive directory named "encoded_sequences". The data should be in the form of .npz files, with each file containing the sequences and lad_percentages arrays.

3. Mount your Google Drive in the Colab notebook or the environment where you will be running the code. You can use the following code snippet to mount the drive:

`from google.colab import drive
drive.mount('/content/gdrive')`

4. Log in to Weights and Biases (wandb) for experiment tracking. You can use the wandb.login() function to log in.  

### Model Architecture
The LAD Transformer model consists of the following components:

* Convolutional layers: A series of 1D convolutional layers followed by ReLU activation and max pooling.
* Transformer layers: A transformer encoder with multi-head attention and positional encoding.
* Fully connected layers: Linear layers for final prediction.
  
The model takes DNA sequences as input and predicts the corresponding LAD percentages.

### Training
To train the LAD Transformer model:

1. Set the desired hyperparameters in the code, such as the number of convolutional layers, hidden sizes, number of transformer layers, learning rate, etc.
   
2. Run the code in a Colab notebook or a Python environment. The code will automatically load the data from the "encoded_sequences" directory, split it into training, validation, and test sets, and start the training process.

3. The training loop iterates over the training dataset, performing forward and backward passes, and updates the model parameters using the Adam optimizer. The training loss is logged at each step using wandb.

4. Every 10 outer steps, the code calculates the validation loss on a subset of the validation data and logs it using wandb.
The training progress and losses can be monitored using the wandb dashboard.

### Evaluation
After training, the code evaluates the trained model on the test set. It calculates the test loss and mean squared error (MSE) between the predicted and actual LAD percentages.

The test results are printed in the console.
