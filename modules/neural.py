import os
import re

import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn

import contractions

#####################################################
################# Global Variables ##################
#####################################################

# Path variables
MODELS_PATH = os.path.join("models")
DATA_PATH = os.path.join("data")

EMB_MATRIX_PATH = os.path.join(MODELS_PATH, "emb_matrix.npy")
TOKENIZER_PATH = os.path.join(MODELS_PATH, "tokenizer.pkl")

# PyTorch variables
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#####################################################
#################### Functions ######################
#####################################################

def prepare_text(text):
    """
        Prepares text for input into deep learning network
    """
    if not os.path.exists(TOKENIZER_PATH):
        print("Unable to find saved Tokenizer!")
        return False

    tokenizer = joblib.load(TOKENIZER_PATH)
    
    X_seq = tokenizer.texts_to_sequences(text)
    print(len(X_seq))
    X_padded = pad_sequences(X_seq, maxlen=410, padding="post")
    X_tensor = torch.LongTensor(X_padded)

    return X_tensor


def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
      return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r"-", " ", text) # Replace dash(es) with single whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


#####################################################
################ Deep Learning Models ###############
#####################################################

# Load the embedding matrix used during the training of
# deep learning models to be used when loading them here
emb_matrix = np.load(EMB_MATRIX_PATH)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False # False = Freeze embeddings
        
        # Sequential Convolutional layers
        self.net = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, 5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout1d(0.5),
            
            nn.Conv1d(64, 128, 5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout1d(0.5),
            
            nn.Flatten(),
            
            nn.Linear(128 * 99, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, X):
        # X should be LongTensor (token indices)
        embedded = self.embedding(X)
        
        #### Gemini Suggestions ####
        # Permute the dimensions to be (batch_size, embedding_dim, sequence_length)
        embedded = embedded.permute(0, 2, 1)
        ############################
        
        out = self.net(embedded)
        return out


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, *, hidden_size=128, num_layers=1, bidirectional=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.direction = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False # False = Freeze embeddings
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=bidirectional)
        
        # Sequential layer
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * self.direction, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        embedded = self.embedding(X)
        
        h0 = torch.zeros(self.layers * self.direction, embedded.size(0), self.hidden_size).to(device)
        out, hidden = self.rnn(embedded, h0)
        
        pooled = out.mean(dim=1)
        log_probs = self.net(pooled)
        return log_probs


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, *, hidden_size=128, num_layers=1, bidirectional=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.direction = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False # Freeze embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=bidirectional)
        
        # Sequential layer
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_size * self.direction, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, X):
        embedded = self.embedding(X)
        
        h0 = torch.zeros(self.layers * self.direction, embedded.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.layers * self.direction, embedded.size(0), self.hidden_size).to(device)
        out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        pooled = out.mean(dim=1)
        log_probs = self.net(pooled)
        return log_probs
