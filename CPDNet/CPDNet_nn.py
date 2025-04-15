"""# **Class: CPDNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class CPDNetNN(nn.Module):
    def __init__(self, sample_interval, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(CPDNetNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_interval = sample_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LSTM1 layer
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        # LSTM2 layer
        self.lstm2 = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        # Fully connected layer1
        self.fc1 = nn.Linear(hidden_size, output_size)
        
        # Additional layers
        self.activation = nn.ReLU()
        self.normalization = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout layer
        
        # Move all layers to device
        self.to(self.device)

    def forward(self, x):
        # Initialize hidden states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.view(batch_size, self.sample_interval, 1)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply batch normalization (before activation)
        out = self.normalization(out.transpose(1, 2)).transpose(1, 2)
        
        # Get final output - no need to reshape
        out = self.fc(out)

        # Apply activation
        out = self.activation(out)

        # Add sigmoid activation to ensure output values are between 0 and 1
        out = torch.tanh(out)
        return out[:, -1, :]