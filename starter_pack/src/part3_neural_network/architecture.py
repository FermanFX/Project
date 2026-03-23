"""
PART 3.1: NEURAL NETWORK ARCHITECTURE (10 points)
==================================================
PyTorch Neural Network - ALGORITHM parts marked with TODO
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


class FootballNN(nn.Module):
    """
    Neural Network for Football Match Prediction.
    
    ALGORITHM:
    TODO: Implement a feedforward neural network with:
    
    1. Input Layer: Takes feature vector x (n_features,)
    
    2. Hidden Layers:
       TODO: Each layer computes: h = f(Wx + b)
       - W: weight matrix (input_size, output_size)
       - b: bias vector (output_size,)
       - f: activation function (ReLU, Tanh, etc.)
    
    3. Output Layer:
       TODO: For 3-class classification, use:
       - Linear layer to 3 outputs
       - Softmax for probability distribution
    
    Architecture choices:
    TODO: Experiment with:
    - Number of hidden layers (2-4)
    - Hidden layer sizes (64, 128, 256)
    - Activation functions
    - Dropout for regularization
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout_rate=0.3):
        """
        Initialize the network architecture.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(FootballNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        ALGORITHM:
        TODO: Implement forward computation:
        1. Pass input through hidden layers
        2. Apply output layer
        3. Return raw logits (no softmax - use CrossEntropyLoss)
        
        Equation: output = W_out * h_last + b_out
        """
        # TODO: Implement forward pass
        pass


class ResidualBlock(nn.Module):
    """
    Residual block for deeper networks.
    
    ALGORITHM:
    TODO: Implement skip connection:
    output = F(x) + x
    
    Where F(x) is the main path transformation.
    This helps with gradient flow in deeper networks.
    """
    
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        # TODO: Define layers
        pass
    
    def forward(self, x):
        # TODO: Implement residual connection
        pass


class AttentionFootballNN(nn.Module):
    """
    Neural Network with attention mechanism.
    
    ALGORITHM:
    TODO: Implement attention mechanism:
    
    1. Compute attention scores:
       TODO: attention_scores = softmax(v^T * tanh(W * h))
    
    2. Compute weighted sum:
       TODO: context = sum(attention_weights * values)
    
    This allows the network to focus on important features.
    """
    
    def __init__(self, input_size, num_heads=4, hidden_size=64):
        super(AttentionFootballNN, self).__init__()
        # TODO: Implement attention layers
        pass
    
    def forward(self, x):
        # TODO: Implement attention forward pass
        pass


def build_model(input_size, config=None):
    """
    Factory function to build neural network.
    
    Args:
        input_size: Number of input features
        config: Dictionary with architecture parameters
    """
    if config is None:
        config = {
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3
        }
    
    model = FootballNN(
        input_size=input_size,
        hidden_sizes=config.get('hidden_sizes', [128, 64, 32]),
        output_size=3,
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
