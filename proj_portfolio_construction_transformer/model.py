# model.py

import torch
import torch.nn as nn
import math
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import logging
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: (ceil(d_model/2),)

        pe[:, 0::2] = torch.sin(position * div_term)  # Assign sine to even indices

        # Handle odd d_model by limiting the div_term slice
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # Assign cosine to odd indices

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor with positional encoding added, shape (batch_size, seq_length, d_model)
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size=5, num_layers=2, dropout=0.1, nhead=2, dim_feedforward=128):
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Set batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)  # Predicting a single value (e.g., next day's close price)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_length, features)
        Returns:
            Tensor of shape (batch_size,)
        """
        src = self.pos_encoder(src)  # Add positional encoding
        output = self.transformer_encoder(src)  # Shape: (batch_size, seq_length, d_model)
        output = output.mean(dim=1)  # Aggregate over the sequence dimension
        output = self.decoder(output).squeeze(1)  # Shape: (batch_size,)
        return output


def train_model(model, train_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """
    Train the transformer model.

    Args:
        model (nn.Module): The transformer model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cpu' or 'cuda').
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}')
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}')


def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    batch_size = 32  # Adjust based on GPU memory availability

    # Create DataLoader for test data
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Calculate evaluation metrics
    mse = np.mean((predictions - actuals) ** 2)
    r2 = 1 - np.sum((predictions - actuals) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

    results = {
        'mse': mse,
        'r2': r2,
        'predictions': pd.DataFrame({
            'Actual': actuals.flatten(),
            'Predicted': predictions.flatten()
        })
    }
    return results
