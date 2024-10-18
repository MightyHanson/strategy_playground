# predict.py

import torch
import numpy as np
from model import TransformerTimeSeries
from data_loader import DataLoader
from preprocess import create_sequences
import os
import pickle
import logging
import sys

def load_scaler(output_dir):
    """
    Load the scaler object from disk.

    Args:
        output_dir (str): Directory where the scaler is saved.

    Returns:
        StandardScaler: Loaded scaler object.
    """
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        logging.error(f"Scaler file {scaler_path} not found.")
        sys.exit(1)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logging.info(f"Loaded scaler from {scaler_path}")
    return scaler

def predict_future(model, recent_data, scaler, seq_length=30, device='cpu'):
    """
    Predict the next close price based on recent data.

    Args:
        model (nn.Module): The trained transformer model.
        recent_data (np.ndarray): Recent feature data.
        scaler (StandardScaler): Scaler used for feature scaling.
        seq_length (int): Sequence length.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        float: Predicted close price.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Scale the recent data
        scaled_recent_data = scaler.transform(recent_data)
        input_seq = torch.tensor(scaled_recent_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, seq_length, features)
        prediction = model(input_seq).squeeze().cpu().item()
    return prediction
