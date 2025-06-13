"""
Refactored QC training script for ring resonator Qc prediction using attention-based or ResNet models.
"""

import os
import time
import random

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.autonotebook import tqdm

import humanize
import psutil
import GPUtil
import seaborn as sns
import matplotlib.pyplot as plt

from Attention import RNN_Dataset, train_network_reg
from util import rsnet18
from sklearn.metrics import r2_score

# ---------------------- Configuration ----------------------
# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ETA = 0.001            # Learning rate
STEP_SIZE = 10         # LR decay period
GAMMA = 0.5            # LR decay factor
EPOCHS = 1             # Number of training epochs
BATCH_SIZE = 32        # Batch size

# Dataset parameters
N_FREQ_TRAIN = 400     # Number of frequencies for training
N_GAP = 40             # Frequency gap between train/test
DATA_PATH = "Qc.pt"  # Path to the dataset file

# Feature toggles
ENABLE_PLOTTING = False  # Set True to display training plots

# -------------------- Utility Functions --------------------

def setup_environment():
    """Print CPU and GPU memory usage."""
    print(f"CPU RAM Free: {humanize.naturalsize(psutil.virtual_memory().available)}")
    for gpu in GPUtil.getGPUs():
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% used")


def load_dataset(path: str,
                 n_freq_train: int,
                 n_gap: int
                 ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """
    Load the Qc dataset and split into training and test sets.

    Returns:
        train_ds: Training subset
        test_ds: Testing subset
        output_dim: Number of target frequencies to predict
    """
    data = torch.load(path)
    total_samples, freq_count = data.shape
    output_dim = freq_count - n_freq_train - n_gap
    dataset = RNN_Dataset(dataset=data, n=n_freq_train, input_dim=None, ngap=n_gap)
    train_size = total_samples - 1000
    test_size = 1000
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    return train_ds, test_ds, output_dim


def create_data_loaders(train_ds, test_ds, batch_size: int):
    """Create PyTorch DataLoaders for train and test sets."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class RMSELoss(nn.Module):
    """Root-mean-square error loss."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)


def build_model(output_dim: int) -> nn.Module:
    """Instantiate the model (ResNet or attention-based)."""
    model = rsnet18(num_classes=output_dim)
    return model.to(DEVICE)


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader
                       ) -> "pd.DataFrame":
    """
    Train the model and evaluate on test set.

    Returns:
        results_df: DataFrame containing metrics per epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=ETA)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss_fn = RMSELoss()

    start_time = time.time()
    results_df = train_network_reg(
        model,
        loss_fn,
        train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        score_funcs={"R^2 score": r2_score},
        device=DEVICE,
        optimizer=optimizer,
        lr_schedule=scheduler
    )
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f} sec")
    return results_df


def plot_results(results_df):
    """Plot the test R^2 score over epochs."""
    sns.lineplot(x="epoch", y="test R^2 score", data=results_df.iloc[1:])
    plt.title("Test R^2 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("R^2 Score")
    plt.gcf().set_size_inches(10, 6)
    plt.show()


def main():
    setup_environment()

    train_ds, test_ds, output_dim = load_dataset(DATA_PATH, N_FREQ_TRAIN, N_GAP)
    train_loader, test_loader = create_data_loaders(train_ds, test_ds, BATCH_SIZE)

    model = build_model(output_dim)
    print(model)

    results = train_and_evaluate(model, train_loader, test_loader)
    print("Max train R^2:", results["train R^2 score"].max())
    print("Max test R^2:", results["test R^2 score"].max())

    if ENABLE_PLOTTING:
        plot_results(results)


if __name__ == "__main__":
    main()
