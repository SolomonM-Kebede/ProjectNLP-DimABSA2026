
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss, title="Loss vs Epoch"):
    epochs = range(1, len(train_loss)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, val_loss, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_metrics(metrics_df, title="Metrics per Epoch"):
    """
    For subtask1 CSV metrics: RMSE, CCC, PCC, etc.
    metrics_df: pandas DataFrame
    """
    plt.figure(figsize=(10,6))
    for col in metrics_df.columns:
        plt.plot(metrics_df.index, metrics_df[col], marker='o', label=col)
    plt.xlabel("Epoch / Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_signal_spectrum(step_losses, title="Signal Spectrum of Training Loss"):
    """
    Plots the spectrum (loss per step) to see jitter / spikes
    """
    plt.figure(figsize=(10,5))
    plt.plot(step_losses, color='purple', alpha=0.7)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt

def mark_best_epoch(train_loss, val_loss, best_epoch, title="Loss with Best Epoch"):
    """
    Highlights the best epoch on the train/val loss plot
    """
    plt = plot_loss(train_loss, val_loss, title=title)
    plt.axvline(x=best_epoch+1, color='red', linestyle='--', label="Best Epoch")
    plt.legend()
    return plt

def plot_f1_with_best_epoch(f1_scores, best_epoch, title="F1 Score vs Epoch"):
    epochs = range(1, len(f1_scores) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1_scores, marker='o', label="F1 Score")

    # Mark best epoch
    plt.axvline(x=best_epoch, color='red', linestyle='--', label="Best Epoch")
    plt.scatter(best_epoch, f1_scores[best_epoch], zorder=5)

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt