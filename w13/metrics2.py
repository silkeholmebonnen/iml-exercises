import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluateNN(results, X, y):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Performance with Different Activation Functions')

    # Define levels for highlighting
    highlight_levels = [0.5]

    # Plot decision boundaries
    for idx, (name, result) in enumerate(results.items()):
        ax = axs[idx // 2, idx % 2]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01))
        Z = result['model'](torch.cat((xx.ravel().unsqueeze(1), yy.ravel().unsqueeze(1)), dim=1))
        Z = Z.view(xx.shape).detach().numpy()

        # Plot filled contours
        contour_filled = ax.contourf(xx.numpy(), yy.numpy(), Z, alpha=0.8, levels=[0, 0.2, 0.4, 0.6, 0.8, 1], cmap='coolwarm')
        
        # Plot contour lines for highlighting specific levels
        for level in highlight_levels:
            ax.contour(xx.numpy(), yy.numpy(), Z, levels=[level], colors='black', linewidths=2)
            ax.contour(xx.numpy(), yy.numpy(), Z, levels=[level], colors='red', linewidths=1, linestyles='--')  # Optional: Add dashed lines

        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        ax.set_title(f'Decision Boundary ({name})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f'{name}')
    plt.title('Loss Curves for Different Activation Functions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
   

    # Print accuracies and training times
    for name, result in results.items():
        print(f'Activation: {name}, Test Accuracy: {result["accuracy"]:.2f}, Training Time: {result["training_time"]:.2f} seconds')

def evaluateNN2(results, X, y, X_test_tensor, y_test, decision_threshold=0.5):

    # Create subplots for decision boundaries
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Decision Boundaries with Different Loss Functions')

    # Plot decision boundaries
    for idx, (loss_name, result) in enumerate(results.items()):
        ax = axs[idx]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01))
        Z = result['model'](torch.cat((xx.ravel().unsqueeze(1), yy.ravel().unsqueeze(1)), dim=1))
        Z = Z.view(xx.shape).detach().numpy()

        # Plot decision boundary with highlighted level 0.5
        contour = ax.contourf(xx.numpy(), yy.numpy(), Z, alpha=0.8, levels=[0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
        ax.contour(xx.numpy(), yy.numpy(), Z, levels=[0.5], colors='red', linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        ax.set_title(f'Loss: {loss_name}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.title('Loss Curves for Different Loss Functions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    # Print accuracies and additional metrics
    for loss_name, result in results.items():
        y_pred_class = (result['model'](X_test_tensor) > result['decision_threshold']).float()
        precision = precision_score(y_test, y_pred_class.numpy())
        recall = recall_score(y_test, y_pred_class.numpy())
        f1 = f1_score(y_test, y_pred_class.numpy())
        print(f'Loss: {loss_name}, Test Accuracy: {result["accuracy"]:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

