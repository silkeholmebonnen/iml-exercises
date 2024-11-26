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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_data(noise=0.2):
    # Generate synthetic data
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Ensure target has shape [N, 1]
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) # Ensure target has shape [N, 1]
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train, X_test, y_train, y_test, X, y


def train(to_train, activation_name, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train, X_test, y_train, y_test, epoch=100, loss=nn.BCELoss(), decision_threshold=0.5):
    
    decision_threshold = decision_threshold # Decision threshold set here
    name = activation_name
    model = to_train

    # Define loss function and optimizer
    criterion = loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = epoch
    train_losses = []

    # Measure training time
    start_time = time.time()
        
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        train_losses.append(loss.item())

    # Measure elapsed time
    end_time = time.time()
    training_time = end_time - start_time

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (y_pred > decision_threshold).float()
        accuracy = accuracy_score(y_test, y_pred_class.numpy().ravel())



    return model, train_losses, accuracy, training_time, decision_threshold











