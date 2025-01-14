import os
import pickle
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import time

from fashionmnist_utils import mnist_reader
from metrics import MetricLogger


class Trainer(ABC):
    def __init__(self, model):
        self.model = model
        self.name = (
            f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'
        )

    @abstractmethod
    def train(self, *args):
        ...

    @abstractmethod
    def predict(self, input):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        ...



def get_data(transform, train=True):
    return FashionMNIST(os.getcwd(), train=train, transform=transform, download=True)

'''
class PyTorchTrainer(Trainer):
    def __init__(self, nn_module, transform, optimizer, batch_size):
        super().__init__(nn_module)

        self.train_data, self.val_data, self.test_data = None, None, None

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.init_data()

        self.logger = SummaryWriter()

    def init_data(self):
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)

        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)


    def train(self, epochs):
        """Train the model using SGD. 

        Args:
            epochs (int): The total number of training epochs.
        """
        self.logger.add_graph(self.model, next(iter(self.train_data))[0])

        update_interval = len(self.train_data) // 5

        train_logger = MetricLogger(classes=10)
        val_logger = MetricLogger(classes=10)
        
        
        # Early stopping  
        # Uncomment lines below this if you want early stopping \
        #last_loss = 1000
        #patience = 5
        #triggertimes = 0
        # / Early stopping
        for e in range(epochs):
            print(f"[Epoch {e + 1}]")
            running_loss = 0.0

            self.model.train()
            for i, (x, y) in enumerate(tqdm(self.train_data, leave=None)):
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_logger.log(out, y)

                if (i + 1) % update_interval == 0:
                    self.logger.add_scalar(
                        "loss",
                        running_loss / update_interval,
                        global_step=i + e * len(self.train_data),
                    )
                    self.logger.add_scalar(
                        "train_accuracy",
                        train_logger.accuracy,
                        i + e * len(self.train_data),
                    )
                    train_logger.reset()
                    running_loss = 0.0

            val_logger.reset()
            self.model.eval()
            running_val_loss=0
            for x, y in tqdm(self.val_data, leave=None):
                out = self.model(x)
                loss_val = F.cross_entropy(out, y)
                val_logger.log(out, y)
                running_val_loss += loss_val.item()

            self.logger.add_scalar("accuracy", val_logger.accuracy, e)


            # Early stopping
            # Uncomment lines below this if you want early stopping \
            # current_acc = running_val_loss

            # if current_loss > last_loss:
            #     trigger_times += 1
            #     print('Trigger Times:', trigger_times)

            #     if trigger_times >= patience:
            #         print('Early stopping!\nStart to test process.')
            #         break

            # else:
            #     print('trigger times: 0')
            #     trigger_times = 0

            # last_loss = current_loss
            ## / Early stopping

            print(
                f"[Validation] acc: {val_logger.accuracy:.4f}, precision: {val_logger.precision.mean():.4f}, recall: {val_logger.recall.mean():.4f}"
            )
            
            
    def train_es(self, epochs, patience):
        """Train the model using SGD. 

        Args:
            epochs (int): The total number of training epochs.
        """
        self.logger.add_graph(self.model, next(iter(self.train_data))[0])

        update_interval = len(self.train_data) // 5

        train_logger = MetricLogger(classes=10)
        val_logger = MetricLogger(classes=10)
        
        
        # Early stopping  
        last_loss = 1000
        trigger_times = 0
        # / Early stopping
        for e in range(epochs):
            print(f"[Epoch {e + 1}]")
            running_loss = 0.0

            self.model.train()
            for i, (x, y) in enumerate(tqdm(self.train_data, leave=None)):
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_logger.log(out, y)

                if (i + 1) % update_interval == 0:
                    self.logger.add_scalar(
                        "loss",
                        running_loss / update_interval,
                        global_step=i + e * len(self.train_data),
                    )
                    self.logger.add_scalar(
                        "train_accuracy",
                        train_logger.accuracy,
                        i + e * len(self.train_data),
                    )
                    train_logger.reset()
                    running_loss = 0.0

            val_logger.reset()
            self.model.eval()
            running_val_loss=0
            for x, y in tqdm(self.val_data, leave=None):
                out = self.model(x)
                loss_val = F.cross_entropy(out, y)
                val_logger.log(out, y)
                running_val_loss += loss_val.item()

            self.logger.add_scalar("accuracy", val_logger.accuracy, e)


            # Early stopping

            current_loss = running_val_loss

            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break

            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss
            ## / Early stopping

            print(
                f"[Validation] acc: {val_logger.accuracy:.4f}, precision: {val_logger.precision.mean():.4f}, recall: {val_logger.recall.mean():.4f}"
            )

    def predict(self, input):
        """Generate predictions for the specified input.

        Args:
            input (tensor): A BxN (for MLP) or Bx1xHxW (for CNN) shaped tensor.

        Returns:
            _type_: _description_
        """
        return self.model(input)

    def evaluate(self):
        """Test the model on the test dataset and collect metric information and predictions.

        Returns:
            (MetricLogger, tensor): The logging results as well as the predictions as class labels.
        """
        test_logger = MetricLogger(classes=10)
        predictions = []
        self.model.eval()
        for x, y in tqdm(self.test_data, leave=None):
            out = self.model(x)
            predictions.append(torch.argmax(out, dim=1))
            test_logger.log(out, y)
        return test_logger, torch.cat(predictions)

    def save(self):
        """Save the trained model to disk.
        """
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join("models", self.name)
        with open(file_name + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str):
        """Instantiate a model from a saved state.
        """
        with open(path, "rb") as file:
            new = pickle.load(file)
            new.init_data()
            return new

'''
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt




class PyTorchTrainer(Trainer):
    def __init__(self, nn_module, transform, optimizer, batch_size):
        super().__init__(nn_module)
        """Initialize the PyTorchTrainer with data loaders and the model."""
        self.model = nn_module
        self.transform = transform
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.train_data = None
        self.val_data = None
        self.test_data = None
        

        self.init_data()

        # Logger
        self.logger = SummaryWriter()

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.training_time = 0
        self.per_class_accuracies = {}


    def init_data(self):
        """Initialize training, validation, and test data loaders."""
        # Assuming get_data is a function to load your dataset
        data = get_data(self.transform, train=True)
        test_data = get_data(self.transform, train=False)

        val_len = int(len(data) * 0.2)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=self.batch_size)
        self.test_data = DataLoader(test_data, batch_size=self.batch_size)

    def train(self, epochs):
        """Train the model using SGD.

        Args:
            epochs (int): The total number of training epochs.
        """
        train_logger = MetricLogger(classes=10)
        val_logger = MetricLogger(classes=10)

        start_time = time.time()  # Start timer

        for e in range(epochs):
            print(f"[Epoch {e + 1}]")
            running_train_loss = 0.0

            self.model.train()
            for x, y in tqdm(self.train_data, leave=None):
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                train_logger.log(out, y)

            avg_train_loss = running_train_loss / len(self.train_data)
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_logger.accuracy)

            # Validation
            val_logger.reset()
            self.model.eval()
            running_val_loss = 0.0
            for x, y in tqdm(self.val_data, leave=None):
                out = self.model(x)
                loss_val = F.cross_entropy(out, y)
                val_logger.log(out, y)
                running_val_loss += loss_val.item()

            avg_val_loss = running_val_loss / len(self.val_data)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_logger.accuracy)

            # Log metrics to tensorboard
            self.logger.add_scalar("Train Loss", avg_train_loss, e)
            self.logger.add_scalar("Train Accuracy", train_logger.accuracy, e)
            self.logger.add_scalar("Validation Loss", avg_val_loss, e)
            self.logger.add_scalar("Validation Accuracy", val_logger.accuracy, e)

            # Update per-class metrics
            self.per_class_accuracies = {
                "accuracy": val_logger.accuracy,
                "precision": val_logger.precision.tolist(),
                "recall": val_logger.recall.tolist()
            }

            print(
                f"[Validation] Epoch {e + 1}: acc: {val_logger.accuracy:.4f}, val_loss: {avg_val_loss:.4f}, train_acc: {train_logger.accuracy:.4f}"
            )

        self.training_time = time.time() - start_time  # End timer


    def plot_metrics(self, class_names):
        """
        Plots training and validation losses, accuracies, and per-class metrics.

        Args:
            class_names (list): List of class names to display on the x-axis for per-class metrics.
        """
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(18, 6))

        # Plot 1: Losses
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.name}\nTraining Time: {self.training_time:.2f} seconds')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Accuracies
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy', color='#1f77b4', linewidth=2)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='#ff7f0e', linestyle='--', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Plot 3: Per-Class Metrics (Precision, Recall, and Accuracy)
        plt.subplot(1, 3, 3)
        precisions = self.per_class_accuracies["precision"]
        recalls = self.per_class_accuracies["recall"]
        accuracies = self.per_class_accuracies["accuracy"]

        bar_width = 0.2
        indices = np.arange(len(class_names))

        plt.bar(indices - bar_width, precisions, width=bar_width, label='Precision', alpha=0.8, color='#1f77b4')
        plt.bar(indices, recalls, width=bar_width, label='Recall', alpha=0.8, color='#ff7f0e')
        plt.bar(indices + bar_width, accuracies, width=bar_width, label='Accuracy', alpha=0.8, color='#2ca02c')

        plt.xticks(indices, class_names, rotation=45, ha='right', fontsize=10) 
        plt.xlabel("Class")
        plt.ylabel("Metrics")
        plt.title("Per-Class Metrics (validation data): Precision, Recall, and Accuracy")
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()




    def predict(self, input):
        """Generate predictions for the specified input."""
        self.model.eval()
        with torch.no_grad():
            return self.model(input)

    def evaluate(self):
        """Evaluate the model on the test dataset."""
        test_logger = MetricLogger(classes=10)
        predictions = []
        self.model.eval()
        for x, y in tqdm(self.test_data, leave=None):
            out = self.model(x)
            predictions.append(torch.argmax(out, dim=1))
            test_logger.log(out, y)
        return test_logger, torch.cat(predictions)

    def save(self):
        """Save the trained model to disk."""
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join("models", self.name)
        with open(file_name + ".pkl", "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(path: str):
        """Instantiate a model from a saved state."""
        with open(path, "rb") as file:
            new = pickle.load(file)
            new.init_data()
            return new


