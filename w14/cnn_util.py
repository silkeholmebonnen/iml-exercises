import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def evaluate_models_and_plot(models_dict, class_names):
    """
    Evaluates each model in the given dictionary and plots accuracy, precision, recall, 
    along with a confusion matrix heatmap.

    Args:
        models_dict (dict): Dictionary with model names as keys and trained PyTorchTrainer objects as values.
        class_names (list): List of class names for labeling.
    """
    metrics_dict = {}


    for model_name, trainer in models_dict.items():
        logger, _ = trainer.evaluate() 
        metrics_dict[model_name] = logger


    evaluate(metrics_dict)

def evaluate(models_dict):
    """
    Evaluates PyTorchTrainer models and plots:
    1. Per-class precision, recall, and accuracy for each model.
    2. Confusion matrices for all models.

    Args:
        models_dict (dict): Dictionary with model names as keys and PyTorchTrainer instances as values.
        class_names (list): List of class names for x and y labels in the heatmap.

    Returns:
        None
    """
    model_names = list(models_dict.keys())
    class_names = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    num_classes = len(class_names)

    # Initialize metric storage
    per_class_metrics = {"accuracy": [], "precision": [], "recall": []}
    confusion_matrices = {}

    # Evaluate models
    for model_name, trainer in models_dict.items():
        logger, _ = trainer.evaluate()

        # Per-class metrics
        per_class_metrics["accuracy"].append(logger.correct / logger.actual_positive)
        per_class_metrics["precision"].append(logger.precision)
        per_class_metrics["recall"].append(logger.recall)

        # Confusion matrix
        confusion_matrices[model_name] = logger.mat

    num_models = len(model_names)
    rows = 1 + -(-num_models // 3)  # 1 row for metrics + enough rows for confusion matrices

    fig, axs = plt.subplots(rows, 3, figsize=(18, 6 * rows))

    # Plot 1: Per-Class Accuracy
    bar_width = 0.2
    indices = np.arange(num_classes)
    for i, model_name in enumerate(model_names):
        axs[0, 0].bar(
            indices + i * bar_width,
            per_class_metrics["accuracy"][i],
            width=bar_width,
            label=model_name,
            alpha=0.8
        )
    axs[0, 0].set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    axs[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axs[0, 0].set_title("Class-wise Accuracy")
    axs[0, 0].set_xlabel("Class")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].legend(loc='best')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Per-Class Precision
    for i, model_name in enumerate(model_names):
        axs[0, 1].bar(
            indices + i * bar_width,
            per_class_metrics["precision"][i],
            width=bar_width,
            label=model_name,
            alpha=0.8
        )
    axs[0, 1].set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    axs[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axs[0, 1].set_title("Class-wise Precision")
    axs[0, 1].set_xlabel("Class")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].legend(loc='best')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 3: Per-Class Recall
    for i, model_name in enumerate(model_names):
        axs[0, 2].bar(
            indices + i * bar_width,
            per_class_metrics["recall"][i],
            width=bar_width,
            label=model_name,
            alpha=0.8
        )
    axs[0, 2].set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    axs[0, 2].set_xticklabels(class_names, rotation=45, ha='right')
    axs[0, 2].set_title("Class-wise Recall")
    axs[0, 2].set_xlabel("Class")
    axs[0, 2].set_ylabel("Recall")
    axs[0, 2].legend(loc='best')
    axs[0, 2].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot confusion matrices for all models
    for idx, model_name in enumerate(model_names):
        row = (idx // 3) + 1  # Start from row 1 (second row in grid)
        col = idx % 3
        sns.heatmap(
            confusion_matrices[model_name],
            annot=True,
            fmt=".0f",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            ax=axs[row, col]
        )
        axs[row, col].set_title(f"{model_name}")
        axs[row, col].set_xlabel("Predicted Class")
        axs[row, col].set_ylabel("True Class")

    # Hide unused subplots
    total_plots = rows * 3
    for idx in range(num_models + 3, total_plots):
        row = idx // 3
        col = idx % 3
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_overall_metrics(models_dict):
    """
    Plots overall accuracy, precision, and recall for all models in the dictionary.

    Args:
        models_dict (dict): Dictionary with model names as keys and PyTorchTrainer instances as values.

    Returns:
        None
    """
    # Extract metrics
    model_names = list(models_dict.keys())
    accuracies = []
    precisions = []
    recalls = []

    for model_name, trainer in models_dict.items():
        logger, _ = trainer.evaluate()  # Get MetricLogger and predictions
        accuracies.append(logger.accuracy)
        precisions.append(logger.precision.mean())
        recalls.append(logger.recall.mean())

    # Create a row of three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Overall Accuracy
    axs[0].bar(model_names, accuracies, color='#1f77b4', alpha=0.8)
    axs[0].set_title("Overall Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xticks(range(len(model_names)))
    axs[0].set_xticklabels(model_names, rotation=45, ha='right')
    axs[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot Overall Precision
    axs[1].bar(model_names, precisions, color='#ff7f0e', alpha=0.8)
    axs[1].set_title("Overall Precision")
    axs[1].set_ylabel("Precision")
    axs[1].set_xticks(range(len(model_names)))
    axs[1].set_xticklabels(model_names, rotation=45, ha='right')
    axs[1].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot Overall Recall
    axs[2].bar(model_names, recalls, color='#2ca02c', alpha=0.8)
    axs[2].set_title("Overall Recall")
    axs[2].set_ylabel("Recall")
    axs[2].set_xticks(range(len(model_names)))
    axs[2].set_xticklabels(model_names, rotation=45, ha='right')
    axs[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


