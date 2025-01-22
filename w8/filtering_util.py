import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from matplotlib.patches import Ellipse


def load_csv1(path):
    """
    Loads a CSV file for a specified test subject.

    Parameters:
        path (str): The path of the folder containing data in cvs format

    Returns:
        data (pd.DataFrame): The loaded CSV file as a pandas DataFrame.
    """
    data = pd.read_csv(path)
    
    return data




def open_img(path, idx):
    """
    Opens a single image from the specified directory and index.

    Parameters:
        path (str): The directory path where the image is located.
        idx (int): The index of the image file (assumed to be in the format '{idx}.jpg').

    Returns:
        np.ndarray: The image as a NumPy array.

    Raises:
        IOError: If the image cannot be read from the specified path.
    """
    img = io.imread(path + f'/{idx}.jpg')
    if img is None:
        raise IOError("Could not read image")
    return img 

def visualize_pupil_centers(csv_file, pattern):
    """
    Create a scatter plot of the detected pupil centers, find the top N most populated grid areas, 
    and calculate the mean center for each.

    Parameters:
        csv_file (str): The path to the CSV file containing the pupil coordinates (px, py).
        output_directory (str): The directory where the scatter plot image will be saved.
        grid_size (int): The size of each grid cell in pixels (default is 7x7).
        top_n (int): The number of top populated areas to consider (default is 10).
    """
    print(os.path.join(os.getcwd(), csv_file))

    df = pd.read_csv(os.path.join(os.getcwd(), csv_file))

    plt.figure(figsize=(8, 8))
    plt.scatter(df['px'], df['py'], c='red', marker='o', label='Pupil Centers')
    
    plt.xlim(150, 60) 
    plt.ylim(150, 60)  

    plt.xlabel('X Coordinate (px)')
    plt.ylabel('Y Coordinate (px)')
    plt.title('Scatter Plot of Pupil Centers')
    plt.legend()
    plt.show()


def visualize_gaussian_1d_filter(truncate=4.0, filter_size=100):
    """
    Visualize the Gaussian 1D filter in two ways:
    1. Plot the Gaussian filter itself.
    2. Apply the Gaussian filter to a sample signal and show the smoothed result.

    Args:
        truncate (float): Truncate the filter at this many standard deviations.
        filter_size (int): Number of points to represent the filter.
    """
    # Define a color map for better distinction
    colors = plt.cm.viridis(np.linspace(0, 1, 11))  # Using 11 colors for sigmas from 1 to 11

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loop over different values of sigma
    for idx, sigma in enumerate(range(1, 12)):
        # 1. Generate a 1D Gaussian filter
        x = np.linspace(-truncate * sigma, truncate * sigma, filter_size)
        gaussian_filter = np.exp(-0.5 * (x / sigma) ** 2)
        gaussian_filter /= gaussian_filter.sum()  # Normalize the filter

        # Plot 1: The Gaussian filter itself
        axes[0].plot(x, gaussian_filter, label=f'sigma={sigma}', color=colors[idx])
        axes[0].set_title('1D Gaussian Filters', fontsize=14)
        axes[0].set_xlabel('Position', fontsize=12)
        axes[0].set_ylabel('Amplitude', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Apply the Gaussian filter to a sample signal (step function with noise)
        np.random.seed(42)
        signal = np.concatenate([np.ones(50), np.zeros(50)]) + np.random.normal(0, 0.1, 100)
        smoothed_signal = gaussian_filter1d(signal, sigma=sigma)

        # Plot 2: The effect of the Gaussian filter on the signal
        if idx == 0:  # Only plot the original signal once
            axes[1].plot(signal, label='Original Signal (noisy)', linestyle='--', color='gray', alpha=0.7)
        axes[1].plot(smoothed_signal, label=f'Smoothed (sigma={sigma})', color=colors[idx])

    # Adjust labels and titles
    axes[1].set_title('Effect of Gaussian Filter on Noisy Signal', fontsize=14)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Combine legends outside the plot
    axes[0].legend(title="Gaussian Filters", fontsize=10)
    axes[1].legend(title="Signal Smoothing", fontsize=10)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_x_and_y(data_x,data_y):
    plt.figure(figsize=(12, 6))

    x = np.linspace(0, len(data_x), len(data_x))
    y = np.linspace(0, len(data_y), len(data_y))

    plt.subplot(2, 1, 1)
    plt.plot(x, data_x, color='royalblue', label='Pupil X Coordinate', linewidth=2)
    plt.title('Pupil X Coordinate Over Frames', fontsize=12)
    plt.xlabel('# Frame', fontsize=10)
    plt.ylabel('Pupil Coordinate (X)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y, data_y, color='seagreen', label='Pupil Y Coordinate', linewidth=2)
    plt.title('Pupil Y Coordinate Over Frames', fontsize=12)
    plt.xlabel('# Frame', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_x_and_y_complete(data_x,data_y, scale = 5):
    last_key = list(data_x.keys())[-1]
    plt.figure(figsize=(12, 6))
    for i in list(data_x.keys()):
        x = np.linspace(0, len(data_x[i]), len(data_x[i]))

        
        if i[:11] == 'underlaying':
            plt.subplot(2, 1, 1)
            plt.plot(x, data_x[i], linestyle='--', color='gray', alpha=0.5, label=f'{i} signal')

        
        elif 'derivative' in i:
            plt.subplot(2, 1, 1)
            plt.plot(x, data_x[i]*scale, alpha=0.7, label=f'{i} signal')

        
        elif i != 'underlaying' and i != last_key:
            plt.subplot(2, 1, 1)
            plt.plot(x, data_x[i], alpha=0.7, label=f'{i} signal')
        
        elif 'saccade' in i:
            plot_consecutive_bars(data_x[i], 50)


        else:
            plt.subplot(2, 1, 1)
            plt.plot(x, data_x[i], color = 'red', label=f'{i} signal')


    for i in list(data_y.keys()):
        y = np.linspace(0, len(data_y[i]), len(data_y[i]))
        
        if i[:11] == 'underlaying':

            plt.subplot(2, 1, 2)
            plt.plot(y, data_y[i], linestyle='--', color='gray', alpha=0.5, label=f'{i} signal')
        
        elif 'derivative' in i:

            plt.subplot(2, 1, 2)
            plt.plot(y, data_y[i]*scale, alpha=0.7, label=f'{i} signal')
    
        
        elif i != 'underlaying' and i != last_key:

            plt.subplot(2, 1, 2)
            plt.plot(y, data_y[i], alpha=0.7, label=f'{i} signal')

        elif 'saccade' in i:
            plot_consecutive_bars(data_y[i], 50)

        else:

            plt.subplot(2, 1, 2)
            plt.plot(y, data_y[i], color = 'red', label=f'{i} signal')

    plt.subplot(2, 1, 1)
    plt.title('Pupil X Coordinate Over Frames', fontsize=12)
    plt.xlabel('# Frame', fontsize=10)
    plt.ylabel('Pupil Coordinate (X)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.title('Pupil Y Coordinate Over Frames', fontsize=12)
    plt.xlabel('# Frame', fontsize=10)
    plt.ylabel('Pupil Coordinate (Y)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    plt.show()


def detect_spikes_derivatives(x_dev, y_dev):
    x_spikes = []
    y_spikes = []
    for i in range(len(x_dev)):
        if x_dev[i] != 0:
            x_spikes.append(i)
        if y_dev[i] != 0:
            y_spikes.append(i)
    return x_spikes, y_spikes


def remove_noise(px, py, x_spikes, y_spikes):
    px_new = []
    py_new = []
    px_recent = px[0]
    py_recent = py[0]
    for i in range(len(px)):
        if i not in x_spikes:
            px_new.append(px[i])
            px_recent = px[i]
        else:
            px_new.append(px_recent)

    for i in range(len(py)):
        if i not in y_spikes:
            py_new.append(py[i])
            py_recent = py[i]
        else:
            py_new.append(py_recent)
    return px_new, py_new


def remove_rows_by_index(arr, indices):
    """
    Remove rows from an Nx2 array based on a list of row indices.

    Args:
        arr (numpy array): The input array of shape (N, 2).
        indices (list or array): The list of row indices to be removed.

    Returns:
        numpy array: The augmented array with specified rows removed.
    """
    indices_set = set(indices)

    mask = np.array([i not in indices_set for i in range(arr.shape[0])])

    return arr[mask]


def plot_pupil_coor(px, py, label):
    plt.scatter(px, py, label=label)
    plt.xlabel('px')
    plt.ylabel('py')
    plt.xlim(150, 60)
    plt.ylim(150, 60)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()



def plot_statistics_comparison(x_orig, y_orig, x_clean, y_clean, datasets):
    """
    Given original and cleaned data, along with a dictionary specifying dataset ranges,
    this function calculates and plots the mean, variance, and covariance for both the original
    and cleaned data in the same plot for each dataset. All plots are displayed in a single figure.

    Args:
        x_orig (array): Original x data.
        y_orig (array): Original y data.
        x_clean (array): Cleaned x data.
        y_clean (array): Cleaned y data.
        datasets (dict): Dictionary with dataset names as keys and tuples (start, end) as values 
                         indicating the range of the data to use for each dataset.
    """
    num_datasets = len(datasets)
    fig, axs = plt.subplots(num_datasets, 3, figsize=(18, 6 * num_datasets))

    if num_datasets == 1:
        axs = [axs]

    for idx, (name, (start, end)) in enumerate(datasets.items()):
        x_o = x_orig[start:end]
        y_o = y_orig[start:end]
        x_c = x_clean[start:end]
        y_c = y_clean[start:end]

        mean_x_o = np.mean(x_o)
        mean_y_o = np.mean(y_o)
        variance_x_o = np.var(x_o)
        variance_y_o = np.var(y_o)
        cov_matrix_o = np.cov(x_o, y_o)

        mean_x_c = np.mean(x_c)
        mean_y_c = np.mean(y_c)
        variance_x_c = np.var(x_c)
        variance_y_c = np.var(y_c)
        cov_matrix_c = np.cov(x_c, y_c)

        padding_x = max(max(x_o) - min(x_o), max(x_c) - min(x_c)) 
        padding_y = max(max(y_o) - min(y_o), max(y_c) - min(y_c))

        x_values = np.linspace(min(min(x_o), min(x_c)) - padding_x, max(max(x_o), max(x_c)) + padding_x, 100)
        y_values = np.linspace(min(min(y_o), min(y_c)) - padding_y, max(max(y_o), max(y_c)) + padding_y, 100)

        gaussian_x_o = norm.pdf(x_values, mean_x_o, np.sqrt(variance_x_o))
        gaussian_y_o = norm.pdf(y_values, mean_y_o, np.sqrt(variance_y_o))
        gaussian_x_c = norm.pdf(x_values, mean_x_c, np.sqrt(variance_x_c))
        gaussian_y_c = norm.pdf(y_values, mean_y_c, np.sqrt(variance_y_c))

        axs[idx][0].plot(x_values, gaussian_x_o, color='tab:blue', linestyle='--', label='Original X Distribution')
        axs[idx][0].plot(x_values, gaussian_x_c, color='dodgerblue', label='Cleaned X Distribution')
        axs[idx][0].axvline(mean_x_o, color='tab:blue', linestyle='--', alpha=0.6)
        axs[idx][0].axvline(mean_x_c, color='dodgerblue', alpha=0.8)
        axs[idx][0].fill_between(x_values, gaussian_x_o, color='tab:blue', alpha=0.2)
        axs[idx][0].fill_between(x_values, gaussian_x_c, color='dodgerblue', alpha=0.2)

        axs[idx][0].plot(y_values, gaussian_y_o, color='tab:orange', linestyle='--', label='Original Y Distribution')
        axs[idx][0].plot(y_values, gaussian_y_c, color='darkorange', label='Cleaned Y Distribution')
        axs[idx][0].axvline(mean_y_o, color='tab:orange', linestyle='--', alpha=0.6)
        axs[idx][0].axvline(mean_y_c, color='darkorange', alpha=0.8)
        axs[idx][0].fill_between(y_values, gaussian_y_o, color='tab:orange', alpha=0.2)
        axs[idx][0].fill_between(y_values, gaussian_y_c, color='darkorange', alpha=0.2)

        axs[idx][0].set_title(f'{name}: Distribution Comparison')
        axs[idx][0].set_xlabel('Value')
        axs[idx][0].set_ylabel('Probability Density')
        axs[idx][0].legend()
        axs[idx][0].grid(True)

        axs[idx][1].bar(['Original x', 'Original y'], [variance_x_o, variance_y_o], color=['tab:blue', 'tab:orange'], alpha=0.6)
        axs[idx][1].bar(['Cleaned x', 'Cleaned y'], [variance_x_c, variance_y_c], color=['dodgerblue', 'darkorange'], alpha=0.8)
        axs[idx][1].set_title(f'{name}: Variance Comparison')
        axs[idx][1].set_ylabel('Variance')
        axs[idx][1].grid(True)

        axs[idx][2].scatter(x_o, y_o, alpha=0.5, label='Original Data', color='tab:gray')
        axs[idx][2].scatter(x_c, y_c, alpha=0.5, label='Cleaned Data', color='tab:green')

        eigenvalues_o, eigenvectors_o = np.linalg.eigh(cov_matrix_o)
        eigenvalues_c, eigenvectors_c = np.linalg.eigh(cov_matrix_c)

        for i in range(2):
            vector_o = eigenvectors_o[:, i] * np.sqrt(eigenvalues_o[i]) * 2
            vector_c = eigenvectors_c[:, i] * np.sqrt(eigenvalues_c[i]) * 2

            axs[idx][2].plot([mean_x_o, mean_x_o + vector_o[0]], [mean_y_o, mean_y_o + vector_o[1]], 
                             color='tab:blue', linewidth=2, linestyle='--', label=f'Original Variance Vector {i + 1}' if i == 0 else "")
            axs[idx][2].plot([mean_x_c, mean_x_c + vector_c[0]], [mean_y_c, mean_y_c + vector_c[1]], 
                             color='tab:green', linewidth=2, label=f'Cleaned Variance Vector {i + 1}' if i == 0 else "")

        angle_o = np.degrees(np.arctan2(*eigenvectors_o[:, 0][::-1]))
        width_o, height_o = 2 * np.sqrt(eigenvalues_o)
        ellipse_o = Ellipse(xy=(mean_x_o, mean_y_o), width=width_o, height=height_o, angle=angle_o,
                            edgecolor='tab:blue', facecolor='none', linestyle='--', linewidth=2, label='Original Covariance Ellipse')
        axs[idx][2].add_patch(ellipse_o)

        angle_c = np.degrees(np.arctan2(*eigenvectors_c[:, 0][::-1]))
        width_c, height_c = 2 * np.sqrt(eigenvalues_c)
        ellipse_c = Ellipse(xy=(mean_x_c, mean_y_c), width=width_c, height=height_c, angle=angle_c,
                            edgecolor='tab:green', facecolor='none', linestyle='-', linewidth=2, label='Cleaned Covariance Ellipse')
        axs[idx][2].add_patch(ellipse_c)

        axs[idx][2].set_title(f'{name}: Covariance Comparison')
        axs[idx][2].set_xlabel('x')
        axs[idx][2].set_ylabel('y')
        axs[idx][2].legend()
        axs[idx][2].grid(True)
        axs[idx][2].invert_xaxis()
        axs[idx][2].invert_yaxis()

    plt.tight_layout()
    plt.show()

def plot_consecutive_bars(numbers, y_fixed):
    numbers = sorted(numbers)
    
    groups = find_consecutive_groups(numbers)
    first = True
    # Plot each group as a horizontal bar
    for start, end in groups:
        if first:
            plt.hlines(y=y_fixed, xmin=start, xmax=end, colors='black', linewidth=1, label='saccade detected')
            first = False
        else:
            plt.hlines(y=y_fixed, xmin=start, xmax=end, colors='black', linewidth=1)
            # Plot vertical line at the start
            plt.vlines(x=start, ymin=y_fixed - 2, ymax=y_fixed + 2, colors='black', linewidth=1)
            # Plot vertical line at the end
            plt.vlines(x=end, ymin=y_fixed - 2, ymax=y_fixed + 2, colors='black', linewidth=1)
        

def find_consecutive_groups(numbers):
    groups = []
    start = numbers[0]

    for i in range(1, len(numbers)):
        # If there is a gap, end the current group and start a new one
        if numbers[i] != numbers[i - 1] + 1:
            groups.append((start, numbers[i - 1]))  # Add the previous group
            start = numbers[i]  # Start a new group
    
    # Don't forget to add the last group
    groups.append((start, numbers[-1]))
    
    return groups

def plot_single(data, scale = 5):
    last_key = list(data.keys())[-1]
    plt.figure(figsize=(12, 3))
    for i in list(data.keys()):
        x = np.linspace(0, len(data[i]), len(data[i]))
        if i[:11] == 'underlaying':
            plt.plot(x, data[i], linestyle='--', color='gray', alpha=0.5, label=f'{i} signal')
        elif i != 'underlaying' and i != last_key:
            plt.plot(x, data[i], alpha=0.7, label=f'{i} signal')
        elif 'magnitude' in i:
            plt.plot(x, data[i]*scale, alpha=0.7, label=f'{i} signal')


        else:
            plt.plot(x, data[i], color = 'red', label=f'{i} signal')

    plt.title('Coordinate Over Frames', fontsize=12)
    plt.xlabel('# Frame', fontsize=10)
    plt.ylabel('Pupil Coordinate', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_pupil_coordinates(frames, px, py, cleaned_px, cleaned_py, figure_size=(12, 6)):
    plt.figure(figsize=figure_size)
    #plt.subplot(1, 2, 2)
    #plt.scatter(px, py, color='black', alpha=0.2)

    for i in frames.keys():
        # Plot on the second subplot (Cleaned pupil coordinates) 
        plt.subplot(1, 2, 2)
        plot_pupil_coor(cleaned_px[frames[i][0]:frames[i][1]], cleaned_py[frames[i][0]:frames[i][1]], f'{i}')

    # Plot on the first subplot (Extracted pupil coordinates)
    plt.subplot(1, 2, 1)
    plt.scatter(px, py)
    plt.xlim(150, 60)
    plt.ylim(150, 60)


    # Settings for the second subplot
    plt.subplot(1, 2, 2)
    plt.title('Cleaned pupil coordinates', fontsize=12)
    plt.xlabel('x Coordinate', fontsize=8)
    plt.ylabel('y Coordinate', fontsize=8)

    # Settings for the first subplot
    plt.subplot(1, 2, 1)
    plt.title('Extracted pupil coordinates', fontsize=12)
    plt.xlabel('x Coordinate', fontsize=8)
    plt.ylabel('y Coordinate', fontsize=8)

    # Add a grid for better readability across all subplots
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()



