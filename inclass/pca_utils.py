import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import matplotlib.cm as cm  # For color maps


def limb_number_plot(s_pose_x,s_pose_y,n1,n2,c="red",label=None):
  if label is not None:
    if (s_pose_x[n1]>0) and (s_pose_x[n2]>0) and (s_pose_y[n1]>0) and (s_pose_y[n2]>0): 
      plt.plot([s_pose_x[n1],s_pose_x[n2]], [s_pose_y[n1], s_pose_y[n2]],color = c, linestyle="-",label=label)
  else:
    if (s_pose_x[n1]>0) and (s_pose_y[n1]>0):
       plt.plot(s_pose_x[n1], s_pose_y[n1],'*',color = c,label=label)
    if (s_pose_x[n2]>0) and (s_pose_y[n2]>0):
       plt.plot(s_pose_x[n2], s_pose_y[n2],'*',color = c,label=label)
    if (s_pose_x[n1]>0) and (s_pose_x[n2]>0) and (s_pose_y[n1]>0) and (s_pose_y[n2]>0):
      plt.plot([s_pose_x[n1],s_pose_x[n2]], [s_pose_y[n1], s_pose_y[n2]],color = c, linestyle="-")

def plot_single_pose(s_pose,c = "darkgreen",label=None,ds='body_25',c_head = 'red',head = True):
    
    s_pose_x=s_pose[::2]
    s_pose_y=s_pose[1::2]
    #torso/body
    limb_number_plot(s_pose_x,s_pose_y,2,5,c)
    if label is not None:

        limb_number_plot(s_pose_x,s_pose_y,9,12,c,label)
    else:
        limb_number_plot(s_pose_x,s_pose_y,9,12,c)
    limb_number_plot(s_pose_x,s_pose_y,2,9,c)
    limb_number_plot(s_pose_x,s_pose_y,5,12,c)

    #left arm (person facing away)
    limb_number_plot(s_pose_x,s_pose_y,2,3,c)
    limb_number_plot(s_pose_x,s_pose_y,3,4,c)

    #right arm
    limb_number_plot(s_pose_x,s_pose_y,5,6,c)
    limb_number_plot(s_pose_x,s_pose_y,6,7,c)

    #left leg / foot
    limb_number_plot(s_pose_x,s_pose_y,9,10,c)
    limb_number_plot(s_pose_x,s_pose_y,10,11,c)
    limb_number_plot(s_pose_x,s_pose_y,11,22,c)
    #right leg / foot
    limb_number_plot(s_pose_x,s_pose_y,12,13,c)
    limb_number_plot(s_pose_x,s_pose_y,13,14,c)
    limb_number_plot(s_pose_x,s_pose_y,14,19,c)

    # head
    if head:
        limb_number_plot(s_pose_x,s_pose_y,0,15,c)
        limb_number_plot(s_pose_x,s_pose_y,0,16,c)

        limb_number_plot(s_pose_x,s_pose_y,15,17,c)
        limb_number_plot(s_pose_x,s_pose_y,16,18,c)
    return True 

def plot_single_sequence(poses, pose_name='Poses',color='blue'):
    """
    Plots a single sequence of skeleton joints.

    Parameters:
        poses (array-like): Skeleton sequence data, shape (T,D).
        poses_name (string, optional): subtitle of each skeleton body in the sequence. 
        color (string, optional): color of skeleton bodies. 
    """
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(25,5))
    plt.title('Ground truth')

    for i in range(len(poses)):
        plt.subplot(5, 10, i + 1)
        plot_single_pose(poses[i], c=color, head=True)
        plt.ylim(1, 0)
        plt.xlim(-1, 1)
        plt.title(f"{pose_name} {i}")
        plt.axis('off')

    plt.show()

def pca_reconstruction(dataset, k, num_frames=40):
    """
    Performs PCA on the dataset with `k` principal components, projects and reconstructs the data,
    and plots the original and reconstructed pose sequences.
    
    Parameters:
    dataset (numpy.ndarray): The original data matrix with shape (N, T*D).
    k (int): Number of principal components to use.
    num_frames (int): Number of frames to plot in the reconstructed sequence.
    """
    
    # Step 1: Calculate mean and center the data
    mean_vector = np.mean(dataset, axis=0)
    centered_data = dataset - mean_vector
    
    # Step 2: Calculate the covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Step 3: Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(-eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top `k` eigenvectors
    phi = sorted_eigenvectors[:, :k]
    
    cumulative_variance_ratio = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_eigenvalues) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
    plt.axvline(x=k, color='r', linestyle=':', label=f'{k} Components')
    plt.xlabel("Principal Component index")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.legend()
    plt.show()
    
    # Display number of components and variance explained
    explained_variance = cumulative_variance_ratio[k-1] * 100
    print(f"Number of Components (k): {k}")
    print(f"Variance Explained by {k} components: {explained_variance:.2f}%")
    
    # Step 4: Project the data onto the top `k` principal components
    projected_data = phi.T @ centered_data.T
    
    # Step 5: Reconstruct the data back to the original space
    reconstructed_data = (phi @ projected_data).T + mean_vector

    # Step 6: Plot original and reconstructed sequences
    reshaped_original = dataset[:num_frames, :].reshape(num_frames, -1)
    reshaped_reconstructed = reconstructed_data[:num_frames, :].reshape(num_frames, -1)

    # Plot original sequence
    plot_single_sequence(reshaped_original, pose_name='Original', color='blue')

    # Plot reconstructed sequence
    plot_single_sequence(reshaped_reconstructed, pose_name='Reconstructed', color='red')
    



def plot_pca_loadings(
    mixing_params,
    dataset=None,
    bar_width=0.4,
    colors=None,
    component_indices=[1, 2],  # Default to first two PCs
    show_individual=True,
    show_combined=True,
    save_plots=False,
    save_dir='plots',
    file_format='png'
):
    """
    Plots the loadings for specified principal components as bar charts.

    Parameters:
    -----------
    mixing_params : numpy.ndarray
        A 2D array of PCA loadings with shape (num_variables, num_components).

    dataset : pandas.DataFrame or numpy.ndarray, optional
        The original dataset used for PCA. Used to label variables based on dataset columns or indices.
        If None, variables are labeled numerically (Var1, Var2, ...).

    bar_width : float, default=0.4
        Width of the bars in the bar plots.

    colors : list of str, optional
        List of colors for the principal components. If None, colors are auto-generated using a color map.

    component_indices : list of int, default=[1, 2]
        Specific principal components to plot (1-based indexing). Default is the first two PCs.

    show_individual : bool, default=True
        Whether to plot individual principal components as separate bar charts.

    show_combined : bool, default=True
        Whether to plot a combined bar chart with multiple principal components side by side.

    save_plots : bool, default=False
        Whether to save the generated plots to files.

    save_dir : str, default='plots'
        Directory where the plots will be saved if `save_plots` is True.

    file_format : str, default='png'
        Format to save the plots (e.g., 'png', 'jpg', 'pdf').

    Returns:
    --------
    None
        Displays the plots.

    Raises:
    -------
    ValueError
        If `mixing_params` is not a 2D array.
        If `component_indices` contains indices out of range.
        If `colors` provided are insufficient for the components being plotted.

    Usage:
    ------
    import numpy as np
    import pandas as pd

    # Example dataset with 50 variables
    dataset = pd.DataFrame({
        f'Var{i+1}': np.random.rand(100) for i in range(50)
    })

    # Example PCA loadings with shape (50, 9)
    mixing_params = np.random.rand(50, 9)

    # Plotting the first two principal components
    plot_pca_loadings(
        mixing_params=mixing_params,
        dataset=dataset,
        bar_width=0.4,
        colors=['skyblue', 'salmon'],
        component_indices=[1, 2],
        show_individual=True,
        show_combined=True,
        save_plots=False
    )
    """

    # Input Validation
    if mixing_params.ndim != 2:
        raise ValueError("mixing_params must be a 2D array with shape (num_variables, num_components).")

    num_variables, total_num_components = mixing_params.shape

    # Validate and adjust component_indices
    if not all(isinstance(i, int) for i in component_indices):
        raise ValueError("component_indices must be a list of integers representing 1-based PC indices.")
    
    # Convert to 0-based indexing and validate
    component_indices_zero_based = []
    for idx in component_indices:
        if idx < 1 or idx > total_num_components:
            raise ValueError(f"component_indices contain out-of-range index: {idx}. Must be between 1 and {total_num_components}.")
        component_indices_zero_based.append(idx - 1)

    num_components_to_plot = len(component_indices_zero_based)

    # Set default colors if not provided or insufficient
    if colors is None or len(colors) < num_components_to_plot:
        cmap = cm.get_cmap('tab10')  # Choose a color map with enough distinct colors
        colors = [cmap(i % cmap.N) for i in range(num_components_to_plot)]
    else:
        if len(colors) < num_components_to_plot:
            raise ValueError("Not enough colors provided for the number of components being plotted.")

    # Determine variable labels
    if dataset is not None:
        if hasattr(dataset, 'columns'):
            variable_labels = dataset.columns
        else:
            # Assume dataset is a NumPy array
            variable_labels = [f'Var{i+1}' for i in range(num_variables)]
    else:
        variable_labels = [f'Var{i+1}' for i in range(num_variables)]

    # Function to save plots if required
    def maybe_save_plot(filename):
        if save_plots:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f'{filename}.{file_format}'))

    # Plot individual principal components
    if show_individual:
        for i, comp_idx in enumerate(component_indices_zero_based):
            plt.figure(figsize=(10, 6))
            x_positions = np.arange(num_variables)
            plt.bar(x_positions, mixing_params[:, comp_idx], width=bar_width, color=colors[i], label=f'PC{comp_idx + 1}')
            plt.xlabel('Variable')
            plt.ylabel('Loadings')
            plt.title(f'Loadings for Principal Component {comp_idx + 1}')
            plt.xticks(x_positions, variable_labels, rotation=90)
            plt.legend(loc='best')
            plt.tight_layout()
            maybe_save_plot(f'PC{comp_idx + 1}_Loadings')
            plt.show()

    # Plot combined principal components
    if show_combined:
        plt.figure(figsize=(13, 6))
        for i, (comp_idx, color) in enumerate(zip(component_indices_zero_based, colors)):
            adjusted_positions = np.arange(num_variables) - bar_width/2 + i*(bar_width / num_components_to_plot)
            plt.bar(adjusted_positions, mixing_params[:, comp_idx], width=bar_width/num_components_to_plot, 
                    color=color, label=f'PC{comp_idx + 1}')

        plt.xlabel('Variable')
        plt.ylabel('Loadings')
        pcs = ', '.join([str(idx) for idx in component_indices])
        plt.title(f'Loadings for Principal Component(s) {pcs}')
        plt.legend(loc='best')
        plt.xticks(np.arange(num_variables), variable_labels, rotation=90)
        plt.tight_layout()
        maybe_save_plot(f'Combined_PCs_Loadings')
        plt.show()
        
        
        

def plot_pca_pairwise_scatter(
    dataset,
    phi,
    num_components=9,
    figsize=(30, 30),
    marker='.',
    marker_size=10,
    alpha=0.6,
    xlim=(-1.5, 1.5),
    ylim=(-1.5, 1.5),
    grid=True,
    save_plot=False,
    save_dir='plots',
    file_format='png'
):
    """
    Plots a pairwise scatter plot grid for the specified number of principal components.
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        The original dataset with shape (n_samples, n_variables).
    
    phi : numpy.ndarray
        The PCA eigenvectors with shape (n_variables, n_components).
    
    num_components : int, default=9
        Number of principal components to consider for plotting.
    
    figsize : tuple, default=(30, 30)
        Size of the entire figure. Adjusted for better visibility with large grids.
    
    marker : str, default='.'
        Marker style for scatter plots.
    
    marker_size : int, default=10
        Size of the markers in scatter plots.
    
    alpha : float, default=0.6
        Transparency level of the markers.
    
    xlim : tuple, default=(-1.5, 1.5)
        Limits for the x-axis of each subplot.
    
    ylim : tuple, default=(-1.5, 1.5)
        Limits for the y-axis of each subplot.
    
    grid : bool, default=True
        Whether to display grid lines on each subplot.
    
    save_plot : bool, default=False
        Whether to save the plot as a file.
    
    save_dir : str, default='plots'
        Directory where the plot will be saved if `save_plot` is True.
    
    file_format : str, default='png'
        File format for saving the plot (e.g., 'png', 'jpg', 'pdf').
    
    Returns:
    --------
    None
        Displays the plot.
    
    Raises:
    -------
    ValueError
        If `phi` does not have enough principal components.
    
    """
    # Validate input dimensions
    if phi.shape[1] < num_components:
        raise ValueError(f"phi has only {phi.shape[1]} components, but num_components is set to {num_components}.")
    
    if dataset.shape[1] != phi.shape[0]:
        raise ValueError(f"Dataset variables ({dataset.shape[1]}) do not match phi dimensions ({phi.shape[0]}).")
    
    # Calculate the mean vector
    mean_vector = np.mean(dataset, axis=0)
    
    # Subtract the mean from each data point
    centered_data = dataset - mean_vector
    
    # Select the first 'num_components' eigenvectors
    selected_eigenvectors = phi[:, :num_components]  # Shape: (n_variables, num_components)
    
    # Project the data onto the selected principal components
    projected_data = centered_data @ selected_eigenvectors  # Shape: (n_samples, num_components)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_components, num_components, figsize=figsize, sharex=False, sharey=False)
    
    # If dataset has column names, use them; else, use Var1, Var2, etc.
    if hasattr(dataset, 'dtype') and hasattr(dataset, 'shape'):
        # Assume dataset is a NumPy array
        variable_labels = [f'Var{i+1}' for i in range(dataset.shape[1])]
    elif hasattr(dataset, 'columns'):
        # Assume dataset is a pandas DataFrame
        variable_labels = dataset.columns.tolist()
    else:
        # Fallback
        variable_labels = [f'Var{i+1}' for i in range(dataset.shape[1])]
    
    # Iterate through each pair of principal components
    for i in range(num_components):
        for j in range(num_components):
            ax = axes[i, j]
            ax.scatter(projected_data[:, j], projected_data[:, i], marker=marker, s=marker_size, alpha=alpha)
            
            # Set axis limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # Add grid if enabled
            if grid:
                ax.grid(True, linestyle='--', alpha=0.5)
            
            # Label only the leftmost and bottom plots to reduce clutter
            if j == 0:
                ax.set_ylabel(f'PC {i + 1}', fontsize=10)
            else:
                ax.set_ylabel('')
                ax.set_yticks([])
            
            if i == num_components - 1:
                ax.set_xlabel(f'PC {j + 1}', fontsize=10)
            else:
                ax.set_xlabel('')
                ax.set_xticks([])
            
            # Add a title only to the diagonal plots
            if i == j:
                ax.set_title(f'PC {i + 1} vs PC {j + 1}', fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot if required
    if save_plot:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'PCA_Pairwise_Scatter.{file_format}'), dpi=300)
    
    # Display the plot
    plt.show()
        
        


