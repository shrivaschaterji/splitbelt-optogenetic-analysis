import matplotlib.pyplot as plt
import numpy as np

# Plotting functions
# Locomotor adaptation
# Baselines


# Learning curves
# Individual animals
def plot_learning_curve_ind_animals(param_sym, current_param, param_labels, animals, colors, split_intervals=None, stim_intervals=None):
    """
    Plots the learning curve for individual animals, each one with the animal's color.
    Overlaps the start and end of split/stimulation intervals, if specified.
    
    Parameters:
    param_sym (numpy.ndarray): A 3D array containing the parameter values for each parameter, animal and trial.
    current_param (int): The index of the current parameter to plot.
    param_labels (list of str): A list of labels for each parameter.
    animals (list of str): A list of animal identifiers.
    colors (dict): A dictionary mapping animal identifiers to colors.
    split_intervals (list, optional): A list containing the start and duration (in trials) of the split interval.
    stim_intervals (list, optional): A list containing the start and duration (in trials) of the stimulation interval.
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    
    if split_intervals is not None:
        split_start, split_duration = split_intervals
        rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_sym[current_param, :, :].flatten())), split_duration,
                                    np.nanmax(param_sym[current_param, :, :].flatten()) - np.nanmin(param_sym[current_param, :, :].flatten()),
                                    fc='lightblue', alpha=0.3)
        plt.gca().add_patch(rectangle)
    
    if stim_intervals is not None:
        stim_start, stim_duration = stim_intervals
        ax.axvline(x=stim_start-0.5, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=stim_start+stim_duration-0.5, color='k', linestyle='-', linewidth=0.5)
    
    plt.hlines(0, 1, len(param_sym[current_param, 0, :]), colors='grey', linestyles='--')
    
    for a in range(np.shape(param_sym)[1]):  # Loop on all animals
        plt.plot(np.linspace(1, len(param_sym[current_param, a, :]), len(param_sym[current_param, a, :])), param_sym[current_param, a, :], color=colors[animals[a]],
                 label=animals[a], linewidth=2)
    
    ax.set_xlabel('Trial', fontsize=24)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(param_labels[current_param] + ' asymmetry', fontsize=24)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig 




# Individual animals with average


# Average with STD or SEM for each experiment


# Average with STD or SEM for all experiments compared



# Learning parameters (only comparing multiple experiments?)
# barplot with average and STD or SEM + scatterplot of ind animals in the middle

# average line + scatterplot (no animals color)

# average line + scatterplot (with animals color)
