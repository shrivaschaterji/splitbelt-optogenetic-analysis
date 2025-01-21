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
def plot_learning_curve_ind_animals_avg(param_sym_avg, current_param, param_labels, animal_list, included_animals, colors, intervals=None, experiment_name=None, ranges=[False, None]):
    '''
    Plot learning curve for individual animals with average.
    
    Parameters:
    param_sym_avg (numpy.ndarray): A 2D array containing the average parameter values for each animal and trial.
    current_param (int): The index of the current parameter to plot.
    param_labels (dict): Dictionary keys are the parameter names and values are the corresponding labels.
    animal_list (list of str): A list of animal identifiers.
    included_animals (list): A list of two lists, the first containing the names/identifiers of the animals to include in the plot, and the second containing the corresponding indices within the all animals list.
    colors (list): A list of dictionaries, the first mapping animal identifiers to colors and the second mapping the experiments to colors.
    intervals (dict, optional): A dictionary containing 'split' and 'stim' intervals with their respective start and duration (in trials).
    experiment_name (str, optional): The name of the experiment for color mapping.
    ranges (list, optional): A list containing a boolean indicating whether to use uniform y-axis ranges and a dictionary containing y-axis ranges for each parameter.
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    '''

    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)

    # Plot learning curves for each animal
    for a in range(np.shape(param_sym_avg)[0]):
        plt.plot(np.linspace(1, len(param_sym_avg[a, :]), len(param_sym_avg[a, :])), param_sym_avg[a, :], linewidth=1, color=colors[0][included_animals[0][a]], label=animal_list[included_animals[1][a]])
    ax.legend(frameon=False)
    
    # Plot average
    plt.plot(np.linspace(1, len(param_sym_avg[0, :]), len(param_sym_avg[0, :])), np.nanmean(param_sym_avg, axis=0), color=colors[1][experiment_name], linewidth=3)

    param_sym_names = list(param_labels.keys())
    param_sym_labels = list(param_labels.values())

    # Define y-axis limits
    if ranges[0] and (ranges[1] is not None):
        ax.set(ylim=ranges[1][param_sym_names[current_param]])
    else:
        ax.set(ylim=[np.nanmin(param_sym_avg[:, :].flatten()), np.nanmax(param_sym_avg[:, :].flatten())])

    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals:
            add_patch_interval(ax, intervals['split'])
        if 'stim' in intervals:
            add_start_end_interval(ax, intervals['stim'])

    # Add horizontal line at 0
    plt.hlines(0, 1, len(param_sym_avg[0, :]), colors='grey', linestyles='--')
    
    # Add figure labels
    ax.set_xlabel('Trial', fontsize=24)
    ax.set_ylabel(param_sym_labels[current_param] + ' symmetry', fontsize=24)          
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig


# Average with STD or SEM for each experiment


# Average with STD or SEM for all experiments compared



# Learning parameters (only comparing multiple experiments?)
# barplot with average and STD or SEM + scatterplot of ind animals in the middle

# average line + scatterplot (no animals color)

# average line + scatterplot (with animals color)
