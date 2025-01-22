import matplotlib.pyplot as plt
import numpy as np

# Plotting functions
# Locomotor adaptation
# Baselines


# Learning curves
# Individual animals
def plot_learning_curve_ind_animals(param_sym, current_param, param_labels, animal_list, colors, intervals=None):
    """
    Plots the learning curve for individual animals, each one with the animal's color.
    Overlaps the start and end of split/stimulation intervals, if specified.
    
    Parameters:
    param_sym (numpy.ndarray): A 3D array containing the parameter values for each parameter, animal and trial.
    current_param (int): The index of the current parameter to plot.
    param_labels (dict): Dictionary keys are the parameter names and values are the corresponding labels.
    animal_list (list of str): A list of animal identifiers.
    colors (dict): A dictionary mapping animal identifiers to colors.
    intervals (dict, optional): A dictionary containing 'split' and 'stim' intervals with their respective start and duration (in trials).

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)

    # Plot learning curves for each animal
    for a in range(np.shape(param_sym)[1]):  # Loop on all animals
        plt.plot(np.linspace(1, len(param_sym[current_param, a, :]), len(param_sym[current_param, a, :])), param_sym[current_param, a, :], color=colors[animal_list[a]],
                 label=animal_list[a], linewidth=2)
         
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals:
            add_patch_interval(ax, intervals['split'])
        if 'stim' in intervals:
            add_start_end_interval(ax, intervals['stim'])

    # Add horizontal line at 0    
    plt.hlines(0, 1, len(param_sym[current_param, 0, :]), colors='grey', linestyles='--')
    
    # Add figure labels and legend
    ax.set_xlabel('Trial', fontsize=24)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(list(param_labels.values())[current_param] + ' asymmetry', fontsize=24)
    
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


# Average with SEM for all experiments compared
def plot_learning_curve_avg_compared(param_sym_multi, current_param, labels_dic, included_animals_list_ids, experiment_colors_dict, intervals=None, experiment_names=None, ranges=[False, None]):
    """
    Plots the average learning curve compared across multiple experiments.

    Parameters:
    param_sym_multi (dict): Dictionary containing the parameter symmetry data for multiple paths.
    current_param (int): Index of the current parameter to be plotted.
    labels_dic (dict): Dictionary keys are the parameter names and values are the corresponding labels.
    included_animals_list_ids (list): A list of two lists, the first containing the names/identifiers of the animals to include in the plot, and the second containing the corresponding indices within the all animals list.
    experiment_colors_dict (dict): Dictionary mapping experiment names to their corresponding colors.
    intervals (dict, optional): Dictionary containing 'split' and 'stim' intervals. Defaults to None.
    experiment_names (list, optional): List of experiment names. Defaults to None.
    ranges (list, optional): List containing a boolean and a dictionary for y-axis limits. Defaults to [False, None].

    Returns:
    fig_multi (matplotlib.figure.Figure): The resulting figure object.
    """
    
    fig_multi, ax_multi = plt.subplots(figsize=(7, 10), tight_layout=True)
    min_rect = 0
    max_rect = 0
    path_index = 0

    paths = list(param_sym_multi.keys())
    
    for path in paths:
        ntrial = len(param_sym_multi[path][current_param][0,:])
        plt.plot(np.linspace(1, ntrial, ntrial), np.nanmean(param_sym_multi[path][current_param], axis = 0), 
                 color=experiment_colors_dict[experiment_names[path_index]],  linewidth=2, label=experiment_names[path_index])
        # Add SE of each session
        ax_multi.fill_between(np.linspace(1, ntrial, ntrial), 
                    np.nanmean(param_sym_multi[path][current_param], axis = 0)+np.nanstd(param_sym_multi[path][current_param], axis = 0)/np.sqrt(len(included_animals_list_ids[0])), 
                    np.nanmean(param_sym_multi[path][current_param], axis = 0)-np.nanstd(param_sym_multi[path][current_param], axis = 0)/np.sqrt(len(included_animals_list_ids[0])), 
                    facecolor=experiment_colors_dict[experiment_names[path_index]], alpha=0.5)
        min_rect = min(min_rect,np.nanmin(np.nanmean(param_sym_multi[path][current_param], axis = 0)-np.nanstd(param_sym_multi[path][current_param], axis = 0)))
        max_rect = max(max_rect,np.nanmax(np.nanmean(param_sym_multi[path][current_param], axis=0)+np.nanstd(param_sym_multi[path][current_param], axis = 0)))
        path_index += 1

    param_sym_names = list(labels_dic.keys())
    param_sym_labels = list(labels_dic.values())

    # Define y-axis limits
    if ranges[0] and (ranges[1] is not None):
        ax_multi.set(ylim=ranges[1][param_sym_names[current_param]])
    else:
        ax_multi.set(ylim=[min_rect, max_rect])

    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals:
            add_patch_interval(ax_multi, intervals['split'])
        if 'stim' in intervals:
            add_start_end_interval(ax_multi, intervals['stim'])

    # Add horizontal line at 0
    plt.hlines(0, 1, ntrial, colors='grey', linestyles='--')
    ax_multi.set_xlabel('1-min trial', fontsize=24)
    ax_multi.set_ylabel(param_sym_labels[current_param] +' asymmetry', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax_multi.spines['right'].set_visible(False)
    ax_multi.spines['top'].set_visible(False)

    return fig_multi


# Learning parameters (only comparing multiple experiments?)
# barplot with average and STD or SEM + scatterplot of ind animals in the middle

# average line + scatterplot (no animals color)

# average line + scatterplot (with animals color)




# Utils plotting functions
def add_patch_interval(ax, intervals):
    """
    Adds split or stimulation intervals to a plot, as a patch light blue rectangle.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to add the intervals to.
    intervals (list): A list containing the start and duration (in trials) of the interval to plot as a patch.
    """
    start, duration = intervals
    rectangle = plt.Rectangle((start - 0.5, ax.get_ylim()[0]), duration,
                                    ax.get_ylim()[1] - ax.get_ylim()[0],
                                    fc='lightblue', alpha=0.3)
    ax.add_patch(rectangle)
    
 
def add_start_end_interval(ax, intervals):
    """
    Adds split or stimulation intervals to a plot, as start and end lines.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to add the intervals to.
    intervals (list): A list containing the start and duration (in trials) of the interval to plot as a start and end line.
    """
    start, duration = intervals
    ax.axvline(x=start-0.5, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=start+duration-0.5, color='k', linestyle='-', linewidth=0.5)
