import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Plotting functions
# Locomotor adaptation
# Baselines

# STANCE SPEED
def plot_stance_speed(data, animal, paw_colors, intervals=None):
    fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
   
    for p in range(4):
        ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals:
            add_patch_interval(ax, intervals['split'])
        if 'stim' in intervals:
            add_start_end_interval(ax, intervals['stim'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize = 24)
    ax.set_ylabel('Stance speed', fontsize = 24)
    ax.tick_params(axis='x',labelsize = 20)
    ax.tick_params(axis='y',labelsize = 20)
    ax.set_title(animal,fontsize=18)

    return fig

# SYMMETRY LEARNING CURVES
# Individual animals
def plot_learning_curve_ind_animals(param_sym, current_param, labels_dic, animal_list, colors, intervals=None):
    """
    Plots the learning curve for individual animals, each one with the animal's color.
    Overlaps the start and end of split/stimulation intervals, if specified.
    
    Parameters:
    param_sym (numpy.ndarray): A 3D array containing the parameter values for each parameter, animal and trial.
    current_param (int): The index of the current parameter to plot.
    labels_dic (dict): Dictionary keys are the parameter names and values are the corresponding labels.
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

    set_symmetry_plot(ax, list(labels_dic.values())[current_param])

    return fig 


# Individual animals with average
def plot_learning_curve_ind_animals_avg(param_sym_avg, current_param, labels_dic, animal_list, included_animals, colors, experiment_name, intervals=None, ranges=[False, None]):
    '''
    Plot learning curve for individual animals with average.
    
    Parameters:
    param_sym_avg (numpy.ndarray): A 2D array containing the average parameter values for each animal and trial.
    current_param (int): The index of the current parameter to plot.
    labels_dic (dict): Dictionary keys are the parameter names and values are the corresponding labels.
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

    param_sym_names = list(labels_dic.keys())
    param_sym_labels = list(labels_dic.values())

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

    set_symmetry_plot(ax, param_sym_labels[current_param])
   
    return fig



def plot_learning_curve_ind_animals_avg_paw(param_sym_avg, current_param, labels_dic, animal_list, included_animals, colors, paw_colors, paw_name, intervals=None, ranges=[False, None]):
    '''
    Plot learning curve for individual animals with average, using paw color for the average.
    '''

    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)

    # Plot learning curves for each animal
    for a in range(np.shape(param_sym_avg)[0]):
        plt.plot(np.linspace(1, len(param_sym_avg[a, :]), len(param_sym_avg[a, :])), param_sym_avg[a, :], linewidth=1, color=colors[0][included_animals[0][a]], label=animal_list[included_animals[1][a]])
    ax.legend(frameon=False)
    
    # Plot average with paw color
    plt.plot(np.linspace(1, len(param_sym_avg[0, :]), len(param_sym_avg[0, :])), np.nanmean(param_sym_avg, axis=0), color=paw_colors[paw_name], linewidth=3)

    param_sym_names = list(labels_dic.keys())
    param_sym_labels = list(labels_dic.values())

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

    set_paw_plot(ax, param_sym_labels[current_param])
   
    return fig

# Average with SEM for all experiments compared
def plot_learning_curve_avg_compared(param_sym_multi, current_param, labels_dic, included_animals_list_ids, experiment_colors_dict, experiment_names, intervals=None,  ranges=[False, None], is_paw_data=False):
    """
    Plots the average learning curve compared across multiple experiments.

    Parameters
    ----------
    param_sym_multi : dict
        Dictionary containing the parameter symmetry data for multiple paths.
    current_param : int
        Index of the current parameter to be plotted.
    labels_dic : dict
        Dictionary where keys are the parameter names and values are the corresponding labels.
    included_animals_list_ids : list
        A list of two lists, the first containing the names/identifiers of the animals to include in the plot, 
        and the second containing the corresponding indices within the all animals list.
    experiment_colors_dict : dict
        Dictionary mapping experiment names to their corresponding colors.
    experiment_names : list, optional
        List of experiment names. Defaults to None.
    intervals : dict, optional
        Dictionary containing 'split' and 'stim' intervals. Defaults to None.
    ranges : list, optional
        List containing a boolean and a dictionary for y-axis limits. Defaults to [False, None].

    Returns
    -------
    fig_multi : matplotlib.figure.Figure
        The resulting figure object.
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
    if is_paw_data:
        set_paw_plot(ax_multi, param_sym_labels[current_param])
    else:
        set_symmetry_plot(ax_multi, param_sym_labels[current_param])

    return fig_multi


# Learning parameters (only comparing multiple experiments?)
# barplot of all learning parameters with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_all_learning_params(learning_params, current_param_sym, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None], is_paw_data=False):
    """
    Plots all learning parameters in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_params : dict
        Dictionary where keys are parameter names and values are 2D arrays (experiments x animals) of learning parameter values.
    current_param_sym : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_params : dict, optional
        Dictionary where keys are parameter names and values are lists of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for each parameter (default is [False, None]).

    Returns:
    --------
    fig_bar : matplotlib.figure.Figure
        The figure object containing the bar plots.
    """
    
    nrows = len(learning_params)//3 + 1         # We want always 3 columns
    fig_bar, ax_bar = plt.subplots(nrows,3)
    ax_bar = ax_bar.flatten()

    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1]

    for i, (lp_name, lp_values) in enumerate(learning_params.items()):
        if i<3:
            subplot_idx = i
        else:
            subplot_idx = i+1
        if not np.isnan(lp_values).all():              # Check for nans
            bars=ax_bar[subplot_idx].bar([0] + list(range(len(experiment_names) + 1, (len(experiment_names)) * 2)),
                        np.nanmean(lp_values, axis=1),
                        yerr=np.nanstd(lp_values, axis=1) / np.sqrt(len(lp_values)),
                        align='center', alpha=0.5, color=experiment_colors, ecolor='black', capsize=6)
        
        # Add scatterplot of individual animals in the middle
        if scatter_single_animals:
            for a in range(len(lp_values[0])):
                ax_bar[subplot_idx].plot(list(range(1,len(experiment_names)+1)),np.array(lp_values)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
        
        # Add plots Statistics
        if len(stat_learning_params)>0:
            # Check if we're using one-sample tests (where both conditions have statistics)
            if len(stat_learning_params[lp_name]['significant']) == len(experiment_names):
                # For one-sample tests against zero mean distribution
                positions = list(range(1, len(experiment_names) + 1))
            else:
                # When there is a control condition
                positions = list(range(len(experiment_names)+1, (len(experiment_names))*2))
                    
            heights = max(max(np.nanmean(lp_values, axis=1)+np.nanstd(lp_values, axis=1)), 0)
            
            # Get significant indices
            significant_indices = [i for i, sig in enumerate(stat_learning_params[lp_name]['significant']) if sig]
            significant_positions = [positions[i] for i in significant_indices]
            significant_heights = [heights for _ in significant_indices]
            
            if significant_positions:
                ax_bar[subplot_idx].plot(significant_positions, significant_heights, '*', color='black', markersize=15, markeredgewidth=2)
            
            # Add p-values for all comparisons
            for j, (pos, p_val) in enumerate(zip(positions, stat_learning_params[lp_name]['p_values'])):
                # Position text slightly above the bar
                height_text = heights * 1.1
                ax_bar[subplot_idx].text(pos, height_text, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=20)
            
            print('stat '+lp_name+': '+str(stat_learning_params[lp_name]))

        # Set titles and labels
        if i==0:
            if is_paw_data:
                ax_bar[subplot_idx].set_ylabel(current_param_sym_label)
            else:
                ax_bar[subplot_idx].set_ylabel(current_param_sym_label + ' asymmetry ')
        ax_bar[subplot_idx].set_title(lp_name, size=9)

        set_learning_param_plot(ax_bar[subplot_idx], current_param_sym_name, lp_name, ranges=ranges)
        
    # Remove empty subplots
    for ax in ax_bar:
        if not ax.has_data():
            fig_bar.delaxes(ax)
    
    fig_bar.suptitle(current_param_sym_label)
    fig_bar.tight_layout()
    fig_bar.legend(bars, experiment_names,
        loc="lower left",   
        borderaxespad=3
        )
    
    return fig_bar

# barplot of one selected learning parameter with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_learning_param(learning_param, current_param_sym, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None], is_paw_data=False):
    """
    Plots a single learning parameter in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_params : list, optional
        List of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for the parameter (default is [False, None]).

    Returns:
    --------
    fig_bar : matplotlib.figure.Figure
        The figure object containing the bar plot.
    """
    
    fig_bar, ax_bar = plt.subplots(figsize=(7, 10), tight_layout=True)
    
    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1] 

    if not np.isnan(learning_param).all():              # Check for nans
        ax_bar.bar([0] + list(range(len(experiment_names) + 1, (len(experiment_names)) * 2)),
                np.nanmean(learning_param, axis=1),
                yerr=np.nanstd(learning_param, axis=1) / np.sqrt(len(learning_param)),
                align='center', alpha=0.5, color=experiment_colors, ecolor='black', capsize=6)
    # Add scatterplot of individual animals in the middle
    if scatter_single_animals:
        for a in range(len(learning_param[0])):
            ax_bar.plot(list(range(1,len(experiment_names)+1)),np.array(learning_param)[:,a],'-o', markersize=8, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
    
    # Add plots Statistics
    if len(stat_learning_params)>0:
        # Check if we're using one-sample tests
        if len(stat_learning_params[lp_name]['significant']) == len(experiment_names):
            # For one-sample tests against zero mean distribution
            positions = list(range(1, len(experiment_names) + 1))   
         
            bar_positions = [0] + list(range(len(experiment_names)+1, (len(experiment_names))*2))
            positions_mapping = {pos: bar_positions[i % len(bar_positions)] for i, pos in enumerate(positions)}
        else:
            # When there is a control condition
            positions = list(range(len(experiment_names)+1, (len(experiment_names))*2))
            positions_mapping = {pos: pos for pos in positions}

        heights = max(max(np.nanmean(learning_param, axis=1)+np.nanstd(learning_param, axis=1)), 0)
        
        # Get significant indices
        significant_indices = [i for i, sig in enumerate(stat_learning_params[lp_name]['significant']) if sig]
        significant_positions = [positions_mapping[positions[i]] for i in significant_indices]
        significant_heights = [heights for _ in significant_indices]
        
        if significant_positions:
            ax_bar.plot(significant_positions, significant_heights, '*', color='black', markersize=15, markeredgewidth=2)
        
        # Add p-values for all comparisons
        for i, (pos, p_val) in enumerate(zip(positions, stat_learning_params[lp_name]['p_values'])):
            # Map positions to bar positions
            bar_pos = positions_mapping[pos]
            # Position text slightly above the bar
            height_text = heights * 1.1

            if len(experiment_names) >= 3:
                ax_bar.text(bar_pos, height_text, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=20, rotation = 45)
            else:
                ax_bar.text(bar_pos, height_text, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=20)

    # Set titles and labels
    if is_paw_data:
        ax_bar.set_ylabel(current_param_sym_label, fontsize=24)
    else:
        ax_bar.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=24)
    ax_bar.set_title(lp_name, fontsize=24)
    ax_bar.set_xticks([0]+list(range(len(experiment_names)+1,(len(experiment_names))*2)))
    ax_bar.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_bar, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    
    return fig_bar

# average line + scatterplot (no animals color or with animals color)
def plot_learning_param_scatter(learning_param, current_param_sym, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict=None, stat_learning_params=None, ranges=[False, None], is_paw_data=False):
    """
    Plots a single learning parameter with a scatter plot of individual animals.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym_name : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_param : list, optional
        List of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for the parameter (default is [False, None]).

    Returns:
    --------
    fig_scatter : matplotlib.figure.Figure
        The figure object containing the scatter plot.
    """
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 10), tight_layout=True)
    
    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1] 

    x=np.linspace(1,len(experiment_names),len(experiment_names))

    # Add single animal data
    if animal_colors_dict is None:              # Plot all animal lines in lightgray
        for a in range(len(learning_param[0])):
            ax_scatter.plot(x,np.array(learning_param)[:,a],'-', color='lightgray', linewidth=1)
    else:
        for a in range(len(learning_param[0])):
            ax_scatter.plot(x,np.array(learning_param)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
    for count_exp, exp in enumerate(experiment_names):
        if animal_colors_dict is None:
            ax_scatter.scatter([x[count_exp]] * len(learning_param[count_exp][:]), learning_param[count_exp][:], s=60, c=experiment_colors[count_exp])             # Plot all animal points in the corresponding experiment color
        # Add avg value
        ax_scatter.plot([x[count_exp]-0.15, x[count_exp]+0.15], [np.nanmean(learning_param[count_exp][:]), np.nanmean(learning_param[count_exp][:])], color=experiment_colors[count_exp], linewidth=4)

    # Add plots Statistics
    if len(stat_learning_params) > 0:
        # Check if we're using one-sample tests
        if len(stat_learning_params[lp_name]['significant']) == len(experiment_names):
            # For one-sample tests against zero mean distribution
            positions = list(range(1, len(experiment_names) + 1))
        else:
            # For comparisons against control condition
            positions = x[1:]

        heights = max(max(np.nanmean(learning_param, axis=1)+2*np.nanstd(learning_param, axis=1)), 0)

        # Plot asterisks only for significant results
        significant_indices = [i for i, sig in enumerate(stat_learning_params[lp_name]['significant']) if sig]
        significant_positions = [positions[i] for i in significant_indices]
        significant_heights = [heights for _ in significant_indices]
        
        if significant_positions:
            ax_scatter.plot(significant_positions, significant_heights, '*', color='black', markersize=15, markeredgewidth=2)
        
        # Add p-values for all comparisons
        for i, (pos, p_val) in enumerate(zip(positions, stat_learning_params[lp_name]['p_values'])):
            # Position text above the data point
            height = max(max(np.nanmean(learning_param, axis=1)+2.2*np.nanstd(learning_param, axis=1)),0)
            ax_scatter.text(pos, height, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=20)

    # Set titles and labels
    if is_paw_data:
        ax_scatter.set_ylabel(current_param_sym_label, fontsize=24)
    else:
        ax_scatter.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=24)
    ax_scatter.set_title(lp_name, fontsize=24)
    ax_scatter.set_xticks(list(range(1,len(experiment_names)+1)))
    ax_scatter.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_scatter, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)

    return fig_scatter



# UTILS plotting functions
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


# TODO: Add functions to set plot parameters
def set_symmetry_plot(ax, param_name):
    # Add horizontal line at 0
    ax.axhline(y=0, color='grey', linestyle='--')

    # Set labels
    ax.set_xlabel('1-min trial', fontsize=24)
    ax.set_ylabel(param_name + ' asymmetry', fontsize=24)

    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Hide right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def set_paw_plot(ax, param_name):
    # Add horizontal line at 0
    ax.axhline(y=0, color='grey', linestyle='--')

    # Set labels
    ax.set_xlabel('1-min trial', fontsize=24)
    ax.set_ylabel(param_name, fontsize=24)  # No 'asymmetry' added here

    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Hide right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def set_learning_param_plot(ax, param_sym_name, lp_name, ranges=[False, None]):

    # Add zero line and set ranges
    ax.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ranges[0] and (ranges[1] is not None):
        ax.set(ylim= ranges[1][param_sym_name])      
        if 'change' in lp_name:      # Increased ranges for _sym_change parameters
            ax.set(ylim= list(30*np.array(ranges[1][param_sym_name])))
    

def save_plot(figure, path, param_name, plot_name='', bs_bool=False, dpi=128):
    """
    Saves the input plot as a .png file.
    Parameters:
    figure (matplotlib.figure.Figure): The figure object to save.
    path (str): The path to save the figure.
    param_name (str): The name of the parameter represented in the figure.
    plot_name (str, optional): The name of the plot. Defaults to ''.
    bs_bool (bool, optional): A boolean indicating if the plot is for the baseline or non baseline subtracted parameter. Defaults to False.
    dpi (int, optional): The resolution of the saved figure. Defaults to 128.

    """
    if not os.path.exists(path):
        os.mkdir(path)
    if bs_bool:
        figure.savefig(path + param_name + '_sym_bs_'+ plot_name, dpi=dpi)
    else:
        figure.savefig(path + param_name + '_sym_non_bs_'+ plot_name, dpi=dpi)


def save_plot_with_paw(figure, path, param_name, paw_name, plot_name='', bs_bool=False, dpi=128):
   
    if not os.path.exists(path):
        os.mkdir(path)
    if bs_bool:
        figure.savefig(path + param_name + paw_name + '_sym_bs_' + plot_name, dpi=dpi)
    else:
        figure.savefig(path + param_name + paw_name + '_sym_non_bs_' + plot_name, dpi=dpi)


def plot_statistical_summary_tables(all_stat_dicts, experiment_names, param_names, param_labels, path_to_save=None):
    """
    Creates summary tables for stance and swing conditions, showing p-values and significance 
    for all parameters and their learning metrics in one consolidated view.
    
    Parameters:
    -----------
    all_stat_dicts : dict
        Dictionary containing statistical results for all parameters
        Structure: {param_name: stat_dict, ...}
    experiment_names : list
        List of names of the experiments (e.g., ["stance stim", "swing stim"])
    param_names : list
        List of parameter names (e.g., ["coo", "step_length", ...])
    param_labels : list
        List of parameter labels for display
    path_to_save : str, optional
        Path to save the figure. If None, the figure is just displayed
    
    Returns:
    --------
    fig_list : list
        List of figure objects created (one for each condition)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create a figure for each condition (stance and swing)
    fig_list = []
    
    # Learning parameters to display
    learning_params = ['initial error', 'adaptation', 'after-effect']
    
    for condition_idx, condition_name in enumerate(experiment_names):
        # Calculate rows needed: each parameter has multiple learning parameters
        row_count = len(param_names) * len(learning_params) + len(param_names)  # +len(param_names) for headers
        
        # Create a figure with appropriate size
        fig_height = max(8, row_count * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        # Set up the table data
        table_data = []
        
        # Add headers for columns
        headers = ['Parameter', 'Learning Metric', 'Experimental p-value', 'Sig.']
        
        # Check if we have validation data
        has_validation = False
        for param_name in param_names:
            if param_name in all_stat_dicts and 'baseline_validation' in all_stat_dicts[param_name]:
                has_validation = True
                break
        
        if has_validation:
            headers.extend(['Baseline Avg p-value', 'Sig.'])
            # Check if we have all-trials validation
            for param_name in param_names:
                if (param_name in all_stat_dicts and 
                    'baseline_validation' in all_stat_dicts[param_name] and 
                    'all_trials_p_values' in all_stat_dicts[param_name]['baseline_validation']):
                    headers.extend(['Baseline All p-value', 'Sig.'])
                    break
        
        # Add data rows for each parameter and learning metric
        for param_idx, param_name in enumerate(param_names):
            param_label = param_labels[param_idx]
            
            # Skip parameters that aren't in the stats dict
            if param_name not in all_stat_dicts:
                continue
                
            stat_dict = all_stat_dicts[param_name]
            
            # Add parameter header row
            param_header_row = [param_label, ""]  # Parameter name in first column, empty in second
            empty_cells = [""] * (len(headers) - 2)
            param_header_row.extend(empty_cells)
            table_data.append(param_header_row)
            
            # Add rows for each learning parameter
            for lp_name in learning_params:
                if lp_name in stat_dict:
                    row = ["", lp_name.capitalize()]  # Empty in first column, learning param in second
                    
                    # Add experimental test results for this condition
                    if condition_idx < len(stat_dict[lp_name]['p_values']):
                        p_value = stat_dict[lp_name]['p_values'][condition_idx]
                        significant = stat_dict[lp_name]['significant'][condition_idx]
                        row.extend([f"{p_value:.4f}", "✓" if significant else "×"])
                    else:
                        row.extend(["N/A", "N/A"])
                    
                    # Add baseline validation if available
                    if has_validation and 'baseline_validation' in stat_dict:
                        if condition_idx < len(stat_dict['baseline_validation']['baseline_p_values']):
                            baseline_p = stat_dict['baseline_validation']['baseline_p_values'][condition_idx]
                            baseline_sig = stat_dict['baseline_validation']['baseline_significant'][condition_idx]
                            row.extend([f"{baseline_p:.4f}", "✓" if baseline_sig else "×"])
                        else:
                            row.extend(["N/A", "N/A"])
                        
                        # Add all-trials validation if available
                        if 'all_trials_p_values' in stat_dict['baseline_validation'] and len(headers) > 6:
                            if condition_idx < len(stat_dict['baseline_validation']['all_trials_p_values']):
                                all_trials_p = stat_dict['baseline_validation']['all_trials_p_values'][condition_idx]
                                all_trials_sig = stat_dict['baseline_validation']['all_trials_significant'][condition_idx]
                                row.extend([f"{all_trials_p:.4f}", "✓" if all_trials_sig else "×"])
                            else:
                                row.extend(["N/A", "N/A"])
                    
                    table_data.append(row)
        
        # Create the table
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['lightgray'] * len(headers)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Highlight parameter rows and color cells based on significance
        for i in range(len(table_data)):
            row = table_data[i]
            # Check if this is a parameter header row (has content in first column, empty in second)
            if row[0] and not row[1]:
                for j in range(len(headers)):
                    cell = table[(i+1, j)]
                    cell.set_facecolor('#E6F2FF')  # Light blue for parameter headers
                    if j == 0:  # Make parameter name bold
                        cell.set_text_props(weight='bold')
            else:
                # Check p-value columns (2, 4, 6) for coloring based on significance
                p_value_cols = list(range(2, len(headers), 2))
                for j in p_value_cols:
                    if j < len(row) and row[j] != "" and row[j] != "N/A":
                        try:
                            p_value = float(row[j])
                            cell = table[(i+1, j)]
                            if p_value < 0.05:
                                cell.set_facecolor('#FFCCCC')  # Light red for significant
                            else:
                                cell.set_facecolor('white')
                        except (ValueError, KeyError, IndexError):
                            pass
        
        # Turn off the axis
        ax.axis('off')
        
        # Add title
        plt.title(f"Statistical Summary for {condition_name}", fontsize=16)
        plt.tight_layout()
        
        # Save the figure if path is provided
        if path_to_save:
            condition_path = path_to_save.replace('.png', f'_summary_{condition_name.replace(" ", "_")}.png')
            plt.savefig(condition_path, dpi=300, bbox_inches='tight')
            print(f"Summary table saved to {condition_path}")
        
        fig_list.append(fig)
    
    return fig_list




def plot_stride_by_stride_binned(binned_data, animal_list, param_name, experiment_names, 
                               experiment_colors_dict, intervals=None, data_type='asymmetry', 
                               show_individual_animals=False, path_to_save=None,
                               strides_per_trial=100, strides_bin_size=1):
    """
    Creates a stride-by-stride plot using binned data.
    Only shows the average across animals with standard error of mean.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Process data for all animals
    all_values = []
    all_bin_indices = []
    animal_values = {animal: [] for animal in animal_list}
    animal_bin_indices = {animal: [] for animal in animal_list}
    
    for path in binned_data:
        for animal in animal_list:
            if animal in binned_data[path] and param_name in binned_data[path][animal]:
                if data_type in binned_data[path][animal][param_name]:
                    values = binned_data[path][animal][param_name][data_type]['values']
                    # Only if we have values
                    if values:
                        # Track bin indices continuously
                        bin_indices = list(range(len(values)))
                        
                        all_values.append(values)
                        all_bin_indices.append(bin_indices)
                        animal_values[animal] = values
                        animal_bin_indices[animal] = bin_indices
    
    # Calculate and plot average across animals
    if all_values:
        # Find max number of bins
        max_bins = max([len(vals) for vals in all_values if vals], default=0)
        
        if max_bins > 0:
            # Create common bins array
            common_bins = np.arange(max_bins)
            
            # Interpolate values to common bins
            aligned_values = []
            for animal in animal_list:
                if animal_values[animal]:
                    # Create animal's data as array with NaNs for missing bins
                    animal_data = np.full(max_bins, np.nan)
                    valid_bins = min(len(animal_values[animal]), max_bins)
                    animal_data[:valid_bins] = animal_values[animal][:valid_bins]
                    aligned_values.append(animal_data)
            
            if aligned_values:
                # Calculate average and SEM across animals
                aligned_values = np.array(aligned_values)
                avg_values = np.nanmean(aligned_values, axis=0)
                
                # Calculate SEM safely
                non_nan_count = np.sum(~np.isnan(aligned_values), axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sem_values = np.nanstd(aligned_values, axis=0) / np.sqrt(non_nan_count)
                sem_values = np.where(np.isnan(sem_values), 0, sem_values)
                
                # Plot average with error band
                ax.plot(common_bins, avg_values, '-', color='black', linewidth=2, label='Average')
                ax.fill_between(common_bins, avg_values - sem_values, avg_values + sem_values, 
                              color='black', alpha=0.2)
    
    # Add shaded regions for intervals
    if intervals:
        # Convert trial numbers to bin indices using strides_per_trial and strides_bin_size
        for interval_name, [start_trial, duration_trials] in intervals.items():
            # Convert trial numbers to bin indices
            start_bin = (start_trial - 1) * strides_per_trial // strides_bin_size  # -1 because trial numbering starts at 1
            end_bin = (start_trial + duration_trials - 1) * strides_per_trial // strides_bin_size
            
            # Ensure we don't exceed the plot boundaries
            start_bin = max(0, start_bin)
            end_bin = min(max_bins if max_bins > 0 else 1000, end_bin)
            
            # Determine color based on interval name
            if interval_name == 'stim':
                color = experiment_colors_dict.get(experiment_names[0], 'grey')
                label = 'Stimulation'
            elif interval_name == 'split':
                color = 'lightgrey'
                label = 'Split Belt'
            
            ax.axvspan(start_bin, end_bin, alpha=0.3, color=color, label=label)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    if data_type == 'asymmetry':
        ax.set_ylabel(f"{param_name.replace('_', ' ').title()} Asymmetry", fontsize=14)
    else:
        ax.set_ylabel(f"{param_name.replace('_', ' ').title()} ({data_type})", fontsize=14)
    
    ax.set_xlabel("Stride Bins", fontsize=14)
    ax.set_title(f"Stride-by-stride {param_name.replace('_', ' ').title()} - {data_type}", fontsize=16)
    
    # Auto-scale y-axis to better show fluctuations
    if aligned_values is not None and len(aligned_values) > 0:
        # Calculate the mean and standard deviation of the mean trace
        mean_trace = avg_values
        std_trace = np.nanstd(mean_trace)
        mean_value = np.nanmean(mean_trace)
        
        # Set y-limits to be mean ± 3*std (or slightly more)
        y_range = max(std_trace * 4, 1.0)  # At least 1.0 units tall
        ax.set_ylim(mean_value - y_range, mean_value + y_range)
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    ax.legend(fontsize=12, frameon=False)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if path_to_save:
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    
    return fig