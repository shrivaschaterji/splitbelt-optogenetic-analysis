import matplotlib.pyplot as plt
import numpy as np
import os

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
# barplot of all learning parameters with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_all_learning_params(learning_params, current_param_sym_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None]):
    """
    Plots all learning parameters in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_params : dict
        Dictionary where keys are parameter names and values are 2D arrays (experiments x animals) of learning parameter values.
    current_param_sym_name : str
        Name of the symmetry parameter to be plotted.
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

    

    for i, (lp_name, lp_values) in enumerate(learning_params.items()):
        if not np.isnan(lp_values).all():              # Check for nans
            bars=ax_bar[i].bar([0] + list(range(len(experiment_names) + 1, (len(experiment_names)) * 2)),
                        np.nanmean(lp_values, axis=1),
                        yerr=np.nanstd(lp_values, axis=1) / np.sqrt(len(lp_values)),
                        align='center', alpha=0.5, color=experiment_colors, ecolor='black', capsize=6)
        
        # Add scatterplot of individual animals in the middle
        if scatter_single_animals:
            for a in range(len(lp_values[0])):
                ax_bar[i].plot(list(range(1,len(experiment_names)+1)),np.array(lp_values)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
        # Add plots Statistics
        if len(stat_learning_params)>0:
            ax_bar[i].plot(list(range(len(experiment_names)+1,(len(experiment_names))*2)),[max(max(np.nanmean(lp_values, axis=1)+np.nanstd(lp_values, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')
            print('stat '+lp_name+': '+str(stat_learning_params[lp_name]))

        # Set titles and labels
        if i==0:
            ax_bar[i].set_ylabel(current_param_sym_name + ' asymmetry ')
        ax_bar[i].set_title(lp_name, size=9)

        # Add zero line and set ranges
        ax_bar[i].axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax_bar[i].spines['right'].set_visible(False)
        ax_bar[i].spines['top'].set_visible(False)
        ax_bar[i].set_xticks([])
        if ranges[0] and (ranges[1] is not None):
            ax_bar[i].set(ylim= ranges[1][current_param_sym_name])      
            if 'change' in lp_name:      # Increased ranges for _sym_change parameters
                ax_bar[i].set(ylim= list(30*np.array(ranges[1][current_param_sym_name])))
        
    # Remove empty subplots
    for j in range(len(learning_params), len(ax_bar)):
        fig_bar.delaxes(ax_bar[j])
    fig_bar.suptitle(current_param_sym_name)
    fig_bar.tight_layout()
    fig_bar.legend(bars, experiment_names,
        loc="lower left",   
        borderaxespad=3
        )
    
    return fig_bar

# barplot of one selected learning parameter with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_learning_param(learning_param, current_param_sym_name, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None]):
    """
    Plots a single learning parameter in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym_name : str
        Name of the symmetry parameter to be plotted.
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
        ax_bar.plot(list(range(len(experiment_names)+1,(len(experiment_names))*2)),[max(max(np.nanmean(learning_param, axis=1)+np.nanstd(learning_param, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')

    # Set titles and labels
    ax_bar.set_ylabel(current_param_sym_name + ' asymmetry ', fontsize=24)
    ax_bar.set_title(lp_name)
    ax_bar.set_xticks([0]+list(range(len(experiment_names)+1,(len(experiment_names))*2)))
    ax_bar.set_xticklabels(experiment_names)

    # Add zero line and set ranges
    ax_bar.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    if ranges[0] and (ranges[1] is not None):
        ax_bar.set(ylim= ranges[1][current_param_sym_name])      
        if 'change' in lp_name:      # Increased ranges for _sym_change parameters
            ax_bar.set(ylim= list(30*np.array(ranges[1][current_param_sym_name])))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    
    return fig_bar

# average line + scatterplot (no animals color or with animals color)
def plot_learning_param_scatter(learning_param, current_param_sym_name, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict=None, stat_learning_params=None, ranges=[False, None]):
    """
    Plots a single learning parameter with a scatter plot of individual animals.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym_name : str
        Name of the symmetry parameter to be plotted.
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
            ax_scatter.scatter([x] * len(learning_param[count_exp][:]), learning_param[count_exp][:], s=60, c=experiment_colors[count_exp])             # Plot all animal points in the corresponding experiment color
        # Add avg value
        ax_scatter.plot([x-0.15, x+0.15], [np.nanmean(learning_param[count_exp][:]), np.nanmean(learning_param[count_exp][:])], color=experiment_colors[count_exp], linewidth=4)
        
    # Add plots Statistics
    if len(stat_learning_params)>0:
        ax_scatter.plot(x,[max(max(np.nanmean(learning_param, axis=1)+np.nanstd(learning_param, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')

    # Set titles and labels
    ax_scatter.set_ylabel(current_param_sym_name + ' asymmetry ', fontsize=24)
    ax_scatter.set_title(lp_name)
    ax_scatter.set_xticks(list(range(1,len(experiment_names)+1)))
    ax_scatter.set_xticklabels(experiment_names)

    # Add zero line and set ranges
    ax_scatter.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    if ranges[0] and (ranges[1] is not None):
        ax_scatter.set(ylim= ranges[1][current_param_sym_name])      
        if 'change' in lp_name:      # Increased ranges for _sym_change parameters
            ax_scatter.set(ylim= list(30*np.array(ranges[1][current_param_sym_name])))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)

    return fig_scatter



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


def save_symmetry_plot(figure, path, param_name, plot_name='', bs_bool=False, dpi=128):
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