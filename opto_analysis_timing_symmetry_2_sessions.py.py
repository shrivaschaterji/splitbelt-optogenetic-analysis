import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#path inputs
path_st = 'C:\\Users\\Ana\\Desktop\\Opto Data\\split right fast stance stim\\'
path_sw = 'C:\\Users\\Ana\\Desktop\\Opto Data\\split right fast swing stim\\'
experiment_type = 'tied'
experiment_st = path_st.split('\\')[-2].replace(' ', '_')
experiment_sw = path_sw.split('\\')[-2].replace(' ', '_')
animals = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
Ntrials = 28
stim_trials = np.arange(9, 19)

param_sym_bs_st = np.load(
    path_st + '\\grouped output\\param_sym_bs.npy')
param_sym_bs_sw = np.load(
    path_sw + '\\grouped output\\param_sym_bs.npy')

min_plot = [-4, -3, -4, -2, -2]
max_plot = [4, 3, 4, 2, 2]
for p in range(np.shape(param_sym_name)[0]):
    mean_data_st = np.nanmean(param_sym_bs_st[p, :, :], axis=0)
    std_data_st = np.nanstd(param_sym_bs_st[p, :, :], axis=0) / np.sqrt(len(animals))
    mean_data_sw = np.nanmean(param_sym_bs_sw[p, :, :], axis=0)
    std_data_sw = np.nanstd(param_sym_bs_sw[p, :, :], axis=0) / np.sqrt(len(animals))
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, min_plot[p]), 10, max_plot[p]-min_plot[p], fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), mean_data_st, linewidth=2, marker='o', color='orange')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_st-std_data_st, mean_data_st+std_data_st, color='orange', alpha=0.5)
    plt.plot(np.arange(1, Ntrials+1), mean_data_sw, linewidth=2, marker='o', color='green')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_sw-std_data_sw, mean_data_sw+std_data_sw, color='green', alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('C:\\Users\\Ana\\Desktop\\Opto Data\\' + experiment_type + '_mean_animals_sym_' + param_sym_name[p] + '.png')

for p in range(3):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for count_animal, animal in enumerate(animals):
        offtracks_phase_stim_st = pd.read_csv(
            os.path.join(path_st, 'grouped output', 'offtracks_phase_stim_' + experiment_st + '_' + animal + '.csv'))
        offtracks_phase_stim_sw = pd.read_csv(
            os.path.join(path_sw, 'grouped output', 'offtracks_phase_stim_' + experiment_sw + '_' + animal + '.csv'))
        ax[0].scatter(offtracks_phase_stim_st['onset'], offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
        ax[1].scatter(offtracks_phase_stim_st['offset'], offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
        ax[0].scatter(offtracks_phase_stim_sw['onset'], offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
        ax[1].scatter(offtracks_phase_stim_sw['offset'], offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
        ax[1].set_xlabel('stride phase (%)', fontsize=20)
        ax[1].set_ylabel(param_sym_label[p] + '\n for stim offset', fontsize=20)
        ax[0].set_ylabel(param_sym_label[p] + '\n for stim onset', fontsize=20)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[1].tick_params(axis='both', labelsize=20)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
    plt.savefig('C:\\Users\\Ana\\Desktop\\Opto Data\\' + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '.png')

