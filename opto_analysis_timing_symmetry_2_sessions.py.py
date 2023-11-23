import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#path inputs
path_st = 'C:\\Users\\Ana\\Desktop\\Opto Data\\tied stance stim\\'
path_sw = 'C:\\Users\\Ana\\Desktop\\Opto Data\\tied swing stim\\'
experiment_type = 'tied'
experiment_st = path_st.split('\\')[-2].replace(' ', '_')
experiment_sw = path_sw.split('\\')[-2].replace(' ', '_')
animals = ['MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
Ntrials = 28

offtracks_phase_stim_animals_st = []
offtracks_phase_stim_animals_sw = []
for animal in animals:
    offtracks_phase_stim_st = pd.read_csv(
        os.path.join(path_st, 'offtracks_phase_stim_' + experiment_st + '_' + animal + '.csv'))
    offtracks_phase_stim_animals_st.append(offtracks_phase_stim_animals_st)
    offtracks_phase_stim_sw = pd.read_csv(
        os.path.join(path_sw, 'offtracks_phase_stim_' + experiment_sw + '_' + animal + '.csv'))
    offtracks_phase_stim_animals_sw.append(offtracks_phase_stim_animals_sw)

#make save param_sym_bs in all the others

for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
    std_data = (np.nanstd(param_sym_bs[p, :, :], axis=0)/np.sqrt(len(animals)))
    mean_data_control = np.nanmean(param_sym_bs_control[p, :, :], axis=0)
    std_data_control = (np.nanstd(param_sym_bs_control[p, :, :], axis=0)/np.sqrt(len(animals)))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data_control+std_data_control)-np.nanmin(mean_data_control-std_data_control), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), mean_data, linewidth=2, marker='o', color=color_cond)
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_control-std_data_control, mean_data_control+std_data_control, color='black', alpha=0.5)
    plt.plot(np.arange(1, Ntrials+1), mean_data_control, linewidth=2, marker='o', color='black')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data-std_data, mean_data+std_data, color=color_cond, alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

for p in range(3):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for count_animal, animal in enumerate(animals):
        ax[0].scatter(offtracks_phase_stim_animals_st[count_animal]['onset'], offtracks_phase_stim_animals_st[count_animal][param_sym_name[p]], s=5, color='orange')
        ax[1].scatter(offtracks_phase_stim_animals_st[count_animal]['offset'], offtracks_phase_stim_animals_st[count_animal][param_sym_name[p]], s=5, color='orange')
        ax[0].scatter(offtracks_phase_stim_animals_sw[count_animal]['onset'], offtracks_phase_stim_animals_sw[count_animal][param_sym_name[p]], s=5, color='green')
        ax[1].scatter(offtracks_phase_stim_animals_sw[count_animal]['offset'], offtracks_phase_stim_animals_sw[count_animal][param_sym_name[p]], s=5, color='green')
        ax[1].set_xlabel('stride phase (%)', fontsize=20)
        ax[1].set_ylabel(param_sym_label[p] + '\n for stim offset', fontsize=20)
        ax[0].set_ylabel(param_sym_label[p] + '\n for stim onset', fontsize=20)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[1].tick_params(axis='both', labelsize=20)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
    plt.savefig('C:\\\Users\\\Ana\\\Desktop\\\Opto Data\\' + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '.png')

