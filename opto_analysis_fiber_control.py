import matplotlib.pyplot as plt
import numpy as np
import os

#path inputs
path_stim = 'J:\\Opto JAWS Data\\split right fast swing stim\\'
path_save = 'J:\\Thesis\\for figures\\fig fiber control\\'
animals = ['MC19123', 'MC19022']
experiment_stim = path_stim.split('\\')[-2].replace(' ', '_')
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
stim_trials = np.arange(9, 19)
Ntrials = 28
rec_size = 10

param_sym_bs_stim = np.load(
    path_stim + '\\grouped output\\param_sym_bs.npy')

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_stim)
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path_stim)

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
animal_list_plot = [a for count_a, a in enumerate(animal_list) if a in animals]
animal_list_plot_idx = np.array([count_a for count_a, a in enumerate(animal_list) if a in animals])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])
session_list_plot = np.array(session_list)[animal_list_plot_idx]
Ntrials = 28

#summary gait parameters
param_sym = np.zeros((len(param_sym_name), len(animal_list_plot), Ntrials))
param_sym[:] = np.nan
for count_animal, animal in enumerate(animal_list_plot):
    session = int(session_list_plot[count_animal])
    filelist = loco.get_track_files(animal, session)
    # get trial list in filelist
    trial_filelist = np.zeros(len(filelist))
    for count_f, f in enumerate(filelist):
        trial_filelist[count_f] = np.int64(f.split('_')[7][:f.split('_')[7].find('D')])
    trials_idx = np.arange(0, 28)
    trials_ses = np.arange(1, 29)
    trials_idx_corr = np.zeros(len(trial_filelist))
    for count_t, t in enumerate(trial_filelist):
        trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
    trials_idx_corr = np.int64(trials_idx_corr)
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            param_sym[count_p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])

#baseline subtracion of parameters
param_sym_bs_control = np.zeros(np.shape(param_sym))
for p in range(np.shape(param_sym)[0]):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p, a, :stim_trials[0]-1])
        param_sym_bs_control[p, a, :] = param_sym[p, a, :] - bs_mean

min_plot = [-3, -4, -6, -4, -4] #tied
max_plot = [3, 4, 6, 4, 4] #tied
# min_plot = [-5, -7, -7, -2, -3] #split right fast
# max_plot = [5, 4, 11, 9, 8] #split right fast
# min_plot = [-2, -3, -11, -6, -9] #split left fast
# max_plot = [4, 6, 8, 2, 3] #split left fast
for p in range(np.shape(param_sym_name)[0]):
    mean_data_stim = np.nanmean(param_sym_bs_stim[p, :, :], axis=0)
    std_data_stim = np.nanstd(param_sym_bs_stim[p, :, :], axis=0) / np.sqrt(np.shape(param_sym_bs_stim)[1])
    mean_data_control = np.nanmean(param_sym_bs_control[p, :, :], axis=0)
    std_data_control = np.nanstd(param_sym_bs_control[p, :, :], axis=0)
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, min_plot[p]), rec_size, max_plot[p]-min_plot[p], fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), mean_data_stim, linewidth=2, marker='o', color='green')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_stim-std_data_stim, mean_data_stim+std_data_stim, color='green', alpha=0.5)
    plt.plot(np.arange(1, Ntrials+1), mean_data_control, linewidth=2, marker='o', color='dimgray')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_control-std_data_control, mean_data_control+std_data_control, color='dimgray', alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, experiment_stim + '_mean_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
    plt.savefig(os.path.join(path_save, experiment_stim + '_mean_animals_symmetry_' + param_sym_name[p] + '.svg'), dpi=128)