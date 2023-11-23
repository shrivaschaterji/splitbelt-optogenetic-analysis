import os
import numpy as np
import matplotlib.pyplot as plt

#path inputs
path_loco = 'C:\\Users\\Ana\\Desktop\\Opto Data\\split right fast swing stim\\'
split_side = path_loco.split('\\')[-2].split(' ')[1]
event_stim = path_loco.split('\\')[-2].split(' ')[-2]
if event_stim == 'stance':
    color_cond = 'orange'
if event_stim == 'swing':
    color_cond = 'green'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
animals = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
animals_triggers = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
stim_trials = np.arange(9, 19)

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path_loco)
path_save = path_loco+'grouped output\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])
Ntrials = 28

#summary gait parameters
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'phase_st', 'stance_speed']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
param_label = ['Center of\noscillation (mm)', 'Step length (mm)',
    'Percentage of\ndouble support', 'Spatial motor\noutput (mm)', 'Swing length(mm)']
phase_label = 'Stance phasing\n(degrees)'
stance_speed_label = 'Stance speed (m/s)'
param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
param_sym[:] = np.nan
param_paw = np.zeros((len(param_sym_name), len(animal_list), 4, Ntrials))
param_paw[:] = np.nan
param_phase = np.zeros((4, len(animal_list), Ntrials))
param_phase[:] = np.nan
stance_speed = np.zeros((4, len(animal_list), Ntrials))
stance_speed[:] = np.nan
for count_animal, animal in enumerate(animal_list):
    session = int(session_list[count_animal])
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
            if param == 'phase_st':
                for p in range(4):
                    param_phase[p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[0][p])
            elif param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                for count_paw, paw in enumerate(paws):
                    param_paw[count_p, count_animal, count_paw, trials_idx_corr[count_trial]] = np.nanmean(param_mat[count_paw])

#Plot
#baseline subtracion of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
param_paw_bs = np.zeros(np.shape(param_paw))
for p in range(np.shape(param_sym)[0]-2):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p, a, :stim_trials[0]-1])
        param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
        for count_paw in range(4):
            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_trials[0]-1])
            param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean

if split_side == 'right':
    param_sym_bs_control = np.load('C:\\Users\\Ana\\Desktop\\Opto Data\\split right fast control\\grouped output\\param_sym_bs.npy')
if split_side == 'left':
    param_sym_bs_control = np.load('C:\\Users\\Ana\\Desktop\\Opto Data\\split left fast control\\grouped output\\param_sym_bs.npy')

#plot symmetry baseline subtracted - mean animals
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
    std_data = (np.nanstd(param_sym_bs[p, :, :], axis=0)/np.sqrt(np.shape(param_sym_bs)[1]))
    mean_data_control = np.nanmean(param_sym_bs_control[p, :, :], axis=0)
    std_data_control = (np.nanstd(param_sym_bs_control[p, :, :], axis=0)/np.sqrt(np.shape(param_sym_bs_control)[1]))
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
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '.svg'), dpi=128)
plt.close('all')

#plot individual limbs baseline subtracted - mean animals
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.vstack((np.nanmean(param_paw_bs[p, :, 0, :], axis=0), np.nanmean(param_paw_bs[p, :, 1, :], axis=0),
        np.nanmean(param_paw_bs[p, :, 2, :], axis=0), np.nanmean(param_paw_bs[p, :, 3, :], axis=0)))
    std_data = np.vstack((np.nanstd(param_paw_bs[p, :, 0, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1]),
        np.nanstd(param_paw_bs[p, :, 1, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1]),
        np.nanstd(param_paw_bs[p, :, 2, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1]),
        np.nanstd(param_paw_bs[p, :, 3, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1])))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    for paw in range(4):
        plt.plot(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs[p, :, paw, :], axis=0), linewidth=2, color=paw_colors[paw])
        plt.fill_between(np.arange(1, Ntrials+1),
            np.nanmean(param_paw_bs[p, :, paw, :], axis=0)-(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1])),
            np.nanmean(param_paw_bs[p, :, paw, :], axis=0)+(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(np.shape(param_paw_bs)[1])), color=paw_colors[paw], alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_paws_' + param_sym_name[p] + '.png'), dpi=128)
    plt.savefig(os.path.join(path_save, 'mean_animals_paws_' + param_sym_name[p] + '.svg'), dpi=128)
plt.close('all')

#plot stance phase - group mean
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
mean_data = np.vstack((np.nanmean(param_phase[0, :, :], axis=0), np.nanmean(param_phase[1, :, :], axis=0),
                       np.nanmean(param_phase[2, :, :], axis=0), np.nanmean(param_phase[3, :, :], axis=0)))
std_data = np.vstack((np.nanstd(param_phase[0, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1]),
                      np.nanstd(param_phase[1, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1]),
                      np.nanstd(param_phase[2, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1]),
                      np.nanstd(param_phase[3, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1])))
rectangle = plt.Rectangle((stim_trials[0] - 0.5, np.rad2deg(np.nanmin(mean_data - std_data))), 10,
                          np.rad2deg(np.nanmax(mean_data + std_data) - np.nanmin(mean_data - std_data)), fc='lightblue', alpha=0.3)
plt.gca().add_patch(rectangle)
for paw in range(4):
    plt.plot(np.arange(1, Ntrials + 1), np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0)), linewidth=2,
             color=paw_colors[paw])
    plt.fill_between(np.arange(1, Ntrials + 1),
                     np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0) - (
                                 np.nanstd(param_phase[paw, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1]))),
                     np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0) + (
                                 np.nanstd(param_phase[paw, :, :], axis=0) / np.sqrt(np.shape(param_phase)[1]))),
                     color=paw_colors[paw], alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel(phase_label, fontsize=20)
plt.xticks(fontsize=20)
ax.set_ylim([70, 230])
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase.png'), dpi=128)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase.svg'), dpi=128)
plt.close('all')

#plot stance phase - group mean
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
mean_data = np.vstack((np.nanmean(stance_speed[0, :, :], axis=0), np.nanmean(stance_speed[1, :, :], axis=0),
                       np.nanmean(stance_speed[2, :, :], axis=0), np.nanmean(stance_speed[3, :, :], axis=0)))
std_data = np.vstack((np.nanstd(stance_speed[0, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1]),
                      np.nanstd(stance_speed[1, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1]),
                      np.nanstd(stance_speed[2, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1]),
                      np.nanstd(stance_speed[3, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1])))
rectangle = plt.Rectangle((stim_trials[0] - 0.5, np.nanmin(mean_data - std_data)), 10,
                          np.nanmax(mean_data + std_data) - np.nanmin(mean_data - std_data), fc='lightblue', alpha=0.3)
plt.gca().add_patch(rectangle)
for paw in range(4):
    plt.plot(np.arange(1, Ntrials + 1), np.nanmean(stance_speed[paw, :, :], axis=0), linewidth=2,
             color=paw_colors[paw])
    plt.fill_between(np.arange(1, Ntrials + 1),
                     np.nanmean(stance_speed[paw, :, :], axis=0) - (
                                 np.nanstd(stance_speed[paw, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1])),
                     np.nanmean(stance_speed[paw, :, :], axis=0) + (
                                 np.nanstd(stance_speed[paw, :, :], axis=0) / np.sqrt(np.shape(stance_speed)[1])),
                     color=paw_colors[paw], alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel(stance_speed_label, fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_speed.png'), dpi=128)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_speed.svg'), dpi=128)
plt.close('all')

#plot symmetry baseline subtracted - individual animals
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.min(param_sym_bs[p, :, :].flatten())), 10, np.max(param_sym_bs[p, :, :].flatten())-np.min(param_sym_bs[p, :, :].flatten()), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, len(param_sym_bs[p, a, :]), colors='grey', linestyles='--')
    for a in range(np.shape(param_sym)[1]):
        plt.plot(np.arange(1, Ntrials+1), param_sym_bs[p, a, :], label=animal_list[a], linewidth=2)
    ax.set_xlabel('Trial', fontsize=20)
    ax.legend(frameon=False)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'ind_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
plt.close('all')

#plot stance speed - individual animals
fig, ax = plt.subplots(3, 2, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
ax = ax.ravel()
for count_a in range(len(animals)):
    for p in range(4):
        ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray', linestyle='--')
        ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray', linestyle='--')
        ax[count_a].plot(np.arange(1, Ntrials+1), stance_speed[p, count_a, :], color=paw_colors[p], linewidth=2)
        ax[count_a].axvline(x=8.5, color='black')
        ax[count_a].axvline(x=18.5, color='black')
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='x')
        ax[count_a].tick_params(axis='y')
        ax[count_a].set_title(animals[count_a])
plt.savefig(os.path.join(path_save, 'ind_animals_stance_speed.png'), dpi=128)

#plot stance phase - individual animals
fig, ax = plt.subplots(3, 2, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
ax = ax.ravel()
for count_a in range(len(animals)):
    for p in range(4):
        ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray', linestyle='--')
        ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray', linestyle='--')
        ax[count_a].plot(np.arange(1, Ntrials+1), param_phase[p, count_a, :], color=paw_colors[p], linewidth=2)
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].set_ylim([70, 230])
        ax[count_a].tick_params(axis='x')
        ax[count_a].tick_params(axis='y')
        ax[count_a].set_title(animals[count_a])
plt.savefig(os.path.join(path_save, 'ind_animals_stance_phase.png'), dpi=128)
plt.close('all')

# accuracy_animals = np.zeros((len(animals_triggers), Ntrials))
# accuracy_animals[:] = np.nan
# light_onset_phase_animals = []
# light_offset_phase_animals = []
# stim_nr_animals = []
# stride_nr_animals = []
# fraction_strides_stim_on_animals = np.zeros((len(animals), Ntrials))
# fraction_strides_stim_on_animals[:] = np.nan
# fraction_strides_stim_off_animals = np.zeros((len(animals), Ntrials))
# fraction_strides_stim_off_animals[:] = np.nan
# for count_a, animal in enumerate(animals_triggers):
#     trials = otrack_class.get_trials(animal)
#     # LOAD PROCESSED DATA
#     [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
#     # READ OFFLINE PAW EXCURSIONS
#     [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
#     final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
#                                                  'st-sw-st')
#     # LASER ACCURACY
#     tp_laser = np.zeros(len(trials))
#     fp_laser = np.zeros(len(trials))
#     tn_laser = np.zeros(len(trials))
#     fn_laser = np.zeros(len(trials))
#     precision_laser = np.zeros(len(trials))
#     recall_laser = np.zeros(len(trials))
#     f1_laser = np.zeros(len(trials))
#     for count_t, trial in enumerate(trials):
#         [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, event_stim, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
#         tp_laser[count_t] = tp_trial
#         fp_laser[count_t] = fp_trial
#         tn_laser[count_t] = tn_trial
#         fn_laser[count_t] = fn_trial
#         precision_laser[count_t] = precision_trial
#         recall_laser[count_t] = recall_trial
#         f1_laser[count_t] = f1_trial
#     fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, 0), stim_trials[-1]-stim_trials[0], 1, fc='lightblue',alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     ax.plot(trials, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
#     ax.set_ylim([0, 1])
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_title(animal, fontsize=16)
#     ax.set_ylabel('Accuracy', fontsize=14)
#     plt.savefig(path_save + 'ind_animals_' + animal + '_laser_performance_accuracy.png')
#
#     # get trial list in filelist
#     trial_filelist = np.copy(trials)
#     stim_trials_in = trials[np.where(trials == 9)[0][0]:np.where(trials == 19)[0][0]]
#     #account for missing trials and sessions of different lengths
#     trials_idx = np.arange(0, 28)
#     trials_ses = np.arange(1, 29)
#     trials_idx_corr = np.zeros(len(trial_filelist))
#     for count_t, t in enumerate(trial_filelist):
#         trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     trials_idx_corr = np.int64(trials_idx_corr)
#     stim_trials_in_idx = np.zeros(len(stim_trials_in))
#     for count_t, t in enumerate(stim_trials_in):
#         stim_trials_in_idx[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     stim_trials_in_idx = np.int64(stim_trials_in_idx)
#     #save accuracy
#     accuracy_animals[count_a, trials_idx_corr] = tp_laser+tn_laser
#
#     #LASER ONSET AND OFFSET PHASE
#     light_onset_phase_all = []
#     light_offset_phase_all = []
#     stim_nr_trials = np.zeros(len(stim_trials_in))
#     stride_nr_trials = np.zeros(len(stim_trials_in))
#     for count_t, trial in enumerate(stim_trials_in):
#         [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
#             otrack_class.laser_presentation_phase(trial, trials, event_stim, offtracks_st, offtracks_sw, laser_on,
#                                                   timestamps_session, final_tracks_phase, 0)
#         stim_nr_trials[count_t] = stim_nr
#         stride_nr_trials[count_t] = stride_nr
#         light_onset_phase_all.extend(light_onset_phase)
#         light_offset_phase_all.extend(light_offset_phase)
#     #Plot laser timing for all trials
#     #normalize by number of stimulations 1, 0
#     [fraction_strides_stim_on, fraction_strides_stim_off] = otrack_class.plot_laser_presentation_phase(light_onset_phase_all,
#                     light_offset_phase_all, event_stim, 16, np.sum(stim_nr_trials), np.sum(stride_nr_trials), 1, 0,
#                     path_save, 'ind_animals_' + animal + '_' + event_stim, 1)
#     light_onset_phase_animals.extend(light_onset_phase_all)
#     light_offset_phase_animals.extend(light_offset_phase_all)
#     stim_nr_animals.append(np.sum(stim_nr_trials))
#     stride_nr_animals.append(np.sum(stride_nr_trials))
#     # Get fraction strides stimulated for each trial
#     for count_t, trial in enumerate(trials):
#         [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
#             otrack_class.laser_presentation_phase(trial, trials, event_stim, offtracks_st, offtracks_sw, laser_on,
#                                                   timestamps_session, final_tracks_phase, 0)
#         [fraction_strides_stim_on, fraction_strides_stim_off] = otrack_class.plot_laser_presentation_phase(
#             light_onset_phase, light_offset_phase, event_stim, 16, stim_nr, stride_nr, 1, 0,
#             path_save, 'ind_animals_' + animal + '_' + event_stim, 0)
#         plt.close('all')
#         fraction_strides_stim_on_animals[count_a, trials_idx_corr[count_t]] = fraction_strides_stim_on
#         fraction_strides_stim_off_animals[count_a, trials_idx_corr[count_t]] = fraction_strides_stim_off
#
# # MEAN LASER ACCURACY
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(accuracy_animals, axis=0)
# std_data = (np.nanstd(accuracy_animals, axis=0) / np.sqrt(np.shape(accuracy_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 10, 1, fc='lightblue', zorder=-1, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.set_ylim([0, 1])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Accuracy', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_performance_accuracy.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_performance_accuracy.svg')
#
# # MEAN LASER TIMING
# [fraction_strides_stim_on_all, fraction_strides_stim_off_all] = \
#     otrack_class.plot_laser_presentation_phase(light_onset_phase_animals, light_offset_phase_animals, event_stim,
#     20, np.sum(stim_nr_animals), np.sum(stride_nr_animals), 1, 0, path_save, 'mean_animals_' + event_stim, 1)
#
# # FRACTION OF STIMULATED STRIDES - MEAN
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(fraction_strides_stim_on_animals, axis=0)
# std_data = (np.nanstd(fraction_strides_stim_on_animals, axis=0) / np.sqrt(np.shape(fraction_strides_stim_on_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 10, 1, fc='lightblue', zorder=0, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# ax.set_ylim([0, 1])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Fraction of stimulated\nstrides in the onset', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_onset.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_onset.svg')
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(fraction_strides_stim_off_animals, axis=0)
# std_data = (np.nanstd(fraction_strides_stim_off_animals, axis=0) / np.sqrt(np.shape(fraction_strides_stim_off_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 10, 1, fc='lightblue', zorder=0, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# ax.set_ylim([0, 1])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Fraction of stimulated\nstrides in the offset', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_offset.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_offset.svg')
#
# # FRACTION OF STIMULATED STRIDES - INDIVIDUAL ANIMALS
# fig, ax = plt.subplots(3, 2, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
# ax = ax.ravel()
# for count_a in range(len(animals)):
#     ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray')
#     ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray')
#     ax[count_a].plot(np.arange(1, Ntrials+1), fraction_strides_stim_on_animals[count_a, :], color='black', linewidth=2)
#     ax[count_a].plot(np.arange(1, Ntrials+1), fraction_strides_stim_off_animals[count_a, :], color='black', linewidth=2,
#     linestyle='dashed')
#     ax[count_a].spines['right'].set_visible(False)
#     ax[count_a].spines['top'].set_visible(False)
#     ax[count_a].tick_params(axis='x')
#     ax[count_a].tick_params(axis='y')
#     ax[count_a].set_title(animals[count_a])
# plt.savefig(path_save + 'ind_animals_laser_' + event_stim + '_fraction_stim_strides_onset.png')
# plt.close('all')