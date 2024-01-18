import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

#path inputs
path_st = 'J:\\Opto JAWS Data\\tied stance stim\\'
path_sw = 'J:\\Opto JAWS Data\\tied swing stim\\'
experiment_type = 'tied'
save_path = 'J:\\Thesis\\for figures\\fig tied opto\\'
experiment_st = path_st.split('\\')[-2].replace(' ', '_')
experiment_sw = path_sw.split('\\')[-2].replace(' ', '_')
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

param_sym_bs_st_withnan = np.load(
    path_st + '\\grouped output\\param_sym_bs.npy')
param_sym_bs_sw_withnan = np.load(
    path_sw + '\\grouped output\\param_sym_bs.npy')
param_paw_bs_st_withnan = np.load(
    path_st + '\\grouped output\\param_paw_bs.npy')
param_paw_bs_sw_withnan = np.load(
    path_sw + '\\grouped output\\param_paw_bs.npy')
param_phase_st_withnan = np.load(
    path_st + '\\grouped output\\param_phase.npy')
param_phase_sw_withnan = np.load(
    path_sw + '\\grouped output\\param_phase.npy')
if experiment_type == 'tied':
    param_sym_bs_st = np.take(param_sym_bs_st_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=2)
    param_sym_bs_sw = np.take(param_sym_bs_sw_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=2)
    param_paw_bs_st = np.take(param_paw_bs_st_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=3)
    param_paw_bs_sw = np.take(param_paw_bs_sw_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=3)
    param_phase_st = np.take(param_phase_st_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=2)
    param_phase_sw = np.take(param_phase_sw_withnan,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25], axis=2)
    stim_trials = np.arange(9, 17)
    animals_st = ['MC16851', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
    animals_sw = ['MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
    #when all animals are included
    # animals_st = ['MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
    # animals_sw = ['MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
    Nanimals = len(animals_st) #because of bad trackig took one animal in each session
    Ntrials = 24
    rec_size = 8
else:
    param_sym_bs_st = param_sym_bs_st_withnan
    param_sym_bs_sw = param_sym_bs_sw_withnan
    param_paw_bs_st = param_paw_bs_st_withnan
    param_paw_bs_sw = param_paw_bs_sw_withnan
    param_phase_st = param_phase_st_withnan
    param_phase_sw = param_phase_sw_withnan
    animals_st = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
    animals_sw = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
    stim_trials = np.arange(9, 19)
    Nanimals = len(animals_st)
    Ntrials = 28
    rec_size = 10

# param_sym_bs_control = np.load('J:\\Opto JAWS Data\\split right fast control\\grouped output\\param_sym_bs.npy')

min_plot = [-3, -4, -6, -4, -4] #tied
max_plot = [3, 4, 6, 4, 4] #tied
# min_plot = [-5, -7, -7, -2, -3] #split right fast
# max_plot = [5, 4, 11, 9, 8] #split right fast
# min_plot = [-2, -3, -11, -6, -9] #split left fast
# max_plot = [4, 6, 8, 2, 3] #split left fast
for p in range(np.shape(param_sym_name)[0]):
    mean_data_st = np.nanmean(param_sym_bs_st[p, :, :], axis=0)
    std_data_st = np.nanstd(param_sym_bs_st[p, :, :], axis=0) / np.sqrt(Nanimals)
    mean_data_sw = np.nanmean(param_sym_bs_sw[p, :, :], axis=0)
    std_data_sw = np.nanstd(param_sym_bs_sw[p, :, :], axis=0) / np.sqrt(Nanimals)
    # mean_data_control = np.nanmean(param_sym_bs_control[p, :, :], axis=0)
    # std_data_control = np.nanstd(param_sym_bs_control[p, :, :], axis=0)
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, min_plot[p]), rec_size, max_plot[p]-min_plot[p], fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    # plt.plot(np.arange(1, Ntrials+1), mean_data_control, linewidth=2, marker='o', color='black')
    # plt.fill_between(np.arange(1, Ntrials+1), mean_data_control-std_data_control, mean_data_control+std_data_control, color='black', alpha=0.5)
    plt.plot(np.arange(1, Ntrials+1), mean_data_st, linewidth=2, marker='o', color='orange')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_st-std_data_st, mean_data_st+std_data_st, color='orange', alpha=0.5)
    plt.plot(np.arange(1, Ntrials+1), mean_data_sw, linewidth=2, marker='o', color='green')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data_sw-std_data_sw, mean_data_sw+std_data_sw, color='green', alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    # if p == 2:
    #     plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, experiment_type + '_mean_animals_sym_' + param_sym_name[p] + '.png'))
    plt.savefig(os.path.join(save_path, experiment_type + '_mean_animals_sym_' + param_sym_name[p] + '.svg'))
plt.close('all')

# Quantifications
# Create dataframe for learning quantifications
group_id = []
ae_amp = []
delta_split = []
param = []
for count_p, p in enumerate(param_sym_name):
    group_st_len = np.shape(param_sym_bs_st)[1]
    group_id.extend(np.repeat(0, group_st_len))
    ae_st_data = np.vstack((param_sym_bs_st[count_p, :, stim_trials[-1]], param_sym_bs_st[count_p, :, stim_trials[-1]+1]))
    ae_amp.extend(np.nanmean(ae_st_data, axis=0))
    #ae_amp.extend(param_sym_bs_st[count_p, :, stim_trials[-1]])
    delta_split.extend(param_sym_bs_st[count_p, :, stim_trials[-1]-1]-param_sym_bs_st[count_p, :, stim_trials[0]-1])
    param.extend(np.repeat(p, group_st_len))
    group_sw_len = np.shape(param_sym_bs_sw)[1]
    group_id.extend(np.repeat(1, group_sw_len))
    ae_sw_data = np.vstack((param_sym_bs_sw[count_p, :, stim_trials[-1]], param_sym_bs_sw[count_p, :, stim_trials[-1]+1]))
    ae_amp.extend(np.nanmean(ae_sw_data, axis=0))
    #ae_amp.extend(param_sym_bs_sw[count_p, :, stim_trials[-1]])
    delta_split.extend(param_sym_bs_sw[count_p, :, stim_trials[-1]-1]-param_sym_bs_sw[count_p, :, stim_trials[0]-1])
    param.extend(np.repeat(p, group_sw_len))
split_quant_df = pd.DataFrame({'param': param, 'group': group_id, 'after-effect': ae_amp, 'delta-split': delta_split})
param_sym_label_ae = ['Center of oscillation\nafter-effect symmetry (mm)', 'Step length\nafter-effect symmetry(mm)', 'Percentage of double support\nafter-effect symmetry', 'Center of oscillation\n stance after-effect symmetry (mm)',
        'Swing length\nafter-effect symmetry (mm)']
for p in range(np.shape(param_sym_name)[0]):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    sns.barplot(x='group', y='after-effect', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
               alpha=0.5, palette={0: 'orange', 1: 'green'}, ci=None)
    sns.stripplot(x='group', y='after-effect', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
                  dodge=True, s=10, ax=ax, palette={0: 'orange', 1: 'green'}, edgecolor='None')
    ax.set_xticklabels(['Stance-like\nstimulation', 'Swing-like\nstimulation'])
    ax.set_xlabel('')
    ax.set_ylabel(param_sym_label_ae[p], fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(save_path + param_sym_name[p] + 'after_effect_quantification', dpi=256)
    plt.savefig(save_path + param_sym_name[p] + 'after_effect_quantification.svg', dpi=256)
    ###### STATS #######
    # Mann-Whitney on param means
    print(param_sym_name[p])
    data_stats = split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]]
    stats_mannwhitney_ae = scipy.stats.mannwhitneyu(data_stats.loc[data_stats['group']==0, 'after-effect'], data_stats.loc[data_stats['group']==1, 'after-effect'], method='exact')
    print(stats_mannwhitney_ae)
# param_sym_label_delta = ['Change over split of\ncenter of oscillation symmetry (mm)', 'Change over split of\nstep length symmetry (mm)', 'Change over split of\npercentage of double support symmetry', 'Change over split of\ncenter of oscillation stance symmetry (mm)',
#         'Change over split of\nswing length symmetry (mm)']
param_sym_label_delta = ['Change over stim. of\ncenter of oscillation symmetry (mm)', 'Change over stim. of\nstep length symmetry (mm)', 'Change over stim. of\npercentage of double support symmetry', 'Change over stim. of\ncenter of oscillation stance symmetry (mm)',
        'Change over stim. of\nswing length symmetry (mm)']
for p in range(np.shape(param_sym_name)[0]):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    sns.barplot(x='group', y='delta-split', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
               alpha=0.5, palette={0: 'orange', 1: 'green'}, ci=None)
    sns.stripplot(x='group', y='delta-split', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
                  dodge=True, s=10, ax=ax, palette={0: 'orange', 1: 'green'}, edgecolor='None')
    ax.set_xticklabels(['Stance-like\nstimulation', 'Swing-like\nstimulation'])
    ax.set_xlabel('')
    ax.set_ylabel(param_sym_label_delta[p], fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(save_path + param_sym_name[p] + 'delta_stim_quantification', dpi=256)
    plt.savefig(save_path + param_sym_name[p] + 'delta_stim_quantification.svg', dpi=256)
    ###### STATS #######
    # Mann-Whitney on param means
    print(param_sym_name[p])
    data_stats = split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]]
    stats_mannwhitney_ds = scipy.stats.mannwhitneyu(data_stats.loc[data_stats['group']==0, 'delta-split'], data_stats.loc[data_stats['group']==1, 'delta-split'], method='exact')
    print(stats_mannwhitney_ds)

# # Individual limbs - st
# xaxis_trials = np.array([stim_trials[0]-1, stim_trials[0], stim_trials[-1], stim_trials[-1]+1])
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     for paw in range(4):
#         param_paw_bs_st_mean = np.nanmean(param_paw_bs_st[p, :, paw, xaxis_trials-1], axis=1)
#         param_paw_bs_st_std = np.nanstd(param_paw_bs_st[p, :, paw, xaxis_trials-1], axis=1)/np.sqrt(np.shape(param_paw_bs_st)[1])
#         ax.errorbar(np.arange(0, len(xaxis_trials)*2, 2)+(paw*0.1), param_paw_bs_st_mean, param_paw_bs_st_std, color=paw_colors[paw], linewidth=1)
#         ax.scatter(np.arange(0, len(xaxis_trials) * 2, 2) + (paw * 0.1), param_paw_bs_st_mean, s=70, color=paw_colors[paw])
#     ax.set_xticks(np.arange(0, len(xaxis_trials)*2, 2)+0.2)
#     ax.set_xticklabels(['baseline', 'first stim.', 'last stim.', 'post-stim.'], rotation=45)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_st.png'), dpi=128)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_st.svg'), dpi=128)
# plt.close('all')
#
# # Individual limbs - sw
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     for paw in range(4):
#         param_paw_bs_sw_mean = np.nanmean(param_paw_bs_sw[p, :, paw, xaxis_trials-1], axis=1)
#         param_paw_bs_sw_std = np.nanstd(param_paw_bs_sw[p, :, paw, xaxis_trials-1], axis=1)/np.sqrt(np.shape(param_paw_bs_sw)[1])
#         ax.errorbar(np.arange(0, len(xaxis_trials)*2, 2)+(paw*0.1), param_paw_bs_sw_mean, param_paw_bs_sw_std, color=paw_colors[paw], linewidth=1)
#         ax.scatter(np.arange(0, len(xaxis_trials) * 2, 2) + (paw * 0.1), param_paw_bs_sw_mean, s=70, color=paw_colors[paw])
#     ax.set_xticks(np.arange(0, len(xaxis_trials)*2, 2)+0.2)
#     ax.set_xticklabels(['baseline', 'first stim.', 'last stim.', 'post-stim.'], rotation=45)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_sw.png'), dpi=128)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_sw.svg'), dpi=128)
# plt.close('all')
#
# Stance phase - st
fig = plt.figure(figsize=(10, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='polar')
for paw in range(4):
    data_mean = scipy.stats.circmean(param_phase_st[paw, :, :], axis=0, nan_policy='omit')
    ax.scatter(data_mean[~np.isnan(data_mean)], np.arange(1, Ntrials+1), c=paw_colors[paw], s=30)
ax.set_yticks([8.5, 8.5+rec_size])
ax.set_yticklabels(['', ''])
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_st.png'), dpi=256)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_st.svg'), dpi=256)

# Stance phase - sw
fig = plt.figure(figsize=(10, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='polar')
for paw in range(4):
    data_mean = scipy.stats.circmean(param_phase_sw[paw, :, :], axis=0)
    ax.scatter(data_mean[~np.isnan(data_mean)], np.arange(1, Ntrials+1), c=paw_colors[paw], s=30)
ax.set_yticks([8.5, 8.5+rec_size])
ax.set_yticklabels(['', ''])
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_sw.png'), dpi=256)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_sw.svg'), dpi=256)
#
# # Timing - single strides
# for p in range(3):
#     fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True, sharex=True, sharey=True)
#     for count_animal, animal in enumerate(animals_st):
#         offtracks_phase_stim_st = pd.read_csv(
#             os.path.join(path_st, 'grouped output', 'offtracks_phase_stim_' + experiment_st + '_' + animal + '.csv'))
#         ax.scatter(offtracks_phase_stim_st['onset']*100, offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
#     for count_animal, animal in enumerate(animals_sw):
#         offtracks_phase_stim_sw = pd.read_csv(
#             os.path.join(path_sw, 'grouped output', 'offtracks_phase_stim_' + experiment_sw + '_' + animal + '.csv'))
#         ax.scatter(offtracks_phase_stim_sw['onset']*100, offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
#     ax.set_ylabel(param_sym_label[p] + '\n for stim onset', fontsize=20)
#     ax.tick_params(axis='both', labelsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_onset.png')
#     plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_onset.svg')
#     fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True, sharex=True, sharey=True)
#     for count_animal, animal in enumerate(animals_st):
#         ax.scatter(offtracks_phase_stim_st['offset']*100, offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
#     for count_animal, animal in enumerate(animals_sw):
#         ax.scatter(offtracks_phase_stim_sw['offset']*100, offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
#     ax.set_xlabel('stride phase (%)', fontsize=20)
#     ax.set_ylabel(param_sym_label[p] + '\n for stim offset', fontsize=20)
#     ax.tick_params(axis='both', labelsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_offset.png')
#     plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_offset.svg')
# plt.close('all')


