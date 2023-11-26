import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

#path inputs
path_st = 'J:\\Opto JAWS Data\\split left fast stance stim\\'
path_sw = 'J:\\Opto JAWS Data\\split left fast swing stim\\'
experiment_type = 'split'
save_path = 'J:\\Thesis\\for figures\\fig split left fast opto\\'
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
    animals = ['MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
    Nanimals = len(animals)-1 #because of bad trackig took one animal in each session
    Ntrials = 24
else:
    param_sym_bs_st = param_sym_bs_st_withnan
    param_sym_bs_sw = param_sym_bs_sw_withnan
    param_paw_bs_st = param_paw_bs_st_withnan
    param_paw_bs_sw = param_paw_bs_sw_withnan
    param_phase_st = param_phase_st_withnan
    param_phase_sw = param_phase_sw_withnan
    animals = ['MC16851', 'MC17319', 'MC17665', 'MC17670']
    stim_trials = np.arange(9, 19)
    Nanimals = len(animals)
    Ntrials = 28

min_plot = [-1, -2.5, -4, -2, -2.5]
max_plot = [1, 2.5, 4, 2, 2.5]
for p in range(np.shape(param_sym_name)[0]):
    mean_data_st = np.nanmean(param_sym_bs_st[p, :, :], axis=0)
    std_data_st = np.nanstd(param_sym_bs_st[p, :, :], axis=0) / np.sqrt(Nanimals)
    mean_data_sw = np.nanmean(param_sym_bs_sw[p, :, :], axis=0)
    std_data_sw = np.nanstd(param_sym_bs_sw[p, :, :], axis=0) / np.sqrt(Nanimals)
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, min_plot[p]), 8, max_plot[p]-min_plot[p], fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
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
    ae_amp.extend(param_sym_bs_st[count_p, :, stim_trials[-1]])
    delta_split.extend(param_sym_bs_st[count_p, :, stim_trials[-1]-1]-param_sym_bs_st[count_p, :, stim_trials[0]-1])
    param.extend(np.repeat(p, group_st_len))
    group_sw_len = np.shape(param_sym_bs_sw)[1]
    group_id.extend(np.repeat(1, group_sw_len))
    ae_amp.extend(np.abs(param_sym_bs_sw[count_p, :, stim_trials[-1]]))
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

# Individual limbs - st
for p in range(np.shape(param_sym_name)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.vstack((np.nanmean(param_paw_bs_st[p, :, 0, :], axis=0), np.nanmean(param_paw_bs_st[p, :, 1, :], axis=0),
        np.nanmean(param_paw_bs_st[p, :, 2, :], axis=0), np.nanmean(param_paw_bs_st[p, :, 3, :], axis=0)))
    std_data = np.vstack((np.nanstd(param_paw_bs_st[p, :, 0, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_st[p, :, 1, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_st[p, :, 2, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_st[p, :, 3, :], axis=0)/np.sqrt(Nanimals)))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    for paw in range(4):
        plt.plot(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs_st[p, :, paw, :], axis=0), linewidth=2, color=paw_colors[paw])
        plt.fill_between(np.arange(1, Ntrials+1),
            np.nanmean(param_paw_bs_st[p, :, paw, :], axis=0)-(np.nanstd(param_paw_bs_st[p, :, paw, :], axis=0)/np.sqrt(Nanimals)),
            np.nanmean(param_paw_bs_st[p, :, paw, :], axis=0)+(np.nanstd(param_paw_bs_st[p, :, paw, :], axis=0)/np.sqrt(Nanimals)), color=paw_colors[paw], alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    # if p == 2:
    #     plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_st.png'), dpi=128)
    plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_st.svg'), dpi=128)
plt.close('all')

# Individual limbs - sw
for p in range(np.shape(param_sym_name)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.vstack((np.nanmean(param_paw_bs_sw[p, :, 0, :], axis=0), np.nanmean(param_paw_bs_sw[p, :, 1, :], axis=0),
        np.nanmean(param_paw_bs_sw[p, :, 2, :], axis=0), np.nanmean(param_paw_bs_sw[p, :, 3, :], axis=0)))
    std_data = np.vstack((np.nanstd(param_paw_bs_sw[p, :, 0, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_sw[p, :, 1, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_sw[p, :, 2, :], axis=0)/np.sqrt(Nanimals),
        np.nanstd(param_paw_bs_sw[p, :, 3, :], axis=0)/np.sqrt(Nanimals)))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    for paw in range(4):
        plt.plot(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs_sw[p, :, paw, :], axis=0), linewidth=2, color=paw_colors[paw])
        plt.fill_between(np.arange(1, Ntrials+1),
            np.nanmean(param_paw_bs_sw[p, :, paw, :], axis=0)-(np.nanstd(param_paw_bs_sw[p, :, paw, :], axis=0)/np.sqrt(Nanimals)),
            np.nanmean(param_paw_bs_sw[p, :, paw, :], axis=0)+(np.nanstd(param_paw_bs_sw[p, :, paw, :], axis=0)/np.sqrt(Nanimals)), color=paw_colors[paw], alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    # if p == 2:
    #     plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_sw.png'), dpi=128)
    plt.savefig(os.path.join(save_path, 'mean_animals_paws_' + param_sym_name[p] + '_sw.svg'), dpi=128)
plt.close('all')

# Stance phase - st
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
rectangle = plt.Rectangle((stim_trials[0] - 0.5, 70), 10,
                          230-70, fc='lightblue', alpha=0.3, zorder=-1)
plt.gca().add_patch(rectangle)
for paw in range(4):
    plt.plot(np.arange(1, Ntrials + 1), np.rad2deg(np.nanmean(param_phase_st[paw, :, :], axis=0)), linewidth=2,
             color=paw_colors[paw])
    plt.fill_between(np.arange(1, Ntrials + 1),
                     np.rad2deg(np.nanmean(param_phase_st[paw, :, :], axis=0) - (
                                 np.nanstd(param_phase_st[paw, :, :], axis=0) / np.sqrt(Nanimals))),
                     np.rad2deg(np.nanmean(param_phase_st[paw, :, :], axis=0) + (
                                 np.nanstd(param_phase_st[paw, :, :], axis=0) / np.sqrt(Nanimals))),
                     color=paw_colors[paw], alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel('Stance phasing\n(degrees)', fontsize=20)
ax.set_ylim([70, 230])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_st.png'), dpi=128)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_st.svg'), dpi=128)

# Stance phase - sw
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
rectangle = plt.Rectangle((stim_trials[0] - 0.5, 70), 10,
                          230-70, fc='lightblue', alpha=0.3, zorder=-1)
plt.gca().add_patch(rectangle)
for paw in range(4):
    plt.plot(np.arange(1, Ntrials + 1), np.rad2deg(np.nanmean(param_phase_sw[paw, :, :], axis=0)), linewidth=2,
             color=paw_colors[paw])
    plt.fill_between(np.arange(1, Ntrials + 1),
                     np.rad2deg(np.nanmean(param_phase_sw[paw, :, :], axis=0) - (
                                 np.nanstd(param_phase_sw[paw, :, :], axis=0) / np.sqrt(Nanimals))),
                     np.rad2deg(np.nanmean(param_phase_sw[paw, :, :], axis=0) + (
                                 np.nanstd(param_phase_sw[paw, :, :], axis=0) / np.sqrt(Nanimals))),
                     color=paw_colors[paw], alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel('Stance phasing\n(degrees)', fontsize=20)
ax.set_ylim([70, 230])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_sw.png'), dpi=128)
plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_sw.svg'), dpi=128)
plt.close('all')

# Timing
for p in range(3):
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True, sharex=True, sharey=True)
    for count_animal, animal in enumerate(animals):
        offtracks_phase_stim_st = pd.read_csv(
            os.path.join(path_st, 'grouped output', 'offtracks_phase_stim_' + experiment_st + '_' + animal + '.csv'))
        offtracks_phase_stim_sw = pd.read_csv(
            os.path.join(path_sw, 'grouped output', 'offtracks_phase_stim_' + experiment_sw + '_' + animal + '.csv'))
        ax.scatter(offtracks_phase_stim_st['onset'], offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
        ax.scatter(offtracks_phase_stim_sw['onset'], offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
        ax.set_ylabel(param_sym_label[p] + '\n for stim onset', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_onset.png')
    plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_onset.svg')
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True, sharex=True, sharey=True)
    for count_animal, animal in enumerate(animals):
        ax.scatter(offtracks_phase_stim_st['offset'], offtracks_phase_stim_st[param_sym_name[p]], s=5, color='orange')
        ax.scatter(offtracks_phase_stim_sw['offset'], offtracks_phase_stim_sw[param_sym_name[p]], s=5, color='green')
        ax.set_xlabel('stride phase (%)', fontsize=20)
        ax.set_ylabel(param_sym_label[p] + '\n for stim offset', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_offset.png')
    plt.savefig(save_path + experiment_type + '_mean_animals_laser_phase_sym_' + param_sym_name[p] + '_offset.svg')
plt.close('all')

