import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

network = 'HR tests'
conditions_plot = ['75percent', '25percent']
speeds = ['0,175', '0,275', '0,375', 'right_fast', 'left_fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
cmap_speeds = plt.get_cmap('terrain')
colors_speeds = [cmap_speeds(i) for i in np.linspace(0, 1, int(np.floor(len(speeds) + 1)))]
summary_path = 'J:\\Data OPTO\\Benchmark plots\\Experiment settings\\'

accuracy_measures_st = np.zeros((6, 4, len(speeds)))
accuracy_measures_sw = np.zeros((6, 4, len(speeds)))
stim_duration_st = []
stim_onset_time_st = []
stim_offset_time_st = []
stim_duration_sw = []
stim_onset_time_sw = []
stim_offset_time_sw = []
for idx_speed in range(len(speeds)):
    trials = trials_reshape[idx_speed, :]
    path_st = os.path.join('J:\\Data OPTO', network, conditions_plot[0])
    import online_tracking_class
    otrack_class = online_tracking_class.otrack_class(path_st)
    import locomotion_class
    loco = locomotion_class.loco_class(path_st)
    benchmark_accuracy_st = pd.read_csv(os.path.join(path_st, 'processed files', 'benchmark_accuracy.csv'))
    stim_duration_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_duration_st.npy'), allow_pickle=True)
    stim_onset_time_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_onset_time_st.npy'), allow_pickle=True)
    stim_offset_time_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_offset_time_st.npy'), allow_pickle=True)
    stim_duration_st.append(list(itertools.chain(*stim_duration_st_list[:, idx_speed])))
    stim_onset_time_st.append(list(itertools.chain(*stim_onset_time_st_list[:, idx_speed])))
    stim_offset_time_st.append(list(itertools.chain(*stim_offset_time_st_list[:, idx_speed])))
    accuracy_measures_st[:, :, idx_speed] = np.array(benchmark_accuracy_st[benchmark_accuracy_st['trial'].isin(trials)].iloc[:,3::2])
    path_sw = os.path.join('J:\\Data OPTO', network, conditions_plot[1])
    import online_tracking_class
    otrack_class = online_tracking_class.otrack_class(path_sw)
    import locomotion_class
    loco = locomotion_class.loco_class(path_sw)
    benchmark_accuracy_sw = pd.read_csv(os.path.join(path_sw, 'processed files', 'benchmark_accuracy.csv'))
    stim_duration_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_duration_sw.npy'), allow_pickle=True)
    stim_onset_time_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_onset_time_sw.npy'), allow_pickle=True)
    stim_offset_time_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_offset_time_sw.npy'), allow_pickle=True)
    stim_duration_sw.append(list(itertools.chain(*stim_duration_sw_list[:, idx_speed])))
    stim_onset_time_sw.append(list(itertools.chain(*stim_onset_time_sw_list[:, idx_speed])))
    stim_offset_time_sw.append(list(itertools.chain(*stim_offset_time_sw_list[:, idx_speed])))
    accuracy_measures_sw[:, :, idx_speed] = np.array(benchmark_accuracy_sw[benchmark_accuracy_sw['trial'].isin(trials)].iloc[:,4::2])

speeds_label = ['0,175', '0,275', '0,375', 'split right\nfast', 'split left\nfast']
# DURATION
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_duration_st, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 0.4])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_duration_st'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_duration_sw, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
        pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 0.2])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_duration_sw'), dpi=128)

# ONSET TIME
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_onset_time_st, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([-400, 75])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_onset_time_st'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_onset_time_sw, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([-250, 100])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_onset_time_sw'), dpi=128)

# OFFSET TIME
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_offset_time_st, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
     pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 150])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_offset_time_st'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
violin_parts = ax.violinplot(stim_offset_time_sw, positions=np.arange(0, 5, 1),
        showextrema=False)
for s, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_speeds[s])
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(speeds_label, fontsize=11)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 75])
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_offset_time_sw'), dpi=128)
plt.close('all')

# ACCURACY
ylabel_names = ['% correct hits', '% F1 score', '% false negatives', '% false positives']
for i in range(len(measure_name)):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for s in range(len(speeds)):
        for a in range(6):
            if a == 0:
                ax.scatter(np.arange(0, 20, 10)+(np.ones(2)*s)+np.random.rand(2), [accuracy_measures_st[a, i, s], accuracy_measures_sw[a, i, s]],
                           s=10, color=colors_speeds[s], linewidth=2, label=speeds_label[s])
            else:
                ax.scatter(np.arange(0, 20, 10)+(np.ones(2)*s)+np.random.rand(2), [accuracy_measures_st[a, i, s], accuracy_measures_sw[a, i, s]],
                           s=10, color=colors_speeds[s], linewidth=2, label='_nolegend_')
    ax.set_xticks(np.arange(0, 20, 10)+2.5)
    ax.set_xticklabels(['stance', 'swing'], fontsize=14)
    ax.set_ylabel(measure_name[i], fontsize=14)
    #ax.legend(speeds_label, frameon=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if i == 0:
        ax.set_ylim([0, 1.05])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, measure_name[i]), dpi=128)
    plt.close('all')

