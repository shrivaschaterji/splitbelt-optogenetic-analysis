import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

networks = ['CM tests', 'HR tests', 'Tailbase tests']
conditions_plot = [['50percent', '50percent'], ['50percent', '50percent'], ['50percent', '50percent']]
speeds = ['0,275', 'right_fast', 'left_fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
colors_networks = ['black', 'teal', 'orange']
summary_path = 'J:\\Data OPTO\\Benchmark plots\\Network comparison\\'

stim_duration_st_net = []
light_onset_phase_st_net = []
light_offset_phase_st_net = []
stim_duration_sw_net = []
light_onset_phase_sw_net = []
light_offset_phase_sw_net = []
stim_nr_st_net = []
stride_nr_st_net = []
stim_nr_sw_net = []
stride_nr_sw_net = []
accuracy_measures_st = np.zeros((6, 4, 3, len(networks)))
accuracy_measures_sw = np.zeros((6, 4, 3, len(networks)))
for count_n, n in enumerate(networks):
    c = conditions_plot[count_n]
    stim_duration_st_cond = []
    light_onset_phase_st_cond = []
    light_offset_phase_st_cond = []
    stim_duration_sw_cond = []
    light_onset_phase_sw_cond = []
    light_offset_phase_sw_cond = []
    stim_nr_st_cond = []
    stride_nr_st_cond = []
    stim_nr_sw_cond = []
    stride_nr_sw_cond = []
    for count_s, s in enumerate(np.array([1, 3, 4])):
        trials = trials_reshape[s, :].flatten()
        path_st = os.path.join('J:\\Data OPTO', n, c[0])
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_st)
        import locomotion_class
        loco = locomotion_class.loco_class(path_st)
        benchmark_accuracy_st = pd.read_csv(os.path.join(path_st, 'processed files', 'benchmark_accuracy.csv'))
        stim_duration_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_duration_st.npy'), allow_pickle=True)
        light_onset_phase_st_list = np.load(os.path.join(path_st, 'processed files', 'light_onset_phase_st.npy'), allow_pickle=True)
        light_offset_phase_st_list = np.load(os.path.join(path_st, 'processed files', 'light_offset_phase_st.npy'), allow_pickle=True)
        stim_nr_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_nr_st.npy'), allow_pickle=True)
        stride_nr_st_list = np.load(os.path.join(path_st, 'processed files', 'stride_nr_st.npy'), allow_pickle=True)
        stim_nr_st_cond.append(list(itertools.chain(*stim_nr_st_list[:, s])))
        stride_nr_st_cond.append(list(itertools.chain(*stride_nr_st_list[:, s])))
        stim_duration_st_cond.append(list(itertools.chain(*stim_duration_st_list[:, s])))
        light_onset_phase_st_cond.append(list(itertools.chain(*light_onset_phase_st_list[:, s])))
        light_offset_phase_st_cond.append(list(itertools.chain(*light_offset_phase_st_list[:, s])))
        accuracy_measures_st[:, :, count_s, count_n] = np.array(benchmark_accuracy_st[benchmark_accuracy_st['trial'].isin(trials)].iloc[:,3::2])
        path_sw = os.path.join('J:\\Data OPTO', n, c[1])
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_sw)
        import locomotion_class
        loco = locomotion_class.loco_class(path_sw)
        benchmark_accuracy_sw = pd.read_csv(os.path.join(path_sw, 'processed files', 'benchmark_accuracy.csv'))
        stim_duration_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_duration_sw.npy'), allow_pickle=True)
        light_onset_phase_sw_list = np.load(os.path.join(path_sw, 'processed files', 'light_onset_phase_sw.npy'), allow_pickle=True)
        light_offset_phase_sw_list = np.load(os.path.join(path_sw, 'processed files', 'light_offset_phase_sw.npy'), allow_pickle=True)
        stim_nr_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_nr_st.npy'), allow_pickle=True)
        stride_nr_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stride_nr_st.npy'), allow_pickle=True)
        stim_nr_sw_cond.append(list(itertools.chain(*stim_nr_sw_list[:, s])))
        stride_nr_sw_cond.append(list(itertools.chain(*stride_nr_sw_list[:, s])))
        stim_duration_sw_cond.append(list(itertools.chain(*stim_duration_sw_list[:, s])))
        light_onset_phase_sw_cond.append(list(itertools.chain(*light_onset_phase_sw_list[:, s])))
        light_offset_phase_sw_cond.append(list(itertools.chain(*light_offset_phase_sw_list[:, s])))
        accuracy_measures_sw[:, :, count_s, count_n] = np.array(benchmark_accuracy_sw[benchmark_accuracy_sw['trial'].isin(trials)].iloc[:,4::2])
    stim_duration_st_net.append(stim_duration_st_cond)
    light_onset_phase_st_net.append(light_onset_phase_st_cond)
    light_offset_phase_st_net.append(light_offset_phase_st_cond)
    stim_duration_sw_net.append(stim_duration_sw_cond)
    light_onset_phase_sw_net.append(light_onset_phase_sw_cond)
    light_offset_phase_sw_net.append(light_offset_phase_sw_cond)
    stim_nr_st_net.append(stim_nr_st_cond)
    stride_nr_st_net.append(stride_nr_st_cond)
    stim_nr_sw_net.append(stim_nr_sw_cond)
    stride_nr_sw_net.append(stride_nr_sw_cond)

# DURATION
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for n in range(len(networks)):
    violin_parts = ax.violinplot(stim_duration_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
        showextrema=False)
    for pc in violin_parts['bodies']:
        pc.set_color(colors_networks[n])
ax.set_xticks(np.arange(0, 9, 3)+0.5)
ax.set_xticklabels(speeds, fontsize=14)
ax.set_title('Stance stim duration', fontsize=16)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 0.3])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'stim_duration_st'), dpi=128)
plt.savefig(os.path.join(summary_path, 'stim_duration_st.svg'), dpi=128)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for n in range(len(networks)):
    violin_parts = ax.violinplot(stim_duration_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
        showextrema=False)
    for pc in violin_parts['bodies']:
        pc.set_color(colors_networks[n])
ax.set_xticks(np.arange(0, 9, 3)+0.5)
ax.set_xticklabels(speeds, fontsize=14)
ax.set_title('Swing stim duration', fontsize=16)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_ylim([0, 0.65])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'stim_duration_sw'), dpi=128)
plt.savefig(os.path.join(summary_path, 'stim_duration_sw.svg'), dpi=128)

# STIMULATION ONSETS AND OFFSETS
fraction_strides_stim_on_st_networks = np.zeros((len(networks), len(speeds)))
fraction_strides_stim_on_sw_networks = np.zeros((len(networks), len(speeds)))
for count_n, n in enumerate(networks):
    for count_s, s in enumerate(speeds):
        [fraction_strides_stim_st_on, fraction_strides_stim_st_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_st_net[count_n][count_s],
                                                                 light_offset_phase_st_net[count_n][count_s], 'stance',
                                                                 16, np.sum(stim_nr_st_net[count_n][count_s]),
                                                                 np.sum(stride_nr_st_net[count_n][count_s]), 'Greys',
                                                                 summary_path,
                                                                 '\\light_stance_' + networks[count_n] + '_' +
                                                                 speeds[count_s])
        fraction_strides_stim_on_st_networks[count_n, count_s] = fraction_strides_stim_st_on
        plt.close('all')
        [fraction_strides_stim_sw_on, fraction_strides_stim_sw_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_sw_net[count_n][count_s],
                                                                 light_offset_phase_sw_net[count_n][count_s], 'swing',
                                                                 16, np.sum(stim_nr_sw_net[count_n][count_s]),
                                                                 np.sum(stride_nr_sw_net[count_n][count_s]), 'Greys',
                                                                 summary_path,
                                                                 '\\light_swing_' + networks[count_n] + '_' +
                                                                 speeds[count_s])
        fraction_strides_stim_on_sw_networks[count_n, count_s] = fraction_strides_stim_sw_on
        plt.close('all')

# FRACTION OF STIMULATED STRIDES
networks_labels = ['Center of mass normalization', 'Hind right paw normalization', 'Base of the tail normalization']
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_labels):
    ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
               fraction_strides_stim_on_st_networks[count_n, :],
               s=40, color=colors_networks[count_n], label=n)
# ax.legend(networks_labels, frameon=False, fontsize=14)
ax.set_xticks(np.arange(0, 30, 10) + 2.5)
ax.set_xticklabels(speeds, fontsize=14)
ax.set_ylabel('Fraction of stimulated\nstrides', fontsize=14)
ax.set_ylim([0, 1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_' + conditions_plot[0][0]), dpi=128)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_' + conditions_plot[0][0] + '.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks):
    ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
               fraction_strides_stim_on_sw_networks[count_n, :],
               s=40, color=colors_networks[count_n], label=n)
ax.legend(networks, frameon=False, fontsize=14)
ax.set_xticks(np.arange(0, 30, 10) + 2.5)
ax.set_xticklabels(speeds, fontsize=14)
ax.set_ylabel('Fraction of stimulated\nstrides', fontsize=14)
ax.set_ylim([0, 1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_' + conditions_plot[0][0]), dpi=128)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_' + conditions_plot[0][0] + '.svg'), dpi=128)

# ACCURACY
ylabel_names = ['% correct hits', '% F1 score', '% false negatives', '% false positives']
for i in range(len(measure_name)):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        for a in range(6):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_st[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label=networks[n])
            else:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_st[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label='_nolegend_')
    ax.set_xticks(np.arange(0, 30, 10))
    ax.set_xticklabels(speeds, fontsize=14)
    ax.set_ylim([0, 1])
    ax.set_title('Stance ' + measure_name[i].replace('_', ' '), fontsize=16)
    ax.set_ylabel(measure_name[i], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.savefig(os.path.join(summary_path, measure_name[i] + '_st'), dpi=128)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_st.svg'), dpi=128)

    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        for a in range(6):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label=networks[n])
            else:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label='_nolegend_')
    ax.set_xticks(np.arange(0, 30, 10))
    ax.set_xticklabels(speeds, fontsize=14)
    ax.set_title('Swing ' + measure_name[i].replace('_', ' '), fontsize=16)
    ax.set_ylabel(measure_name[i], fontsize=14)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw'), dpi=128)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw.svg'), dpi=128)
    plt.close('all')

