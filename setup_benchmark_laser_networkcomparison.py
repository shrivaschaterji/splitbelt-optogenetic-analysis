import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

networks = ['CM tests', 'HR tests', 'Tailbase tests']
conditions = ['25percent', '50percent', '75percent']
speeds = ['0,175', '0,275', '0,375', 'right_fast', 'left_fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
colors_networks = ['black', 'teal', 'orange']
summary_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Benchmark plots\\Condition comparison\\'

for idx_speed in range(len(speeds)):
    accuracy_measures_st = np.zeros((6, 4, len(conditions), len(networks)))
    accuracy_measures_sw = np.zeros((6, 4, len(conditions), len(networks)))
    stim_duration_st_net = []
    stim_duration_sw_net = []
    stim_onset_phase_st_net = []
    stim_onset_phase_sw_net = []
    stim_onset_time_st_net = []
    stim_onset_time_sw_net = []
    stim_offset_phase_st_net = []
    stim_offset_phase_sw_net = []
    stim_offset_time_st_net = []
    stim_offset_time_sw_net = []
    for count_n, n in enumerate(networks):
        stim_duration_st_cond = []
        stim_duration_sw_cond = []
        stim_onset_phase_st_cond = []
        stim_onset_phase_sw_cond = []
        stim_onset_time_st_cond = []
        stim_onset_time_sw_cond = []
        stim_offset_phase_st_cond = []
        stim_offset_phase_sw_cond = []
        stim_offset_time_st_cond = []
        stim_offset_time_sw_cond = []
        for count_c, c in enumerate(conditions):
            path = os.path.join('C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests', n, c)
            if not os.path.exists(os.path.join(path, 'plots')):
                os.mkdir(os.path.join(path, 'plots'))
            import online_tracking_class
            otrack_class = online_tracking_class.otrack_class(path)
            import locomotion_class
            loco = locomotion_class.loco_class(path)
            stim_duration_st_list = np.load(os.path.join(path, 'processed files', 'stim_duration_st.npy'), allow_pickle=True)
            stim_duration_st_cond.append(list(itertools.chain(*stim_duration_st_list[:, idx_speed])))
            stim_duration_sw_list = np.load(os.path.join(path, 'processed files', 'stim_duration_sw.npy'), allow_pickle=True)
            stim_duration_sw_cond.append(list(itertools.chain(*stim_duration_sw_list[:, idx_speed])))
            stim_onset_phase_st_list = np.load(os.path.join(path, 'processed files', 'stim_onset_phase_st.npy'), allow_pickle=True)
            stim_onset_phase_st_cond.append(list(itertools.chain(*stim_onset_phase_st_list[:, idx_speed])))
            stim_onset_phase_sw_list = np.load(os.path.join(path, 'processed files', 'stim_onset_phase_sw.npy'), allow_pickle=True)
            stim_onset_phase_sw_cond.append(list(itertools.chain(*stim_onset_phase_sw_list[:, idx_speed])))
            stim_offset_phase_st_list = np.load(os.path.join(path, 'processed files', 'stim_offset_phase_st.npy'), allow_pickle=True)
            stim_offset_phase_st_cond.append(list(itertools.chain(*stim_offset_phase_st_list[:, idx_speed])))
            stim_offset_phase_sw_list = np.load(os.path.join(path, 'processed files', 'stim_offset_phase_sw.npy'), allow_pickle=True)
            stim_offset_phase_sw_cond.append(list(itertools.chain(*stim_offset_phase_sw_list[:, idx_speed])))
            stim_onset_time_st_list = np.load(os.path.join(path, 'processed files', 'stim_onset_time_st.npy'), allow_pickle=True)
            stim_onset_time_st_cond.append(list(itertools.chain(*stim_onset_time_st_list[:, idx_speed])))
            stim_onset_time_sw_list = np.load(os.path.join(path, 'processed files', 'stim_onset_time_sw.npy'), allow_pickle=True)
            stim_onset_time_sw_cond.append(list(itertools.chain(*stim_onset_time_sw_list[:, idx_speed])))
            stim_offset_time_st_list = np.load(os.path.join(path, 'processed files', 'stim_offset_time_st.npy'), allow_pickle=True)
            stim_offset_time_st_cond.append(list(itertools.chain(*stim_offset_time_st_list[:, idx_speed])))
            stim_offset_time_sw_list = np.load(os.path.join(path, 'processed files', 'stim_offset_time_sw.npy'), allow_pickle=True)
            stim_offset_time_sw_cond.append(list(itertools.chain(*stim_offset_time_sw_list[:, idx_speed])))
            benchmark_accuracy = pd.read_csv(os.path.join(path, 'processed files', 'benchmark_accuracy.csv'))
            trials = trials_reshape[idx_speed, :]
            accuracy_measures_mean = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].mean()[1:]
            accuracy_measures_std = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].std()[1:]
            accuracy_measures_st[:, :, count_c, count_n] = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].iloc[:, 3::2]
            accuracy_measures_sw[:, :, count_c, count_n] = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].iloc[:, 4::2]
        stim_duration_st_net.append(stim_duration_st_cond)
        stim_duration_sw_net.append(stim_duration_sw_cond)
        stim_onset_phase_st_net.append(stim_onset_phase_st_cond)
        stim_onset_phase_sw_net.append(stim_onset_phase_sw_cond)
        stim_onset_time_st_net.append(stim_onset_time_st_cond)
        stim_onset_time_sw_net.append(stim_onset_time_sw_cond)
        stim_offset_phase_st_net.append(stim_offset_phase_st_cond)
        stim_offset_phase_sw_net.append(stim_offset_phase_sw_cond)
        stim_offset_time_st_net.append(stim_offset_time_st_cond)
        stim_offset_time_sw_net.append(stim_offset_time_sw_cond)

    # DURATION
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_duration_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Stance stim duration ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (s)', fontsize=14)
    ax.set_ylim([-0.1, 0.85])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_st_' + speeds[idx_speed]), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_duration_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
             pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Swing stim duration ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (s)', fontsize=14)
    ax.set_ylim([-0.1, 0.85])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_sw_' + speeds[idx_speed]), dpi=128)

    # ONSET PHASE
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_onset_phase_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Stance stim onset phase ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('stance phase', fontsize=14)
    ax.set_ylim([-10, 3])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_onset_phase_st_' + speeds[idx_speed]), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_onset_phase_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Swing stim onset phase ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('swing phase', fontsize=14)
    ax.set_ylim([-10, 3])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_onset_phase_sw_' + speeds[idx_speed]), dpi=128)

    # OFFSET PHASE
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_offset_phase_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Stance stim offset phase ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('stance phase', fontsize=14)
    ax.set_ylim([-0.1, 1.1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_offset_phase_st_' + speeds[idx_speed]), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_offset_phase_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Swing stim offset phase ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('swing phase', fontsize=14)
    ax.set_ylim([-0.1, 1.1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_offset_phase_sw_' + speeds[idx_speed]), dpi=128)

    # ONSET TIME
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_onset_time_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Stance stim onset time ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_ylim([-500, 150])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_onset_time_st_' + speeds[idx_speed]), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_onset_time_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Swing stim onset time ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_ylim([-500, 150])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_onset_time_sw_' + speeds[idx_speed]), dpi=128)

    # OFFSET TIME
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_offset_time_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Stance stim offset time ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_ylim([-10, 250])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_offset_time_st_' + speeds[idx_speed]), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        violin_parts = ax.violinplot(stim_offset_time_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
            showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_networks[n])
        # violin_parts['cbars'].set_color(colors_networks[n])
        # violin_parts['cmins'].set_color(colors_networks[n])
        # violin_parts['cmaxes'].set_color(colors_networks[n])
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_title('Swing stim offset time ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_ylim([-10, 250])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_offset_time_sw_' + speeds[idx_speed]), dpi=128)

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
        ax.legend(networks, frameon=False, fontsize=12)
        ax.set_xticks(np.arange(0, 30, 10))
        ax.set_xticklabels(conditions, fontsize=14)
        ax.set_title('Stance ' + measure_name[i].replace('_', ' ') + ' ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
        ax.set_ylabel(measure_name[i], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_st_' + speeds[idx_speed]), dpi=128)

        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
        for n in range(len(networks)):
            for a in range(6):
                if a == 0:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                               s=10, color=colors_networks[n], linewidth=2, label=networks[n])
                else:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                               s=10, color=colors_networks[n], linewidth=2, label='_nolegend_')
        ax.legend(networks, frameon=False, fontsize=12)
        ax.set_xticks(np.arange(0, 30, 10))
        ax.set_xticklabels(conditions, fontsize=14)
        ax.set_title('Swing ' + measure_name[i].replace('_', ' ') + ' ' + speeds[idx_speed].replace('_', ' '), fontsize=16)
        ax.set_ylabel(measure_name[i], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw_' + speeds[idx_speed]), dpi=128)
        plt.close('all')

