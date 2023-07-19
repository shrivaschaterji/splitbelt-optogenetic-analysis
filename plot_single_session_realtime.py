import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\Real-time plots\\swing stim tied session\\'
event = path.split('\\')[-2].split(' ')[0]
animals = ['MC16851','MC17319', 'MC17665', 'MC17670'] #'MC16851',
# stim_trials = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) #split
stim_trials = np.array([9, 10, 11, 12, 13, 14, 15, 16]) #tied
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

accuracy = []
f1_score = []
fn = []
fp = []
animal_id = []
trial_id = []
stim_duration = []
stim_onset_phase = []
stim_onset_time = []
stim_offset_phase = []
stim_offset_time = []
for count_a, animal in enumerate(animals):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

    # LASER ACCURACY
    tp_laser = np.zeros(len(stim_trials))
    tn_laser = np.zeros(len(stim_trials))
    f1_laser = np.zeros(len(stim_trials))
    fn_laser = np.zeros(len(stim_trials))
    fp_laser = np.zeros(len(stim_trials))
    for count_t, trial in enumerate(stim_trials):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        tn_laser[count_t] = tn_trial
        f1_laser[count_t] = f1_trial
        fn_laser[count_t] = fn_trial
        fp_laser[count_t] = fp_trial
    accuracy.extend(tp_laser+tn_laser)
    f1_score.extend(f1_laser)
    fn.extend(fn_laser)
    fp.extend(fp_laser)
    trial_id.extend(stim_trials)
    animal_id.extend(np.repeat(animal, len(stim_trials)))

    #LASER DURATION AND PHASE
    stim_duration_trials = []
    stim_onset_phase_trials = []
    stim_onset_time_trials = []
    stim_offset_phase_trials = []
    stim_offset_time_trials = []
    for count_t, t in enumerate(stim_trials):
        # stim duration
        stim_duration_singletrial = laser_on.loc[laser_on['trial'] == t]['time_off'] - laser_on.loc[laser_on['trial'] == t]['time_on']
        stim_duration_trials.extend(list(np.array(stim_duration_singletrial) * 1000))
        # stim onset and offset
        [light_onset_phase, light_offset_phase] = otrack_class.laser_presentation_phase(t, event, offtracks_st,
                                                                                              offtracks_sw, laser_on, 0)
        [light_onset_time, light_offset_time] = otrack_class.laser_presentation_phase(t, event, offtracks_st,
                                                                                            offtracks_sw, laser_on, 1)
        stim_onset_phase_trials.extend(light_onset_phase)
        stim_onset_time_trials.extend(list(np.array(light_onset_time) * 1000))  # in msec
        stim_offset_phase_trials.extend(light_offset_phase)
        stim_offset_time_trials.extend(list(np.array(light_offset_time) * 1000))
    stim_duration.append(stim_duration_trials)
    stim_onset_phase.append(stim_onset_phase_trials)
    stim_onset_time.append(stim_onset_time_trials)
    stim_offset_phase.append(stim_offset_phase_trials)
    stim_offset_time.append(stim_offset_time_trials)

benchmark_accuracy = pd.DataFrame(
    {'animal': animal_id, 'trial': trial_id, 'accuracy': accuracy, 'f1_score': f1_score, 'fn': fn, 'fp': fp})
benchmark_accuracy.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_accuracy.csv'), sep=',', index=False)

np.save(os.path.join(otrack_class.path, 'processed files', 'stim_duration_st.npy'), stim_duration, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_onset_phase_st.npy'), stim_onset_phase, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_onset_time_st.npy'), stim_onset_time, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_offset_phase_st.npy'), stim_offset_phase, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_offset_time_st.npy'), stim_offset_time, allow_pickle=True)

# STIMULATION ACCURACY MEASURES
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
for count_a, animal in enumerate(animals):
    f1_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'f1_score'])
    ax.plot(stim_trials, f1_data, marker='o', markersize=10)
ax.set_xticks(stim_trials)
ax.set_xticklabels(list(map(str, stim_trials)), fontsize=14)
ax.set_title(event + ' F1 score', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'f1_score'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    accuracy_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'accuracy'])
    ax.plot(stim_trials, accuracy_data, marker='o', markersize=10)
ax.set_xticks(stim_trials)
ax.set_xticklabels(list(map(str, stim_trials)), fontsize=14)
ax.set_title(event + ' accuracy score', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'accuracy_st'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fn_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fn'])
    ax.plot(stim_trials, fn_data, marker='o', markersize=10)
ax.set_xticks(stim_trials)
ax.set_xticklabels(list(map(str, stim_trials)), fontsize=14)
ax.set_title(event + ' false negatives', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fn'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fp_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fp'])
    ax.plot(stim_trials, fp_data, marker='o', markersize=10)
ax.set_xticks(stim_trials)
ax.set_xticklabels(list(map(str, stim_trials)), fontsize=14)
ax.set_title(event + ' false positives', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fp'), dpi=128)
plt.close('all')

colors_animals = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors_animals = colors_animals[1:] for swing stim
# STIMULATION DURATION
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
violin_parts = ax.violinplot(stim_duration, positions=np.arange(len(animals)))
for count_a, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_animals[count_a])
violin_parts['cbars'].set_color(colors_animals)
violin_parts['cmins'].set_color(colors_animals)
violin_parts['cmaxes'].set_color(colors_animals)
ax.set_xticks(np.arange(len(animals)))
ax.set_xticklabels(animals, fontsize=14)
ax.set_ylim([0, 400])
ax.set_ylabel('Light on\nduration (ms)', fontsize=14)
ax.set_title(event + ' stimulus duration', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_duration'), dpi=128)

# STIMULATION ONSET AND OFFSET IN %STRIDE
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
violin_parts = ax.violinplot(stim_onset_phase)
for count_a, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_animals[count_a])
violin_parts['cbars'].set_color(colors_animals)
violin_parts['cmins'].set_color(colors_animals)
violin_parts['cmaxes'].set_color(colors_animals)
ax.set_xticks(np.arange(len(animals))+1)
ax.set_xticklabels(animals, fontsize=14)
ax.set_ylim([-20, 1])
ax.set_ylabel('Light on\nlatency (%stride)', fontsize=14)
ax.set_title(event + ' stimulus onset phase', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_onset_phase'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
violin_parts = ax.violinplot(stim_offset_phase)
for count_a, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_animals[count_a])
violin_parts['cbars'].set_color(colors_animals)
violin_parts['cmins'].set_color(colors_animals)
violin_parts['cmaxes'].set_color(colors_animals)
ax.set_xticks(np.arange(len(animals))+1)
ax.set_xticklabels(animals, fontsize=14)
ax.set_ylim([-0.1, 1.1])
ax.set_ylabel('Light on\nlatency (%stride)', fontsize=14)
ax.set_title(event + ' stimulus offset phase', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_offset_phase'), dpi=128)

# STIMULATION ONSET AND OFFSET IN TIME
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
violin_parts = ax.violinplot(stim_onset_time)
for count_a, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_animals[count_a])
violin_parts['cbars'].set_color(colors_animals)
violin_parts['cmins'].set_color(colors_animals)
violin_parts['cmaxes'].set_color(colors_animals)
ax.set_xticks(np.arange(len(animals))+1)
ax.set_xticklabels(animals, fontsize=14)
ax.set_ylim([-250, 200])
ax.set_ylabel('Light on\nlatency (ms)', fontsize=14)
ax.set_title(event + ' stimulus onset time', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_onset_time'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
violin_parts = ax.violinplot(stim_offset_time)
for count_a, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_animals[count_a])
violin_parts['cbars'].set_color(colors_animals)
violin_parts['cmins'].set_color(colors_animals)
violin_parts['cmaxes'].set_color(colors_animals)
ax.set_xticks(np.arange(len(animals))+1)
ax.set_xticklabels(animals, fontsize=14)
ax.set_ylim([0, 200])
ax.set_ylabel('Light on\nlatency (ms)', fontsize=14)
ax.set_title(event + ' stimulus offset time', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_offset_time'), dpi=128)
plt.close('all')
