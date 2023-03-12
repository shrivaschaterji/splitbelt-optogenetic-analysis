import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Online Tracking Treadmill\\Test Bonsai OneBelt Two Belts 100322\\MC16947\\'
subdir = 'TwoBeltsCM\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16947CM'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()

# LOAD PROCESSED DATA
[otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session] = otrack_class.load_processed_files()

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws_bottom(loco, animal, session)

# LATENCY OF OTRACK IN RELATION TO OFFTRACK
[tracks_hits_st, tracks_hits_sw, otrack_st_miss, otrack_sw_miss] = otrack_class.get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
latency_st = []
for trial in trials:
    diff_otrack_offtrack = tracks_hits_st.loc[tracks_hits_st['trial'] == trial, 'otrack_times']-tracks_hits_st.loc[tracks_hits_st['trial'] == trial, 'offtrack_times']
    diff_otrack_offtrack_peak_idx = find_peaks(-diff_otrack_offtrack)[0]
    latency_st.append(np.array(diff_otrack_offtrack)[diff_otrack_offtrack_peak_idx]*1000)
latency_sw = []
for trial in trials:
    diff_otrack_offtrack = tracks_hits_sw.loc[tracks_hits_sw['trial'] == trial, 'otrack_times']-tracks_hits_sw.loc[tracks_hits_sw['trial'] == trial, 'offtrack_times']
    diff_otrack_offtrack_peak_idx = find_peaks(-diff_otrack_offtrack)[0]
    latency_sw.append(np.array(diff_otrack_offtrack)[diff_otrack_offtrack_peak_idx]*1000)
# latency plot
fig, ax = plt.subplots(2, len(trials), tight_layout=True)
for count_t, trial in enumerate(trials):
    ax[0, count_t].hist(latency_st[count_t], bins=100, color='black', range=(-50, np.max([i for sublist in latency_st for i in sublist])))
    ax[0, count_t].set_title('stance latency trial '+str(trial))
    ax[0, count_t].spines['right'].set_visible(False)
    ax[0, count_t].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1, count_t].hist(latency_sw[count_t], bins=100, color='black', range=(0, np.max([i for sublist in latency_sw for i in sublist])))
    ax[1, count_t].set_title('swing latency trial '+str(trial))
    ax[1, count_t].spines['right'].set_visible(False)
    ax[1, count_t].spines['top'].set_visible(False)
# misses plot
fig, ax = plt.subplots(2, 1, tight_layout=True, sharey=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, np.array(otrack_st_miss), color='black')
ax[0].set_title('stance misses')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, np.array(otrack_sw_miss), color='black')
ax[1].set_title('swing misses')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
# frames poorly detected
[detected_frames_bad_st, detected_frames_bad_sw] = otrack_class.frames_outside_st_sw(trials, offtracks_st, offtracks_sw, otracks_st, otracks_sw)
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, detected_frames_bad_st, color='black')
ax[0].set_title('# frames where stance was assigned wrong')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, detected_frames_bad_sw, color='black')
ax[1].set_title('# frames where stance was assigned wrong')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

# LATENCY OF LIGHT IN RELATION TO OTRACK
fig, ax = plt.subplots(2, len(trials), tight_layout=True)
for count_t, trial in enumerate(trials):
    latency_light_st = np.load(os.path.join(otrack_class.path, 'processed files', 'latency_light_st_trial' + str(trial) + '.npy'), allow_pickle=True)
    ax[0, count_t].hist(latency_light_st, bins=100, color='black', range=(-50, np.nanmax(latency_light_st)))
    ax[0, count_t].set_title('led stance latency trial ' + str(trial))
    ax[0, count_t].spines['right'].set_visible(False)
    ax[0, count_t].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    latency_light_sw = np.load(os.path.join(otrack_class.path, 'processed files', 'latency_light_sw_trial' + str(trial) + '.npy'), allow_pickle=True)
    ax[1, count_t].hist(latency_light_sw, bins=100, color='black', range=(-50, np.nanmax(latency_light_sw)))
    ax[1, count_t].spines['right'].set_visible(False)
    ax[1, count_t].spines['top'].set_visible(False)
    ax[1, count_t].set_title('led swing latency trial ' + str(trial))

# PERIODS WITH LIGHT ON
trial = 1
st_led_trials = np.load(os.path.join(otrack_class.path, 'processed files', 'st_led_trials_trial' + str(trial) + '.npy'), allow_pickle=True)
sw_led_trials = np.load(os.path.join(otrack_class.path, 'processed files', 'sw_led_trials_trial' + str(trial) + '.npy'), allow_pickle=True)
sw_led_trials[1,-1] = 19970
paw_colors = ['red', 'magenta', 'blue', 'cyan']
p = 0
fig, ax = plt.subplots(tight_layout=True)
for r in range(np.shape(st_led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial-1][st_led_trials[0, r]], min(final_tracks_trials[trial-1][0, p, :-1])), timestamps_session[trial-1][st_led_trials[1, r]]-timestamps_session[trial-1][st_led_trials[0, r]], max(final_tracks_trials[trial-1][0, p, :-1]) - min(final_tracks_trials[trial-1][0, p, :-1]), fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
# for p in range(4):
ax.plot(timestamps_session[trial-1], final_tracks_trials[trial-1][0, p, :-1], color=paw_colors[p], linewidth=2)
ax.set_title('light on stance')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig, ax = plt.subplots(tight_layout=True)
for r in range(np.shape(sw_led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial-1][sw_led_trials[0, r]], min(final_tracks_trials[trial-1][0, p, :-1])), timestamps_session[trial-1][sw_led_trials[1, r]]-timestamps_session[trial-1][sw_led_trials[0, r]], max(final_tracks_trials[trial-1][0, p, :-1]) - min(final_tracks_trials[trial-1][0, p, :-1]), fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
ax.plot(timestamps_session[trial-1], final_tracks_trials[trial-1][0, p, :-1], color=paw_colors[p])
ax.set_title('light on swing')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
