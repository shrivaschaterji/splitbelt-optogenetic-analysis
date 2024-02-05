import os
import numpy as np
import matplotlib.pyplot as plt

paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\'
subdir = '040423 mobile network crop bottom tests\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16946'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()

# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, st_led_on, sw_led_on] = otrack_class.load_processed_files()

# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_data)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

# PROCESS SYNCHRONIZER LASER SIGNALS
laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

trial = 1
#measure difference since the first time it was on and laser on
otracks_st_frames_trial = np.array(otracks_st.loc[otracks_st['trial']==trial, 'frames'])
st_on_otrack_trial = np.where(np.diff(otracks_st_frames_trial)>4)[0]+1
st_on_otrack_trial_all = np.insert(st_on_otrack_trial, 0, 0)
otracks_sw_frames_trial = np.array(otracks_sw.loc[otracks_sw['trial']==trial, 'frames'])
sw_on_otrack_trial = np.where(np.diff(otracks_sw_frames_trial)>4)[0]+1
sw_on_otrack_trial_all = np.insert(sw_on_otrack_trial, 0, 0)
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
        color='black')
ax.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'time'],
           otracks_st.loc[otracks_st['trial'] == trial, 'x'], color='orange', s=100)
for i in st_on_otrack_trial_all:
    ax.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'time'][i],
               otracks_st.loc[otracks_st['trial'] == trial, 'x'][i], color='red', s=40)
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
        color='black')
ax.scatter(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'],
           otracks_sw.loc[otracks_sw['trial'] == trial, 'x'], color='green', s=100)
for i in sw_on_otrack_trial_all:
    ax.scatter(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'][i],
               otracks_sw.loc[otracks_sw['trial'] == trial, 'x'][i], color='red', s=40)
st_on_otrack_trial_all_time = np.array(otracks_st.loc[otracks_st['trial'] == trial, 'time'])[st_on_otrack_trial_all]
sw_on_otrack_trial_all_time = np.array(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'])[sw_on_otrack_trial_all]
time_laser_on = laser_on.loc[laser_on['trial'] == trial]['time_on']
time_led_on = timestamps_session[trial-1][sw_led_on.loc[sw_led_on['trial'] == trial, 'frames_on']]
otrack_laser_st_latency = np.zeros(len(st_on_otrack_trial_all_time))
laser_st_idx_match = np.zeros(len(st_on_otrack_trial_all_time))
for count_i, i in enumerate(st_on_otrack_trial_all_time):
    laser_st_idx_match[count_i] = np.argmin(np.abs(i-time_laser_on)) #TODO this finds closest peak not the next one
    otrack_laser_st_latency[count_i] = i-time_laser_on[np.int64(laser_st_idx_match[count_i])]
otrack_laser_sw_latency = np.zeros(len(sw_on_otrack_trial_all_time))
laser_sw_idx_match = np.zeros(len(sw_on_otrack_trial_all_time))
for count_j, j in enumerate(sw_on_otrack_trial_all_time):
    laser_sw_idx_match[count_j] = np.argmin(np.abs(j-time_led_on)) #TODO this finds closest peak not the next one
    otrack_laser_sw_latency[count_j] = j-time_led_on[np.int64(laser_sw_idx_match[count_j])]
plt.figure()
plt.scatter(st_on_otrack_trial_all_time[np.int64(laser_st_idx_match)], otrack_laser_st_latency, color='orange')
plt.scatter(sw_on_otrack_trial_all_time[np.int64(laser_sw_idx_match)], otrack_laser_sw_latency, color='green')

# measure difference across time between peak otrack and st offline - same for through otrack and sw offline
trial = 2
from scipy.signal import find_peaks
otrack_x = otracks.loc[otracks['trial'] == trial, 'x']
otrack_time = np.array(otracks.loc[otracks['trial'] == trial, 'time'])
st_time = np.array(offtracks_st.loc[offtracks_st['trial'] == trial, 'time'])[:-1]
sw_time = np.array(offtracks_sw.loc[offtracks_sw['trial'] == trial, 'time'])[:-1]
peaks = find_peaks(otrack_x)
throughs = find_peaks(-otrack_x)
peaks_time = otrack_time[peaks[0]]
throughs_time = otrack_time[throughs[0]]
peaks_diff = np.zeros(len(peaks_time))
st_idx_match = np.zeros(len(peaks_time))
for count_i, i in enumerate(peaks_time):
    st_idx_match[count_i] = np.argmin(np.abs(st_time-i)) #TODO this finds closest peak not the next one
    peaks_diff[count_i] = i-st_time[np.int64(st_idx_match[count_i])]
throughs_diff = np.zeros(len(throughs_time))
sw_idx_match = np.zeros(len(throughs_time))
for count_j, j in enumerate(throughs_time):
    sw_idx_match[count_j] = np.argmin(np.abs(sw_time - j))
    throughs_diff[count_j] = j - sw_time[np.int64(sw_idx_match[count_j])]
plt.figure()
plt.scatter(st_time[np.int64(st_idx_match)], peaks_diff, color='green')
plt.scatter(sw_time[np.int64(sw_idx_match)], throughs_diff, color='orange')

p = 0
led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
        color='black')
ax.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'time'],
           otracks_st.loc[otracks_st['trial'] == trial, 'x'], color='orange')
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((led_trials[0, r], -400),
                              led_trials[1, r] - led_trials[0, r], 800, fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, :4, :])
ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
        color=paw_colors[p], linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

trial = 2
p = 0
led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
        color='black')
ax.scatter(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'],
           otracks_sw.loc[otracks_sw['trial'] == trial, 'x'], color='green')
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], -400),
                              timestamps_session[trial - 1][led_trials[1, r] - led_trials[0, r]], 800, fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, :4, :])
ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
        color=paw_colors[p], linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# # SETUP ACCURACY
# th_st_all = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200])
# th_sw_all = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
# st_correct_setup = np.zeros(len(trials))
# sw_correct_setup = np.zeros(len(trials))
# for count_t, trial in enumerate(trials):
#     th_st = th_st_all[count_t]
#     th_sw = th_sw_all[count_t]
#     [st_correct_trial, sw_correct_trial] = otrack_class.setup_accuracy(trial, otracks, otracks_st, otracks_sw, th_st, th_sw, 1)
#     st_correct_setup[count_t] = st_correct_trial
#     sw_correct_setup[count_t] = sw_correct_trial
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# rectangle1 = plt.Rectangle((10.5, 0), 2, 110, fc='dimgrey', alpha=0.3)
# rectangle2 = plt.Rectangle((22.5, 0), 2, 110, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle1)
# plt.gca().add_patch(rectangle2)
# ax.bar(trials, st_correct_setup, color='orange')
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\setup_accuracy_st.png')
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# rectangle1 = plt.Rectangle((10.5, 0), 2, 110, fc='dimgrey', alpha=0.3)
# rectangle2 = plt.Rectangle((22.5, 0), 2, 110, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle1)
# plt.gca().add_patch(rectangle2)
# ax.bar(trials, sw_correct_setup, color='green')
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\setup_accuracy_sw.png')
#
# # OTRACK EXAMPLE
# th_st = 200
# th_sw = 40
# trial = 18
# fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
# ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial']==trial, 'x'], color='black')
# ax.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'time'], otracks_st.loc[otracks_st['trial'] == trial, 'x'],
#            color='orange')
# ax.scatter(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'], otracks_sw.loc[otracks_sw['trial'] == trial, 'x'],
#            color='green')
# ax.axhline(th_st, color='orange')
# ax.axhline(th_sw, color='green')
# ax.set_title('trial' + str(trial))
# ax.set_ylim([-100, 300])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# # OTRACK ACCURACY
# [accuracy_st, accuracy_sw, precision_st, precision_sw, recall_st, recall_sw, f1_st,
#  f1_sw, fn_st, fn_sw, tn_st, tn_sw, fp_st, fp_sw, tp_st, tp_sw] = otrack_class.accuracy_scores_otrack(otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# for i in trials[:12]-1:
#     ax.scatter(0.5+np.random.rand(), accuracy_st[i], color='orange')
#     ax.scatter(2.5+np.random.rand(), accuracy_sw[i], color='green')
#     if i == 10 or i == 11:
#         ax.scatter(0.5 + np.random.rand(), accuracy_st[i], color='darkgrey')
#         ax.scatter(2.5 + np.random.rand(), accuracy_sw[i], color='darkgrey')
# for i in trials[12:]-1:
#     ax.scatter(4.5+np.random.rand(), accuracy_st[i], color='orange')
#     ax.scatter(6.5+np.random.rand(), accuracy_sw[i], color='green')
#     if i == 22 or i == 23:
#         ax.scatter(4.5 + np.random.rand(), accuracy_st[i], color='darkgrey')
#         ax.scatter(6.5 + np.random.rand(), accuracy_sw[i], color='darkgrey')
# ax.bar(1, np.nanmean(accuracy_st[trials[:12]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(3, np.nanmean(accuracy_sw[trials[:12]-1]), color='green', zorder=0, alpha=0.7)
# ax.bar(5, np.nanmean(accuracy_st[trials[12:]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(7, np.nanmean(accuracy_sw[trials[12:]-1]), color='green', zorder=0, alpha=0.7)
# ax.set_xticks([1, 3, 5, 7])
# ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
# ax.set_ylabel('Accuracy (%)', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_accuracy.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# for i in trials[:12]-1:
#     ax.scatter(0.5+np.random.rand(), f1_st[i], color='orange')
#     ax.scatter(2.5+np.random.rand(), f1_sw[i], color='green')
#     if i == 10 or i == 11:
#         ax.scatter(0.5 + np.random.rand(), f1_st[i], color='darkgrey')
#         ax.scatter(2.5 + np.random.rand(), f1_sw[i], color='darkgrey')
# for i in trials[12:]-1:
#     ax.scatter(4.5+np.random.rand(), f1_st[i], color='orange')
#     ax.scatter(6.5+np.random.rand(), f1_sw[i], color='green')
#     if i == 22 or i == 23:
#         ax.scatter(4.5 + np.random.rand(), f1_st[i], color='darkgrey')
#         ax.scatter(6.5 + np.random.rand(), f1_sw[i], color='darkgrey')
# ax.bar(1, np.nanmean(f1_st[trials[:12]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(3, np.nanmean(f1_sw[trials[:12]-1]), color='green', zorder=0, alpha=0.7)
# ax.bar(5, np.nanmean(f1_st[trials[12:]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(7, np.nanmean(f1_sw[trials[12:]-1]), color='green', zorder=0, alpha=0.7)
# ax.set_xticks([1, 3, 5, 7])
# ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
# ax.set_ylabel('F1 score (%)', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_f1.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# ax.bar(1, np.nanmean(tp_st[trials[:10]-1]), color='green')
# ax.bar(1, np.nanmean(tn_st[trials[:10]-1]), bottom = np.nanmean(tp_st[trials[:10]-1]), color='darkgreen')
# ax.bar(1, np.nanmean(fp_st[trials[:10]-1]), bottom = np.nanmean(tp_st[trials[:10]-1]) + np.nanmean(tn_st[trials[:10]-1]), color='red')
# ax.bar(1, np.nanmean(fn_st[trials[:10]-1]), bottom = np.nanmean(tp_st[trials[:10]-1]) + np.nanmean(tn_st[trials[:10]-1]) + np.nanmean(fp_st[trials[:10]-1]), color='crimson')
# ax.bar(2, np.nanmean(tp_st[trials[10:12]-1]), color='green')
# ax.bar(2, np.nanmean(tn_st[trials[10:12]-1]), bottom = np.nanmean(tp_st[trials[10:12]-1]), color='darkgreen')
# ax.bar(2, np.nanmean(fp_st[trials[10:12]-1]), bottom = np.nanmean(tp_st[trials[10:12]-1]) + np.nanmean(tn_st[trials[10:12]-1]), color='red')
# ax.bar(2, np.nanmean(fn_st[trials[10:12]-1]), bottom = np.nanmean(tp_st[trials[10:12]-1]) + np.nanmean(tn_st[trials[10:12]-1]) + np.nanmean(fp_st[trials[10:12]-1]), color='crimson')
# ax.bar(4, np.nanmean(tp_sw[trials[:10]-1]), color='green')
# ax.bar(4, np.nanmean(tn_sw[trials[:10]-1]), bottom = np.nanmean(tp_sw[trials[:10]-1]), color='darkgreen')
# ax.bar(4, np.nanmean(fp_sw[trials[:10]-1]), bottom = np.nanmean(tp_sw[trials[:10]-1]) + np.nanmean(tn_sw[trials[:10]-1]), color='red')
# ax.bar(4, np.nanmean(fn_sw[trials[:10]-1]), bottom = np.nanmean(tp_sw[trials[:10]-1]) + np.nanmean(tn_sw[trials[:10]-1]) + np.nanmean(fp_sw[trials[:10]-1]), color='crimson')
# ax.bar(5, np.nanmean(tp_sw[trials[10:12]-1]), color='green')
# ax.bar(5, np.nanmean(tn_sw[trials[10:12]-1]), bottom = np.nanmean(tp_sw[trials[10:12]-1]), color='darkgreen')
# ax.bar(5, np.nanmean(fp_sw[trials[10:12]-1]), bottom = np.nanmean(tp_sw[trials[10:12]-1]) + np.nanmean(tn_sw[trials[10:12]-1]), color='red')
# ax.bar(5, np.nanmean(fn_sw[trials[10:12]-1]), bottom = np.nanmean(tp_sw[trials[10:12]-1]) + np.nanmean(tn_sw[trials[10:12]-1]) + np.nanmean(fp_sw[trials[10:12]-1]), color='crimson')
# ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
# ax.set_xticks([1, 2, 4, 5])
# ax.set_xticklabels(['stance low th', 'stance low th split', 'swing low th', 'swing low th split'])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_low_th_performance.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# ax.bar(1, np.nanmean(tp_st[trials[12:22]-1]), color='green')
# ax.bar(1, np.nanmean(tn_st[trials[12:22]-1]), bottom = np.nanmean(tp_st[trials[12:22]-1]), color='darkgreen')
# ax.bar(1, np.nanmean(fp_st[trials[12:22]-1]), bottom = np.nanmean(tp_st[trials[12:22]-1]) + np.nanmean(tn_st[trials[12:22]-1]), color='red')
# ax.bar(1, np.nanmean(fn_st[trials[12:22]-1]), bottom = np.nanmean(tp_st[trials[12:22]-1]) + np.nanmean(tn_st[trials[12:22]-1]) + np.nanmean(fp_st[trials[12:22]-1]), color='crimson')
# ax.bar(2, np.nanmean(tp_st[trials[22:]-1]), color='green')
# ax.bar(2, np.nanmean(tn_st[trials[22:]-1]), bottom = np.nanmean(tp_st[trials[22:]-1]), color='darkgreen')
# ax.bar(2, np.nanmean(fp_st[trials[22:]-1]), bottom = np.nanmean(tp_st[trials[22:]-1]) + np.nanmean(tn_st[trials[22:]-1]), color='red')
# ax.bar(2, np.nanmean(fn_st[trials[22:]-1]), bottom = np.nanmean(tp_st[trials[22:]-1]) + np.nanmean(tn_st[trials[22:]-1]) + np.nanmean(fp_st[trials[22:]-1]), color='crimson')
# ax.bar(4, np.nanmean(tp_sw[trials[12:22]-1]), color='green')
# ax.bar(4, np.nanmean(tn_sw[trials[12:22]-1]), bottom = np.nanmean(tp_sw[trials[12:22]-1]), color='darkgreen')
# ax.bar(4, np.nanmean(fp_sw[trials[12:22]-1]), bottom = np.nanmean(tp_sw[trials[12:22]-1]) + np.nanmean(tn_sw[trials[12:22]-1]), color='red')
# ax.bar(4, np.nanmean(fn_sw[trials[12:22]-1]), bottom = np.nanmean(tp_sw[trials[12:22]-1]) + np.nanmean(tn_sw[trials[12:22]-1]) + np.nanmean(fp_sw[trials[12:22]-1]), color='crimson')
# ax.bar(5, np.nanmean(tp_sw[trials[22:]-1]), color='green')
# ax.bar(5, np.nanmean(tn_sw[trials[22:]-1]), bottom = np.nanmean(tp_sw[trials[22:]-1]), color='darkgreen')
# ax.bar(5, np.nanmean(fp_sw[trials[22:]-1]), bottom = np.nanmean(tp_sw[trials[22:]-1]) + np.nanmean(tn_sw[trials[22:]-1]), color='red')
# ax.bar(5, np.nanmean(fn_sw[trials[22:]-1]), bottom = np.nanmean(tp_sw[trials[22:]-1]) + np.nanmean(tn_sw[trials[22:]-1]) + np.nanmean(fp_sw[trials[22:]-1]), color='crimson')
# ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
# ax.set_xticks([1, 2, 4, 5])
# ax.set_xticklabels(['stance high th', 'stance high th split', 'swing high th', 'swing high th split'])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_high_th_performance.png')
#
# [tracks_hits_st, tracks_hits_sw, otrack_st_hits, otrack_sw_hits] = otrack_class.get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
# fig, ax = plt.subplots(tight_layout=True, sharey=True)
# rectangle1 = plt.Rectangle((10.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
# rectangle2 = plt.Rectangle((22.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle1)
# plt.gca().add_patch(rectangle2)
# for count_t, trial in enumerate(trials):
#     ax.bar(trials, np.array(otrack_st_hits), color='orange')
#     ax.bar(trials, [np.shape(offtracks_st.loc[offtracks_st['trial']==i])[0] for i in trials]-np.array(otrack_st_hits), bottom = np.array(otrack_st_hits), color='goldenrod')
# ax.set_xlabel('Trials', fontsize=16)
# ax.set_ylabel('Stance points', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_misses_stance.png')
# fig, ax = plt.subplots(tight_layout=True, sharey=True)
# rectangle1 = plt.Rectangle((10.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
# rectangle2 = plt.Rectangle((22.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle1)
# plt.gca().add_patch(rectangle2)
# for count_t, trial in enumerate(trials):
#     ax.bar(trials, np.array(otrack_sw_hits), color='green')
#     ax.bar(trials,
#               [np.shape(offtracks_sw.loc[offtracks_sw['trial'] == i])[0] for i in trials] - np.array(otrack_sw_hits),
#               bottom = np.array(otrack_sw_hits), color='darkolivegreen')
# ax.set_xlabel('Trials', fontsize=16)
# ax.set_ylabel('Swing points', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_misses_swing.png')
#
# # LASER ACCURACY
# tp_st_laser = np.zeros(len(trials))
# fp_st_laser = np.zeros(len(trials))
# tn_st_laser = np.zeros(len(trials))
# fn_st_laser = np.zeros(len(trials))
# precision_st_laser = np.zeros(len(trials))
# recall_st_laser = np.zeros(len(trials))
# f1_st_laser = np.zeros(len(trials))
# event = 'stance'
# for count_t, trial in enumerate(trials):
#     [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
#     tp_st_laser[count_t] = tp_trial
#     fp_st_laser[count_t] = fp_trial
#     tn_st_laser[count_t] = tn_trial
#     fn_st_laser[count_t] = fn_trial
#     precision_st_laser[count_t] = precision_trial
#     recall_st_laser[count_t] = recall_trial
#     f1_st_laser[count_t] = f1_trial
# tp_sw_laser = np.zeros(len(trials))
# fp_sw_laser = np.zeros(len(trials))
# tn_sw_laser = np.zeros(len(trials))
# fn_sw_laser = np.zeros(len(trials))
# precision_sw_laser = np.zeros(len(trials))
# recall_sw_laser = np.zeros(len(trials))
# f1_sw_laser = np.zeros(len(trials))
# event = 'swing'
# for count_t, trial in enumerate(trials):
#     [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, 0)
#     tp_sw_laser[count_t] = tp_trial
#     fp_sw_laser[count_t] = fp_trial
#     tn_sw_laser[count_t] = tn_trial
#     fn_sw_laser[count_t] = fn_trial
#     precision_sw_laser[count_t] = precision_trial
#     recall_sw_laser[count_t] = recall_trial
#     f1_sw_laser[count_t] = f1_trial
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# ax.bar(1, np.nanmean(tp_st_laser[trials[:10]-1]), color='green')
# ax.bar(1, np.nanmean(tn_st_laser[trials[:10]-1]), bottom = np.nanmean(tp_st_laser[trials[:10]-1]), color='darkgreen')
# ax.bar(1, np.nanmean(fp_st_laser[trials[:10]-1]), bottom = np.nanmean(tp_st_laser[trials[:10]-1]) + np.nanmean(tn_st_laser[trials[:10]-1]), color='red')
# ax.bar(1, np.nanmean(fn_st_laser[trials[:10]-1]), bottom = np.nanmean(tp_st_laser[trials[:10]-1]) + np.nanmean(tn_st_laser[trials[:10]-1]) + np.nanmean(fp_st_laser[trials[:10]-1]), color='crimson')
# ax.bar(2, np.nanmean(tp_st_laser[trials[10:12]-1]), color='green')
# ax.bar(2, np.nanmean(tn_st_laser[trials[10:12]-1]), bottom = np.nanmean(tp_st_laser[trials[10:12]-1]), color='darkgreen')
# ax.bar(2, np.nanmean(fp_st_laser[trials[10:12]-1]), bottom = np.nanmean(tp_st_laser[trials[10:12]-1]) + np.nanmean(tn_st_laser[trials[10:12]-1]), color='red')
# ax.bar(2, np.nanmean(fn_st_laser[trials[10:12]-1]), bottom = np.nanmean(tp_st_laser[trials[10:12]-1]) + np.nanmean(tn_st_laser[trials[10:12]-1]) + np.nanmean(fp_st_laser[trials[10:12]-1]), color='crimson')
# ax.bar(4, np.nanmean(tp_sw_laser[trials[:10]-1]), color='green')
# ax.bar(4, np.nanmean(tn_sw_laser[trials[:10]-1]), bottom = np.nanmean(tp_sw_laser[trials[:10]-1]), color='darkgreen')
# ax.bar(4, np.nanmean(fp_sw_laser[trials[:10]-1]), bottom = np.nanmean(tp_sw_laser[trials[:10]-1]) + np.nanmean(tn_sw_laser[trials[:10]-1]), color='red')
# ax.bar(4, np.nanmean(fn_sw_laser[trials[:10]-1]), bottom = np.nanmean(tp_sw_laser[trials[:10]-1]) + np.nanmean(tn_sw_laser[trials[:10]-1]) + np.nanmean(fp_sw_laser[trials[:10]-1]), color='crimson')
# ax.bar(5, np.nanmean(tp_sw_laser[trials[10:12]-1]), color='green')
# ax.bar(5, np.nanmean(tn_sw_laser[trials[10:12]-1]), bottom = np.nanmean(tp_sw_laser[trials[10:12]-1]), color='darkgreen')
# ax.bar(5, np.nanmean(fp_sw_laser[trials[10:12]-1]), bottom = np.nanmean(tp_sw_laser[trials[10:12]-1]) + np.nanmean(tn_sw_laser[trials[10:12]-1]), color='red')
# ax.bar(5, np.nanmean(fn_sw_laser[trials[10:12]-1]), bottom = np.nanmean(tp_sw_laser[trials[10:12]-1]) + np.nanmean(tn_sw_laser[trials[10:12]-1]) + np.nanmean(fp_sw_laser[trials[10:12]-1]), color='crimson')
# ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
# ax.set_xticks([1, 2, 4, 5])
# ax.set_xticklabels(['stance low th', 'stance low th split', 'swing low th', 'swing low th split'])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\laser_low_th_performance.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# ax.bar(1, np.nanmean(tp_st_laser[trials[12:22]-1]), color='green')
# ax.bar(1, np.nanmean(tn_st_laser[trials[12:22]-1]), bottom = np.nanmean(tp_st_laser[trials[12:22]-1]), color='darkgreen')
# ax.bar(1, np.nanmean(fp_st_laser[trials[12:22]-1]), bottom = np.nanmean(tp_st_laser[trials[12:22]-1]) + np.nanmean(tn_st_laser[trials[12:22]-1]), color='red')
# ax.bar(1, np.nanmean(fn_st_laser[trials[12:22]-1]), bottom = np.nanmean(tp_st_laser[trials[12:22]-1]) + np.nanmean(tn_st_laser[trials[12:22]-1]) + np.nanmean(fp_st_laser[trials[12:22]-1]), color='crimson')
# ax.bar(2, np.nanmean(tp_st_laser[trials[22:]-1]), color='green')
# ax.bar(2, np.nanmean(tn_st_laser[trials[22:]-1]), bottom = np.nanmean(tp_st_laser[trials[22:]-1]), color='darkgreen')
# ax.bar(2, np.nanmean(fp_st_laser[trials[22:]-1]), bottom = np.nanmean(tp_st_laser[trials[22:]-1]) + np.nanmean(tn_st_laser[trials[22:]-1]), color='red')
# ax.bar(2, np.nanmean(fn_st_laser[trials[22:]-1]), bottom = np.nanmean(tp_st_laser[trials[22:]-1]) + np.nanmean(tn_st_laser[trials[22:]-1]) + np.nanmean(fp_st_laser[trials[22:]-1]), color='crimson')
# ax.bar(4, np.nanmean(tp_sw_laser[trials[12:22]-1]), color='green')
# ax.bar(4, np.nanmean(tn_sw_laser[trials[12:22]-1]), bottom = np.nanmean(tp_sw_laser[trials[12:22]-1]), color='darkgreen')
# ax.bar(4, np.nanmean(fp_sw_laser[trials[12:22]-1]), bottom = np.nanmean(tp_sw_laser[trials[12:22]-1]) + np.nanmean(tn_sw_laser[trials[12:22]-1]), color='red')
# ax.bar(4, np.nanmean(fn_sw_laser[trials[12:22]-1]), bottom = np.nanmean(tp_sw_laser[trials[12:22]-1]) + np.nanmean(tn_sw_laser[trials[12:22]-1]) + np.nanmean(fp_sw_laser[trials[12:22]-1]), color='crimson')
# ax.bar(5, np.nanmean(tp_sw_laser[trials[22:]-1]), color='green')
# ax.bar(5, np.nanmean(tn_sw_laser[trials[22:]-1]), bottom = np.nanmean(tp_sw_laser[trials[22:]-1]), color='darkgreen')
# ax.bar(5, np.nanmean(fp_sw_laser[trials[22:]-1]), bottom = np.nanmean(tp_sw_laser[trials[22:]-1]) + np.nanmean(tn_sw_laser[trials[22:]-1]), color='red')
# ax.bar(5, np.nanmean(fn_sw_laser[trials[22:]-1]), bottom = np.nanmean(tp_sw_laser[trials[22:]-1]) + np.nanmean(tn_sw_laser[trials[22:]-1]) + np.nanmean(fp_sw_laser[trials[22:]-1]), color='crimson')
# ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
# ax.set_xticks([1, 2, 4, 5])
# ax.set_xticklabels(['stance high th', 'stance high th split', 'swing high th', 'swing high th split'])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\laser_high_th_performance.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# for i in trials[:12]-1:
#     ax.scatter(0.5+np.random.rand(), tp_st_laser[i] + tn_st_laser[i], color='orange')
#     ax.scatter(2.5+np.random.rand(), tp_sw_laser[i] + tn_sw_laser[i], color='green')
#     if i == 10 or i == 11:
#         ax.scatter(0.5 + np.random.rand(), tp_st_laser[i] + tn_st_laser[i], color='darkgrey')
#         ax.scatter(2.5 + np.random.rand(), tp_sw_laser[i] + tn_sw_laser[i], color='darkgrey')
# for i in trials[12:]-1:
#     ax.scatter(4.5+np.random.rand(), tp_st_laser[i] + tn_st_laser[i], color='orange')
#     ax.scatter(6.5+np.random.rand(), tp_sw_laser[i] + tn_sw_laser[i], color='green')
#     if i == 22 or i == 23:
#         ax.scatter(4.5 + np.random.rand(), tp_st_laser[i] + tn_st_laser[i], color='darkgrey')
#         ax.scatter(6.5 + np.random.rand(), tp_sw_laser[i] + tn_sw_laser[i], color='darkgrey')
# ax.bar(1, np.nanmean(tp_st_laser[trials[:12]-1]+tn_st_laser[trials[:12]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(3, np.nanmean(tp_sw_laser[trials[:12]-1]+tn_sw_laser[trials[:12]-1]), color='green', zorder=0, alpha=0.7)
# ax.bar(5, np.nanmean(tp_st_laser[trials[12:]-1]+tn_st_laser[trials[12:]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(7, np.nanmean(tp_sw_laser[trials[12:]-1]+tn_sw_laser[trials[12:]-1]), color='green', zorder=0, alpha=0.7)
# ax.set_xticks([1, 3, 5, 7])
# ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
# ax.set_ylabel('Accuracy (%)', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\laser_accuracy.png')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
# for i in trials[:12]-1:
#     ax.scatter(0.5+np.random.rand(), f1_st_laser[i], color='orange')
#     ax.scatter(2.5+np.random.rand(), f1_sw_laser[i], color='green')
#     if i == 10 or i == 11:
#         ax.scatter(0.5 + np.random.rand(), f1_st_laser[i], color='darkgrey')
#         ax.scatter(2.5 + np.random.rand(), f1_sw_laser[i], color='darkgrey')
# for i in trials[12:]-1:
#     ax.scatter(4.5+np.random.rand(), f1_st_laser[i], color='orange')
#     ax.scatter(6.5+np.random.rand(), f1_sw_laser[i], color='green')
#     if i == 22 or i == 23:
#         ax.scatter(4.5 + np.random.rand(), f1_st_laser[i], color='darkgrey')
#         ax.scatter(6.5 + np.random.rand(), f1_sw_laser[i], color='darkgrey')
# ax.bar(1, np.nanmean(f1_st_laser[trials[:12]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(3, np.nanmean(f1_sw_laser[trials[:12]-1]), color='green', zorder=0, alpha=0.7)
# ax.bar(5, np.nanmean(f1_st_laser[trials[12:]-1]), color='orange', zorder=0, alpha=0.7)
# ax.bar(7, np.nanmean(f1_sw_laser[trials[12:]-1]), color='green', zorder=0, alpha=0.7)
# ax.set_xticks([1, 3, 5, 7])
# ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
# ax.set_ylabel('F1 (%)', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('C:\\Users\\Ana\\Desktop\\laser_f1.png')
#
# # LASER ONSETS AND OFFSETS IN RELATION TO STRIDE
# light_onset_phase_tied_st_wide = []
# light_offset_phase_tied_st_wide = []
# light_onset_phase_split_st_wide = []
# light_offset_phase_split_st_wide = []
# light_onset_phase_tied_st_narrow = []
# light_offset_phase_tied_st_narrow = []
# light_onset_phase_split_st_narrow = []
# light_offset_phase_split_st_narrow = []
# event = 'stance'
# for trial in trials[:10]:
#     [light_onset_phase, light_offset_phase] = otrack_class.laser_presentation_phase(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_tied_st_wide.extend(light_onset_phase)
#     light_offset_phase_tied_st_wide.extend(light_offset_phase)
# for trial in trials[10:12]:
#     [light_onset_phase, light_offset_phase] = otrack_class.laser_presentation_phase(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_split_st_wide.extend(light_onset_phase)
#     light_offset_phase_split_st_wide.extend(light_offset_phase)
# for trial in trials[12:22]:
#     [light_onset_phase, light_offset_phase] = otrack_class.laser_presentation_phase(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_tied_st_narrow.extend(light_onset_phase)
#     light_offset_phase_tied_st_narrow.extend(light_offset_phase)
# for trial in trials[22:]:
#     [light_onset_phase, light_offset_phase] = otrack_class.laser_presentation_phase(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_split_st_narrow.extend(light_onset_phase)
#     light_offset_phase_split_st_narrow.extend(light_offset_phase)
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(np.array(light_onset_phase_tied_st_wide)*100, bins = 100, range=(-120, 120), color='orange')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied onset', fontsize=14)
# ax[0].set_xlabel('% stance', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(np.array(light_offset_phase_tied_st_wide)*100, bins = 100, range=(-120, 120), color='orange')
# ax[1].set_title('tied offset', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('% stance', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# ax[2].hist(np.array(light_onset_phase_split_st_wide)*100, bins = 100, range=(-120, 120), color='orange')
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].set_title('split onset', fontsize=14)
# ax[2].set_xlabel('% stance', fontsize=14)
# ax[2].tick_params(axis='both', which='major', labelsize=14)
# ax[3].hist(np.array(light_offset_phase_split_st_wide)*100, bins = 100, range=(-120, 120), color='orange')
# ax[3].set_title('split offset', fontsize=14)
# ax[3].spines['right'].set_visible(False)
# ax[3].spines['top'].set_visible(False)
# ax[3].set_xlabel('% stance', fontsize=14)
# ax[3].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\stance_stim_phase_low_threshold.png')
#
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(np.array(light_onset_phase_tied_st_narrow)*100, bins = 100, range=(-120, 120), color='orange')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied onset', fontsize=14)
# ax[0].set_xlabel('% stance', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(np.array(light_offset_phase_tied_st_narrow)*100, bins = 100, range=(-120, 120), color='orange')
# ax[1].set_title('tied offset', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('% stance', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# ax[2].hist(np.array(light_onset_phase_split_st_narrow)*100, bins = 100, range=(-120, 120), color='orange')
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].set_title('split onset', fontsize=14)
# ax[2].set_xlabel('% stance', fontsize=14)
# ax[2].tick_params(axis='both', which='major', labelsize=14)
# ax[3].hist(np.array(light_offset_phase_split_st_narrow)*100, bins = 100, range=(-120, 120), color='orange')
# ax[3].set_title('split offset', fontsize=14)
# ax[3].spines['right'].set_visible(False)
# ax[3].spines['top'].set_visible(False)
# ax[3].set_xlabel('% stance', fontsize=14)
# ax[3].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\stance_stim_phase_high_threshold.png')
#
# light_onset_phase_tied_sw_wide = []
# light_offset_phase_tied_sw_wide = []
# light_onset_phase_split_sw_wide = []
# light_offset_phase_split_sw_wide = []
# light_onset_phase_tied_sw_narrow = []
# light_offset_phase_tied_sw_narrow = []
# light_onset_phase_split_sw_narrow = []
# light_offset_phase_split_sw_narrow = []
# event = 'swing'
# for trial in trials[:10]:
#     [light_onset_phase, light_offset_phase] = otrack_class.light_presentation_phase(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_tied_sw_wide.extend(light_onset_phase)
#     light_offset_phase_tied_sw_wide.extend(light_offset_phase)
# for trial in trials[10:12]:
#     [light_onset_phase, light_offset_phase] = otrack_class.light_presentation_phase(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_split_sw_wide.extend(light_onset_phase)
#     light_offset_phase_split_sw_wide.extend(light_offset_phase)
# for trial in trials[12:22]:
#     [light_onset_phase, light_offset_phase] = otrack_class.light_presentation_phase(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_tied_sw_narrow.extend(light_onset_phase)
#     light_offset_phase_tied_sw_narrow.extend(light_offset_phase)
# for trial in trials[22:]:
#     [light_onset_phase, light_offset_phase] = otrack_class.light_presentation_phase(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session)
#     light_onset_phase_split_sw_narrow.extend(light_onset_phase)
#     light_offset_phase_split_sw_narrow.extend(light_offset_phase)
#
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(np.array(light_onset_phase_tied_sw_wide)*100, bins = 100, range=(-120, 120), color='green')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied onset', fontsize=14)
# ax[0].set_xlabel('% swing', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(np.array(light_offset_phase_tied_sw_wide)*100, bins = 100, range=(-120, 120), color='green')
# ax[1].set_title('tied offset', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('% swing', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# ax[2].hist(np.array(light_onset_phase_split_sw_wide)*100, bins = 100, range=(-120, 120), color='green')
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].set_title('split onset', fontsize=14)
# ax[2].set_xlabel('% swing', fontsize=14)
# ax[2].tick_params(axis='both', which='major', labelsize=14)
# ax[3].hist(np.array(light_offset_phase_split_sw_wide)*100, bins = 100, range=(-120, 120), color='green')
# ax[3].set_title('split offset', fontsize=14)
# ax[3].spines['right'].set_visible(False)
# ax[3].spines['top'].set_visible(False)
# ax[3].set_xlabel('% swing', fontsize=14)
# ax[3].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\swing_stim_phase_low_threshold.png')
#
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(np.array(light_onset_phase_tied_sw_narrow)*100, bins = 100, range=(-120, 120), color='green')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied onset', fontsize=14)
# ax[0].set_xlabel('% swing', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(np.array(light_offset_phase_tied_sw_narrow)*100, bins = 100, range=(-120, 120), color='green')
# ax[1].set_title('tied offset', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('% swing', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# ax[2].hist(np.array(light_onset_phase_split_sw_narrow)*100, bins = 100, range=(-120, 120), color='green')
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].set_title('split onset', fontsize=14)
# ax[2].set_xlabel('% swing', fontsize=14)
# ax[2].tick_params(axis='both', which='major', labelsize=14)
# ax[3].hist(np.array(light_offset_phase_split_sw_narrow)*100, bins = 100, range=(-120, 120), color='green')
# ax[3].set_title('split offset', fontsize=14)
# ax[3].spines['right'].set_visible(False)
# ax[3].spines['top'].set_visible(False)
# ax[3].set_xlabel('% swing', fontsize=14)
# ax[3].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\swing_stim_phase_high_threshold.png')
#
# #STIMULATION DURATION
# data_st_narrow_all = []
# data_st_wide_all = []
# data_sw_narrow_all = []
# data_sw_wide_all = []
# data_st_narrow_all_split = []
# data_st_wide_all_split = []
# data_sw_narrow_all_split = []
# data_sw_wide_all_split = []
# for i in trials[:10]:
#     data_st_wide_all.extend((laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000)
#     data_sw_wide_all.extend((sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000)
# for i in trials[10:12]:
#     data_st_wide_all_split.extend((laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000)
#     data_sw_wide_all_split.extend((sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000)
# for i in trials[12:22]:
#     data_st_narrow_all.extend((laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000)
#     data_sw_narrow_all.extend((sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000)
# for i in trials[22:]:
#     data_st_narrow_all_split.extend((laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000)
#     data_sw_narrow_all_split.extend((sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000)
# fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(data_st_wide_all, bins = 100, range=(0, 200), color='orange')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied', fontsize=14)
# ax[0].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(data_st_wide_all_split, bins = 100, range=(0, 200), color='orange')
# ax[1].set_title('split', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\stance_stim_duration_low_threshold.png')
#
# fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(data_sw_wide_all, bins = 100, range=(0, 200), color='green')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied', fontsize=14)
# ax[0].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(data_sw_wide_all_split, bins = 100, range=(0, 200), color='green')
# ax[1].set_title('split', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\swing_stim_duration_low_threshold.png')
#
# fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(data_st_narrow_all, bins = 100, range=(0, 200), color='orange')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_title('tied', fontsize=14)
# ax[0].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(data_st_narrow_all_split, bins = 100, range=(0, 200), color='orange')
# ax[1].set_title('split', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\stance_stim_duration_high_threshold.png')
#
# fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 5))
# ax = ax.ravel()
# ax[0].hist(data_sw_narrow_all, bins = 100, range=(0, 200), color='green')
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[0].set_title('tied', fontsize=14)
# ax[0].tick_params(axis='both', which='major', labelsize=14)
# ax[1].hist(data_sw_narrow_all_split, bins = 100, range=(0, 200), color='green')
# ax[1].set_title('split', fontsize=14)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].set_xlabel('stimulus duration (ms)', fontsize=14)
# ax[1].tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('C:\\Users\\Ana\\Desktop\\swing_stim_duration_high_threshold.png')
#
#
