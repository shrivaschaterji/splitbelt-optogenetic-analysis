# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks

paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Online Tracking Treadmill\\NetworkTest_280223\\'
subdir = 'Full Network\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MFullNetwork'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()
# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[timestamps_session, frame_counter_session, frame_counter_0_session, timestamps_0_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[trial_signal_session, sync_signal_session] = otrack_class.get_synchronizer_data(plot_data)

# READ ONLINE DLC TRACKS
[otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(frame_counter_0_session, timestamps_0_session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

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
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].scatter(np.repeat(trial*2, len(latency_st[count_t]))+np.random.rand(len(latency_st[count_t])), latency_st[count_t], color='black')
ax[0].set_xticks(trials*2+0.5)
ax[0].set_xticklabels(labels=map(str, trials))
ax[0].set_title('stance')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].scatter(np.repeat(trial*2, len(latency_sw[count_t]))+np.random.rand(len(latency_sw[count_t])), latency_sw[count_t], color='black')
ax[1].set_xticks(trials*2+0.5)
ax[1].set_xticklabels(labels=map(str, trials))
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_title('swing')
# misses plot
fig, ax = plt.subplots(2, 1, tight_layout=True, sharey=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, np.array(otrack_st_miss), color='black')
ax[0].set_title('stance')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, np.array(otrack_sw_miss), color='black')
ax[1].set_title('swing')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
# frames poorly detected
[detected_frames_bad_st, detected_frames_bad_sw] = otrack_class.frames_outside_st_sw(trials, offtracks_st, offtracks_sw, otracks_st, otracks_sw)
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, detected_frames_bad_st, color='black')
ax[0].set_title('stance')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, detected_frames_bad_sw, color='black')
ax[1].set_title('swing')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

# LATENCY OF LIGHT IN RELATION TO OTRACK
#don't know which light is pointing to what
latency_light_st = []
latency_light_sw = []
for trial in trials:
    [latency_trial_st, latency_trial_sw, st_led_frames, sw_led_frames ] = otrack_class.measure_light_on_videos(trial, timestamps_session, otracks_st, otracks_sw)
    latency_light_st.append(latency_trial_st)
    latency_light_sw.append(latency_trial_sw)
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].scatter(np.repeat(trial*2, len(latency_light_st[count_t]))+np.random.rand(len(latency_light_st[count_t])), latency_light_st[count_t], color='black')
ax[0].set_xticks(trials*2+0.5)
ax[0].set_xticklabels(labels=map(str, trials))
ax[0].set_title('stance')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].scatter(np.repeat(trial*2, len(latency_light_sw[count_t]))+np.random.rand(len(latency_light_sw[count_t])), latency_light_sw[count_t], color='black')
ax[1].set_xticks(trials*2+0.5)
ax[1].set_xticklabels(labels=map(str, trials))
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_title('swing')

# PERIODS WITH LIGHT ON
paw_colors = ['red', 'magenta', 'blue', 'cyan']
fig, ax = plt.subplots(tight_layout=True)
for r in range(np.shape(st_led_frames)[1]):
    ax.axvline(timestamps_session[trial-1][st_led_frames[0, r]], color='lightgray')
    ax.axvline(timestamps_session[trial - 1][st_led_frames[1, r]], color='lightgray')
for p in range(4):
    ax.plot(timestamps_session[trial-1], final_tracks_trials[trial-1][0, p, :-1], color=paw_colors[p])
ax.set_title('light on stance')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
paw_colors = ['red', 'magenta', 'blue', 'cyan']
fig, ax = plt.subplots(tight_layout=True)
for r in range(np.shape(sw_led_frames)[1]):
    ax.axvline(timestamps_session[trial-1][sw_led_frames[0, r]], color='lightgray')
    ax.axvline(timestamps_session[trial - 1][sw_led_frames[1, r]], color='lightgray')
for p in range(4):
    ax.plot(timestamps_session[trial-1], final_tracks_trials[trial-1][0, p, :-1], color=paw_colors[p])
ax.set_title('light on swing')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# TODO measure the time that the light was on and if that period passed to the next period (either stance or swing)

# # READ MP4 AND OVERLAY OFFLINE AND ONLINE DLC TRACKS
# trial = 3
# otrack_class.overlay_tracks_video(trial, paw_otrack, offtracks_st, offtracks_sw, otracks_st, otracks_sw)


