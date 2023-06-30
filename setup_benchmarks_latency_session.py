# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

paw_otrack = 'FR'
path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\Tailbase tests\\75percent\\'
th_st_all = np.repeat(165, 10) 
th_sw_all = np.repeat(215, 10)
animals = ['MC18089', 'MC18090', 'MC18091']
session = 1
condition = path.split('\\')[-2]
main_dir = path.split('\\')[:-2]
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))
th_st_cross_on = []
th_st_cross_off = []
th_st_detect_on = []
th_st_detect_off = []
th_st_laser_on = []
th_st_laser_off = []
th_st_on = []
th_st_off = []
animal_st_id = []
trial_st_id = []
condition_st_id = []
th_sw_cross_on = []
th_sw_cross_off = []
th_sw_detect_on = []
th_sw_detect_off = []
th_sw_laser_on = []
th_sw_laser_off = []
th_sw_on = []
th_sw_off = []
animal_sw_id = []
trial_sw_id = []
condition_sw_id = []
for animal in animals:
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)
    # LOAD DATA FOR BENCHMARK ANALYSIS
    [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)
    # GET BENCHMARK DATA STANCE
    [condition_id_st_singleanimal, trial_id_st_singleanimal, animal_id_st_singleanimal, th_cross_on_st_singleanimal,
     th_cross_off_st_singleanimal, th_detect_on_st_singleanimal, th_detect_off_st_singleanimal, th_laser_on_st_singleanimal,
     th_laser_off_st_singleanimal, th_st_on_st_singleanimal, th_st_off_st_singleanimal] = otrack_class.get_benchmark_data_laser('stance',
        th_st_all, condition, animal, otracks, otracks_st, otracks_sw, laser_on, offtracks_st, offtracks_sw)
    th_st_cross_on.extend(th_cross_on_st_singleanimal)
    th_st_cross_off.extend(th_cross_off_st_singleanimal)
    th_st_detect_on.extend(th_detect_on_st_singleanimal)
    th_st_detect_off.extend(th_detect_off_st_singleanimal)
    th_st_laser_on.extend(th_laser_on_st_singleanimal)
    th_st_laser_off.extend(th_laser_off_st_singleanimal)
    th_st_on.extend(th_st_on_st_singleanimal)
    th_st_off.extend(th_st_off_st_singleanimal)
    animal_st_id.extend(animal_id_st_singleanimal)
    trial_st_id.extend(trial_id_st_singleanimal)
    condition_st_id.extend(condition_id_st_singleanimal)
    # GET BENCHMARK DATA SWING
    [condition_id_sw_singleanimal, trial_id_sw_singleanimal, animal_id_sw_singleanimal, th_cross_on_sw_singleanimal,
     th_cross_off_sw_singleanimal, th_detect_on_sw_singleanimal, th_detect_off_sw_singleanimal, th_laser_on_sw_singleanimal,
     th_laser_off_sw_singleanimal, th_sw_on_sw_singleanimal, th_sw_off_sw_singleanimal] = otrack_class.get_benchmark_data_led('swing',
        th_sw_all, condition, animal, otracks, otracks_st, otracks_sw, st_led_on, sw_led_on, offtracks_st, offtracks_sw)
    th_sw_cross_on.extend(th_cross_on_sw_singleanimal)
    th_sw_cross_off.extend(th_cross_off_sw_singleanimal)
    th_sw_detect_on.extend(th_detect_on_sw_singleanimal)
    th_sw_detect_off.extend(th_detect_off_sw_singleanimal)
    th_sw_laser_on.extend(th_laser_on_sw_singleanimal)
    th_sw_laser_off.extend(th_laser_off_sw_singleanimal)
    th_sw_on.extend(th_sw_on_sw_singleanimal)
    th_sw_off.extend(th_sw_off_sw_singleanimal)
    animal_sw_id.extend(animal_id_sw_singleanimal)
    trial_sw_id.extend(trial_id_sw_singleanimal)
    condition_sw_id.extend(condition_id_sw_singleanimal)
benchmark_data_st = pd.DataFrame(
    {'condition': condition_st_id, 'animal': animal_st_id, 'trial': trial_st_id, 'th_cross_on': th_st_cross_on,
     'th_cross_off': th_st_cross_off, 'th_detect_on': th_st_detect_on, 'th_detect_off': th_st_detect_off,
     'th_laser_on': th_st_laser_on, 'th_laser_off': th_st_laser_off, 'th_st_on': th_st_on, 'th_st_off': th_st_off})
benchmark_data_st.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_data_st.csv'), sep=',', index=False)

benchmark_data_sw = pd.DataFrame(
    {'condition': condition_sw_id, 'animal': animal_sw_id, 'trial': trial_sw_id, 'th_cross_on': th_sw_cross_on,
     'th_cross_off': th_sw_cross_off, 'th_detect_on': th_sw_detect_on, 'th_detect_off': th_sw_detect_off,
     'th_laser_on': th_sw_laser_on, 'th_laser_off': th_sw_laser_off, 'th_sw_on': th_sw_on, 'th_sw_off': th_sw_off})
benchmark_data_sw.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_data_sw.csv'), sep=',', index=False)

cmap = plt.get_cmap('rainbow')
color_speeds = [cmap(i) for i in np.linspace(0, 1, 6)]
trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
def get_colors_plot(trial, color_speeds):
    color_idx = np.where(trials_reshape==trial)[0]
    color_plot = color_speeds[color_idx[0]]
    return color_plot

# Latency summary
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_st_animal = benchmark_data_st.loc[benchmark_data_st['animal'] == a]
    for trial in trials:
        latency_cross_true_st = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_detect_on'] - \
                                benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_cross_on']
        latency_cross_laser_st = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_laser_on'] - \
                                 benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_cross_on']
        if (trial % 2) == 0:
            plot_scatter = ax.plot(np.array([1, 2]), np.array([np.nanmean(latency_cross_true_st), np.nanmean(latency_cross_laser_st)]), 'o-',
                    color=get_colors_plot(trial, color_speeds), label='_nolegend_')
        else:
            plot_scatter = ax.plot(np.array([1, 2]), np.array([np.nanmean(latency_cross_true_st), np.nanmean(latency_cross_laser_st)]), 'o-',
                    color=get_colors_plot(trial, color_speeds), label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xticks(np.array([1, 2]))
ax.set_xticklabels(['latency threshold\n cross and detection', 'latency threshold\n cross and light on'], fontsize=10)
ax.set_ylabel('Latency (s)', fontsize=14)
ax.set_title('latency stance ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'latency_summary_stance_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_sw_animal = benchmark_data_sw.loc[benchmark_data_sw['animal'] == a]
    for trial in trials:
        latency_cross_true_sw = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_detect_on'] - \
                                benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_cross_on']
        latency_cross_laser_sw = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_laser_on'] - \
                                 benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_cross_on']
        if (trial % 2) == 0:
            plot_scatter = ax.plot(np.array([1, 2]), np.array([np.nanmean(latency_cross_true_sw), np.nanmean(latency_cross_laser_sw)]), 'o-',
                    color=get_colors_plot(trial, color_speeds), label='_nolegend_')
        else:
            plot_scatter = ax.plot(np.array([1, 2]), np.array([np.nanmean(latency_cross_true_sw), np.nanmean(latency_cross_laser_sw)]), 'o-',
                    color=get_colors_plot(trial, color_speeds), label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xticks(np.array([1, 2]))
ax.set_xticklabels(['latency threshold\n cross and detection', 'latency threshold\n cross and light on'], fontsize=10)
ax.set_ylabel('Latency (s)', fontsize=14)
ax.set_title('latency swing ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'latency_summary_swing_' + condition), dpi=128)

# Stance/swing duration relation with the duration of threshold crossing
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_st_animal = benchmark_data_st.loc[benchmark_data_st['animal'] == a]
    for trial in trials:
        th_cross_duration_st = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_cross_off'] - \
                                benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_cross_on']
        st_duration = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_st_off'] - \
                                 benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_st_on']
        if (trial % 2) == 0:
            plot_scatter = ax.scatter(np.nanmean(th_cross_duration_st), np.nanmean(st_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label='_nolegend_')
        else:
            plot_scatter = ax.scatter(np.nanmean(th_cross_duration_st), np.nanmean(st_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xlabel('threshold cross duration (s)', fontsize=14)
ax.set_ylabel('stance duration (s)', fontsize=14)
ax.set_title('threshold crossing duration vs\nstance duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'thcrossduration_stanceduration_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_sw_animal = benchmark_data_sw.loc[benchmark_data_sw['animal'] == a]
    for trial in trials:
        th_cross_duration_sw = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_cross_off'] - \
                                benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_cross_on']
        sw_duration = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_sw_off'] - \
                                 benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_sw_on']
        if (trial % 2) == 0:
            plot_scatter = ax.scatter(np.nanmean(th_cross_duration_sw), np.nanmean(sw_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label='_nolegend_')
        else:
            plot_scatter = ax.scatter(np.nanmean(th_cross_duration_sw), np.nanmean(sw_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xlabel('threshold cross duration (s)', fontsize=14)
ax.set_ylabel('swing duration (s)', fontsize=14)
ax.set_title('threshold crossing duration vs\nswing duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'thcrossduration_swingduration_' + condition), dpi=128)

# Stance/swing duration relation with the duration of laser presentation
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_st_animal = benchmark_data_st.loc[benchmark_data_st['animal'] == a]
    for trial in trials:
        laser_duration_st = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_laser_off'] - \
                                benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_laser_on']
        st_duration = benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_st_off'] - \
                                 benchmark_data_st_animal.loc[benchmark_data_st_animal['trial'] == trial, 'th_st_on']
        if (trial % 2) == 0:
            plot_scatter = ax.scatter(np.nanmean(laser_duration_st), np.nanmean(st_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label='_nolegend_')
        else:
            plot_scatter = ax.scatter(np.nanmean(laser_duration_st), np.nanmean(st_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xlabel('laser duration (s)', fontsize=14)
ax.set_ylabel('stance duration (s)', fontsize=14)
ax.set_title('laser duration vs\nstance duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'laserduration_stanceduration_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for a in animals:
    benchmark_data_sw_animal = benchmark_data_sw.loc[benchmark_data_sw['animal'] == a]
    for trial in trials:
        laser_duration_sw = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_laser_off'] - \
                                benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_laser_on']
        sw_duration = benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_sw_off'] - \
                                 benchmark_data_sw_animal.loc[benchmark_data_sw_animal['trial'] == trial, 'th_sw_on']
        if (trial % 2) == 0:
            plot_scatter = ax.scatter(np.nanmean(laser_duration_sw), np.nanmean(sw_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label='_nolegend_')
        else:
            plot_scatter = ax.scatter(np.nanmean(laser_duration_sw), np.nanmean(sw_duration),
                    color=get_colors_plot(trial, color_speeds), s=60, label=str(trial))
lgnd = ax.legend(['0.175', '0.275', '0.375', 'ipsi fast', 'left fast'], frameon=False, fontsize=12, loc='upper center', ncol=3)
for i in range(len(color_speeds)-1):
    lgnd.legendHandles[i]._sizes = [30]
ax.set_xlabel('laser duration (s)', fontsize=14)
ax.set_ylabel('swing duration (s)', fontsize=14)
ax.set_title('laser duration vs\nswing duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'laserduration_swingduration_' + condition), dpi=128)
plt.close('all')

# #obs: different fps in otrack trial, synchronizer, LED on measurements
# # Sync sampling rate
# [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(
#     animal, 0)
# [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session,
#  laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, animal, 0)
# fps_sync = (len(trial_signal_session.loc[trial_signal_session['trial'] == 1, 'time'])-2)/trial_signal_session.loc[trial_signal_session['trial'] == 1, 'time'].iloc[-1]
