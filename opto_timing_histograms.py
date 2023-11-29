import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

path = 'J:\\Opto Benchmarks\\CM tests\\75percent\\'
condition = path.split('\\')[-2]
animals = ['MC18089', 'MC18090', 'MC18091', 'VIV40922', 'VIV40923', 'VIV40924']
colors_animals = ['blue', 'orange', 'darkgreen', 'crimson', 'purple', 'gray']
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

animal = 'VIV40922'
trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
 laser_on] = otrack_class.load_processed_files(animal)
# LOAD DATA FOR BENCHMARK ANALYSIS
[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
# READ OFFLINE PAW EXCURSIONS
[final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                             'st-sw-st')
event = 'swing'
trial = 3
[light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                otrack_class.light_presentation_phase_all(trial, trials, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on,
                                                      timestamps_session, final_tracks_phase, 0)

# Plot timing as step-like histograms
onset_data = light_onset_phase_st_trial
offset_data = light_offset_phase_st_trial
fontsize_plot = 20
hist_onset = np.histogram(onset_data, range=(
        np.min(onset_data), np.max(onset_data)))
hist_offset = np.histogram(offset_data, range=(
        np.min(offset_data), np.max(offset_data)))
amp_plot = np.max([hist_onset[0], hist_offset[0]])/2
time = np.arange(-1, 2, np.round(1 / loco.sr, 3))
FR = amp_plot * np.sin(2 * np.pi * time + (np.pi / 2))+amp_plot
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(time, FR, color='lightgray', zorder=0)
ax.hist(onset_data, histtype='step', color='black', linewidth=4)
ax.hist(offset_data, histtype='step', color='dimgray', linewidth=4)
ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])
ax.set_xticklabels(['-100', '-50', '0', '50', '100', '150', '200'])
ax.set_xlabel('Phase (%)', fontsize=fontsize_plot)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fontsize_plot - 2)

# Plot timing as histograms with median duration
otrack_class.plot_laser_presentation_phase(light_onset_phase_st_trial, light_offset_phase_st_trial,
                                           event, 16, np.array(stim_nr_st), np.array(stride_nr_st), 1, 0,
                                           'J:\\Opto Benchmarks', 'test_all', 0)

