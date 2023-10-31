import os
import numpy as np
np.warnings.filterwarnings('ignore')

path = 'J:\\Data OPTO\\CM tests\\75percent\\'
condition = path.split('\\')[-2]
animals = ['MC18089', 'MC18090', 'MC18091']
colors_animals = ['black', 'teal', 'orange']
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

animal = 'MC18089'
trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
# LOAD DATA FOR BENCHMARK ANALYSIS
[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
# READ OFFLINE PAW EXCURSIONS
[final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                             'st-sw-st')
time_bool = 0
norm_stim = 1
norm_stride = 0
fontsize_plot = 16
trials_plot = np.array([5, 6])
light_onset_phase_st_all = []
light_offset_phase_st_all = []
stim_nr_trials = np.zeros(len(trials_plot))
stride_nr_trials = np.zeros(len(trials_plot))
for count_t, trial in enumerate(trials_plot):
    [light_onset_phase_st, light_offset_phase_st, stim_nr, stride_nr] = \
        otrack_class.laser_presentation_phase(trial, trials, 'stance', offtracks_st, offtracks_sw, laser_on,
        timestamps_session, final_tracks_phase, time_bool)
    stim_nr_trials[count_t] = stim_nr
    stride_nr_trials[count_t] = stride_nr
    light_onset_phase_st_all.extend(light_onset_phase_st)
    light_offset_phase_st_all.extend(light_offset_phase_st)
otrack_class.plot_laser_presentation_phase(light_onset_phase_st_all, light_offset_phase_st_all, 'stance', fontsize_plot,
            np.sum(stim_nr_trials), np.sum(stride_nr_trials), norm_stim, norm_stride)
light_onset_phase_sw_all = []
light_offset_phase_sw_all = []
stim_nr_trials = np.zeros(len(trials_plot))
stride_nr_trials = np.zeros(len(trials_plot))
for count_t, trial in enumerate(trials_plot):
    [light_onset_phase_sw, light_offset_phase_sw, stim_nr, stride_nr] = \
        otrack_class.light_presentation_phase(trial, trials, 'swing', offtracks_st, offtracks_sw, st_led_on, sw_led_on,
                            timestamps_session, final_tracks_phase, time_bool)
    stim_nr_trials[count_t] = stim_nr
    stride_nr_trials[count_t] = stride_nr
    light_onset_phase_sw_all.extend(light_onset_phase_sw)
    light_offset_phase_sw_all.extend(light_offset_phase_sw)
otrack_class.plot_laser_presentation_phase(light_onset_phase_sw_all, light_offset_phase_sw_all, 'swing', fontsize_plot,
            np.sum(stim_nr_trials), np.sum(stride_nr_trials), norm_stim, norm_stride)

