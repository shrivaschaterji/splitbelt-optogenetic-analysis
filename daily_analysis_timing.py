import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'stance'
if laser_event == 'stance':
    color_cond = 'orange'
if laser_event == 'swing':
    color_cond = 'green'
trials_plot = np.arange(9, 19) #trials with stimulation to check phase of laser
path = 'J:\\Opto ChR2 RT\\20240621 tied stance onset stim th200st\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

light_onset_phase_animals_hist = []
light_offset_phase_animals_hist = []
for count_a, animal in enumerate(animal_list):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)

    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    # LASER ACCURACY
    tp_laser = np.zeros(len(trials_plot))
    fp_laser = np.zeros(len(trials_plot))
    tn_laser = np.zeros(len(trials_plot))
    fn_laser = np.zeros(len(trials_plot))
    precision_laser = np.zeros(len(trials_plot))
    recall_laser = np.zeros(len(trials_plot))
    f1_laser = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        fp_laser[count_t] = fp_trial
        tn_laser[count_t] = tn_trial
        fn_laser[count_t] = fn_trial
        precision_laser[count_t] = precision_trial
        recall_laser[count_t] = recall_trial
        f1_laser[count_t] = f1_trial

    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    ax.plot(trials_plot, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(animal, fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    plt.savefig(path_save + animal + '_laser_performance_accuracy.png')

    #LASER ONSET AND OFFSET PHASE
    light_onset_phase_all = []
    light_offset_phase_all = []
    stim_nr_trials = np.zeros(len(trials_plot))
    stride_nr_trials = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
            otrack_class.laser_presentation_phase_all(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                  timestamps_session, final_tracks_phase, "FR")
        stim_nr_trials[count_t] = stim_nr
        stride_nr_trials[count_t] = stride_nr
        light_onset_phase_all.extend(light_onset_phase)
        light_offset_phase_all.extend(light_offset_phase)
    otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_all, light_offset_phase_all,
                    16, path_save, animal+'_'+laser_event+'_session_'+session_list[count_a], True)
    light_onset_phase_animals_hist.append(light_onset_phase_all)
    light_offset_phase_animals_hist.append(light_offset_phase_all)
    plt.close('all')

if color_cond == 'green':
    color_onset = 'green'
    color_offset = 'lightgreen'
if color_cond == 'orange':
    color_onset = 'orange'
    color_offset = 'gold'
amp_plot = 400
time = np.arange(-0.5, 1.5, np.round(1 / loco.sr, 3))
FR = amp_plot * np.sin(2 * np.pi * time + (np.pi / 2)) + amp_plot
fontsize_plot = 22
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(time, FR, color='lightgray', zorder=0)
for count_a in range(len(light_onset_phase_animals_hist)):
    hist_onset = np.histogram(light_onset_phase_animals_hist[count_a], range=(
        np.min(light_onset_phase_animals_hist[count_a]), np.max(light_onset_phase_animals_hist[count_a])), bins=20)
    hist_offset = np.histogram(light_offset_phase_animals_hist[count_a], range=(
        np.min(light_offset_phase_animals_hist[count_a]), np.max(light_offset_phase_animals_hist[count_a])), bins=20)
    # weights_onset = np.ones_like(light_onset_phase_animals[count_a]) / np.max(hist_onset[0])
    # weights_offset = np.ones_like(light_offset_phase_animals[count_a]) / np.max(hist_offset[0])
    # ax.hist(light_onset_phase_animals[count_a], histtype='step', color=color_onset, alpha=1-(count_a)*0.2, linewidth=2, weights=weights_onset)
    # ax.hist(light_offset_phase_animals[count_a], histtype='step', color=color_offset, alpha=1-(count_a)*0.2, linewidth=2, weights=weights_offset)
    ax.hist(light_onset_phase_animals_hist[count_a], histtype='step', color=color_onset, alpha=1-(count_a)*0.1, linewidth=2)
    ax.hist(light_offset_phase_animals_hist[count_a], histtype='step', color=color_offset, alpha=1-(count_a)*0.1, linewidth=2)
ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
ax.set_xticklabels(['-50', '0', '50', '100', '150'])
ax.set_xlabel('Stride phase (%)', fontsize=fontsize_plot)
ax.set_ylabel('Laser presentation\ncounts', fontsize=fontsize_plot)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fontsize_plot - 2)
plt.savefig(path_save + 'all_animals_hist_' + laser_event + '.png', dpi=256)
plt.savefig(path_save + 'all_animals_hist_' + laser_event + '.svg', dpi=256)


