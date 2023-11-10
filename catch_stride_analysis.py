import numpy as np
import matplotlib.pyplot as plt
import os
import online_tracking_class

import locomotion_class

# Inputs
laser_event = 'stance'
trials_plot = np.arange(9, 19) #trials with stimulation to check phase of laser
path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\singletrial\\20230607 tied swing stim\\'
included_animal_list = ['MC17670'] 
Ntrials = 24   
stim_start = 9
split_start = 9    
stim_duration = 8  
split_duration = 8 


otrack_class = online_tracking_class.otrack_class(path)
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
if len(included_animal_list) == 0:              # All animals included
    included_animal_list = animal_list
included_animals_id = [animal_list.index(i) for i in included_animal_list]
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

for count_a, animal in enumerate(included_animal_list):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)

    # EXTRACT REBOUND TIMES for EACH TRIAL in the stimulation phase
    rebound_times = []          # List of rebound times in each trial - len(rebound_times)=stim_duration
    for trial in range(stim_start,stim_start+stim_duration):
        # Select current trial
        current_trial_laser = laser_on.loc[laser_on['trial'] == trial]
        rebound_times.append(np.array(current_trial_laser['time_off'])*1000)        # [ms]

    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    
    # CATCH STRIDE ANALYSIS
    # Compute continuous asymmetry measures in each stride
    param_sym_name = ['coo', 'step_length', 'double_support']
    
    # CONTINUOUS STEP LENGTH WITH LASER ON
    trials = np.arange(1, Ntrials+1)
    
    for ps in range(len(param_sym_name)):
        st_strides_trials = []          # For each trial, contains the st_strides_mat with stridesx2x5 matrices for each paw; columns are: st/sw in ms; x(st/sw); y(st/sw); z(st/sw); st idx/sw idx. 2 middle columns for beginning and end of stride
        rebound_strides_trials = []     # For each trial, contains a boolean array saying which strides have rebound and which do not        
        current_param_sym = []
        session = int(session_list[count_a])
        filelist = loco.get_track_files(animal, session)
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_sym_name[ps])
            current_param_sym.append(param_mat)
            st_strides_trials.append(st_strides_mat)
            # If we are in stim trials, check for strides containing a rebound spike (at laser offset) 
            if count_trial>=stim_start-1 and count_trial<stim_start+stim_duration-1:
                print(count_trial, count_trial-stim_start+1)
                within_interval = (np.array(st_strides_mat[0])[:,0,0][:,np.newaxis] <= rebound_times[count_trial-stim_start+1]) & (rebound_times[count_trial-stim_start+1] <= np.array(st_strides_mat[0])[:,1,0][:,np.newaxis])
                rebound_strides_array = np.any(within_interval, axis=1)         # boolean array with shape = stridesx1
                rebound_strides_trials.append(rebound_strides_array)
        [stride_idx, trial_continuous, param_sym_time, param_sym_values] = loco.param_continuous_sym(current_param_sym, st_strides_trials, trials, 'FR', 'FL', 1, 0)

        #cumulative_idx_array, cumulative_trial, param_all_time_notnan, param_all_notnan

        #for trial_stim in range(stim_start-1, stim_start+stim_duration):
        rebound_strides_trials_continuous = np.zeros(param_sym_time.shape,dtype=bool)
        offset_index = np.where(np.array(trial_continuous)==stim_start)[0][0]
        set to true the ones that are true in rebound_strides_trials
        param_sym_values_stim = param_sym_values[rebound_strides_trials_continuous]
        param_sym_values_stim_no_stim = param_sym_values[~rebound_strides_trials_continuous]

        
        fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
        sl_time_start = param_sym_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
        sl_time_duration = param_sym_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(loco.trial_time*stim_duration)
        rectangle = plt.Rectangle((sl_time_start, np.nanmin(param_sym_values)), sl_time_duration-sl_time_start, np.nanmax(param_sym_values)+np.abs(np.nanmin(param_sym_values)), fc='g', alpha=0.3)
        plt.gca().add_patch(rectangle)
        ax.plot(param_sym_time, param_sym_values, color='black')
        ax.set_xlabel('time (s)')
        ax.set_title('continuous '+param_sym_name[ps])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + animal_list[a] + param_sym_name[ps]+'_continuous', dpi=96)

        fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
        plt.gca().add_patch(rectangle)
        ax.plot(param_sym_time, param_sym_values_stim, linestyle='None', color='green', markersize=4)
        ax.plot(param_sym_time, param_sym_values_no_stim, linestyle='None', color='gray', markersize=4)
        ax.set_xlabel('time (s)')
        ax.set_title('continuous '+param_sym_name[ps])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + animal_list[a] + param_sym_name[ps]+'_continuous', dpi=96)


    
    # OR # Get strides with rebound spike (at laser offset) during stance, during swing, without rebound
    
    # For each trial, plot the average asymmetry in strides with rebound spike and without rebound spike
    
    # For each trial, plot the average asymmetry in strides with rebound spike and trials just after a stride with rebound spike (excluding strides with rebound spike)
    
    
    # For each trial, plot the average asymmetry in strides with rebound spike during stance and without rebound spike
    
    
    
    '''
    # LASER ACCURACY
    tp_laser = np.zeros(len(trials))
    fp_laser = np.zeros(len(trials))
    tn_laser = np.zeros(len(trials))
    fn_laser = np.zeros(len(trials))
    precision_laser = np.zeros(len(trials))
    recall_laser = np.zeros(len(trials))
    f1_laser = np.zeros(len(trials))
    for count_t, trial in enumerate(trials):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        fp_laser[count_t] = fp_trial
        tn_laser[count_t] = tn_trial
        fn_laser[count_t] = fn_trial
        precision_laser[count_t] = precision_trial
        recall_laser[count_t] = recall_trial
        f1_laser[count_t] = f1_trial

    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    rectangle = plt.Rectangle((trials_plot[0]+0.5, 0), trials_plot[-1]-trials_plot[0],
            1, fc='dimgrey',alpha=0.3)
    plt.gca().add_patch(rectangle)
    ax.plot(trials, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
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
            otrack_class.laser_presentation_phase(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                  timestamps_session, final_tracks_phase, 0)
        stim_nr_trials[count_t] = stim_nr
        stride_nr_trials[count_t] = stride_nr
        light_onset_phase_all.extend(light_onset_phase)
        light_offset_phase_all.extend(light_offset_phase)
    otrack_class.plot_laser_presentation_phase(light_onset_phase_all, light_offset_phase_all, laser_event,
                    16, np.sum(stim_nr_trials), np.sum(stride_nr_trials), 0, 1,
                    path_save, animal+'_'+laser_event+'_'+session_list[count_a])
    plt.close('all')


'''