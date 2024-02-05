import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'swing'
single_animal_analysis = 0
if single_animal_analysis:
    animal = 'VIV41329'
plot_continuous = 0
compare_baselines = 0
compute_statistics = 0
significance_threshold = 0.05

#axes_ranges = {'coo': [-5, 3], 'step_length': [-12, 5], 'double_support': [-7, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
#bars_ranges = {'coo': [-2, 5], 'step_length': [-3, 12], 'double_support': [-5, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
axes_ranges = {'coo': [-3, 5], 'step_length': [-5, 12], 'double_support': [-13, 7], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
bars_ranges = {'coo': [-2, 5], 'step_length': [-3, 12], 'double_support': [-5, 17], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
uniform_ranges = 1

# List of paths for each experiment - it is possible to have only one element
experiment_names = ['swing stim', 'stance stim', 'swing control']                #'trial stim', 

#paths = ['D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\28092023 split right fast trial stim 30s\\']

#paths = [
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\18092023 split right fast trial stim\\',
 #      'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\20092023 split right fast stance stim\\',
  #   'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\22092023 split right fast swing stim\\'
   #   ]

paths = ['F:\\TRACKED\\20240129 tied swing stim IOcontrol\\']               #'F:\\TRACKED\\20240125 tied swing stim\\','F:\\TRACKED\\20240126 tied stance stim\\', 

#paths = [
#        'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20231005 tied stance stim\\',
#        'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20231006 tied swing stim\\'
#        ]

#[
  #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\19092023 split left fast trial stim\\',
     #   'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\21092023 split left fast stance stim\\',
     # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\25092023 split left fast swing stim\\'
   #  ]


#paths = [
      #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\11092023 tied trial stim\\',
       # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\12092023 tied stance stim\\',
      #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\13092023 tied swing stim\\'
 #       ]


#paths = ['D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\05062023 tied trial stim\\',
#        'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\06062023 tied stance stim\\',
#       'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\07062023 tied swing stim\\']

#paths = [
       # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\selected tied trial stim\\',
      # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\selected tied stance stim\\',
      #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\selected tied swing stim\\'
 #       ]



# ['E:\\090523 split right fast stance stim only split\\']
#['E:\\tied trial stim\\','E:\\tied stance wide stim\\','E:\\tied swing wide stim\\']
#['D:\\Ali\\19042023 split left fast trial stim\\']
#['D:\\Ali\\tied belt stim trial\\','D:\\Ali\\tied belt stance wide stim\\','D:\\Ali\\tied belt swing wide stim\\']
#paths = ['D:\\Ali\\19042023 split left fast trial stim\\', 'D:\\Ali\\21042023 split left fast stance large stim\\', 'D:\\Ali\\25042023 split left fast swing large stim\\']
#paths = ['D:\\Ali\\18042023 split right fast trial stim\\', 'D:\\Ali\\20042023 split right fast stance large stim\\', 'D:\\Ali\\24042023 split right fast swing large stim\\']
#['D:\\Ali\\25042023 split left fast swing large stim\\']
#
experiment_colors_dict = {'trial stim':'purple', 'stance stim':'orange','swing stim': 'green', 'swing control': 'black', 'stance control': 'black'}      # stim on: trial stance swing    'trial stim':'purple', 
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
import matplotlib.colors as mcolors
animal_colors_dict = {'MC16851': animal_colors[0], 'MC17319': animal_colors[1],'MC17665': animal_colors[2],'MC17670': animal_colors[3],'MC17666': animal_colors[4],'MC17668': animal_colors[5],'MC17669': animal_colors[6],
                      'MC19022': animal_colors[7],'MC19082': animal_colors[8],'MC19123': animal_colors[9],'MC19124': '#FF00FF', 'MC19130': '#00FFFF','MC19132': '#0000FF','MC19214': '#00FF00',
                      'MC18737': animal_colors[0], 'MC19107': animal_colors[1], 'VIV41329': animal_colors[2], "VIV41330": animal_colors[3], "VIV41344": animal_colors[5],"VIV41345": animal_colors[6],"VIV40958": animal_colors[4]}

#included_animal_list = [ 'MC17319','MC17665','MC17666','MC17668','MC17669','MC17670']
included_animal_list = []
    
    
    #'MC19022','MC19082','MC19123','MC19130','MC19132','MC19124','MC19214']             #  
#included_animal_list = ['MC16851','MC17319','MC17665','MC17666','MC17669','MC17670']#, 'MC19022','MC19082','MC19123','MC19124','MC19214']
#['C:\\Users\\alice\\Documents\\25042023 split left fast swing large stim\\']
# ['C:\\Users\\alice\\Carey Lab Dropbox\\Tracking Movies\\AnaG+Alice\\090523 split right fast stance stim only split\\']
#['C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\18042023 split right fast trial stim (copied MC16848 T3 to mimic T2)\\']
bs_bool = 1
session = 1
Ntrials = 28   #28    #56       # 28
stim_start = 9 #9  #18 #9
split_start = 9    #9 #18        #9
stim_duration = 10  #10  #20      #8
split_duration = 10 #10 #20         #8
plot_rig_signals = 0
print_plots = 1
bs_bool = 0
control_bool = 0
control_ses = 'left'
control_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

if single_animal_analysis:
    trials = otrack_class.get_trials()
    # READ CAMERA TIMESTAMPS AND FRAME COUNTER
    [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_rig_signals)

    # READ SYNCHRONIZER SIGNALS
    [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_rig_signals)

    # READ ONLINE DLC TRACKS
    otracks = otrack_class.get_otrack_excursion_data(timestamps_session)
    [otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(timestamps_session)

    # READ OFFLINE DLC TRACKS
    [offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

    # PROCESS SYNCHRONIZER LASER SIGNALS
    laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

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
    for count_t, trial in enumerate(trials):
        ax.bar(trial, tp_laser[count_t], color='green')
        ax.bar(trial, tn_laser[count_t], bottom=tp_laser[count_t], color='darkgreen')
        ax.bar(trial, fp_laser[count_t], bottom=(tp_laser[count_t] + tn_laser[count_t]), color='red')
        ax.bar(trial, fn_laser[count_t], bottom=(tp_laser[count_t] + tn_laser[count_t] + fp_laser[count_t]), color='crimson')
    ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance.png')
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    for count_t, trial in enumerate(trials):
        ax.bar(trial, f1_laser[count_t], color='darkgrey')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance_f1.png')
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    for count_t, trial in enumerate(trials):
        ax.bar(trial, tp_laser[count_t]+tn_laser[count_t], color='darkgrey')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance_accuracy.png')

    # SETUP ACCURACY
    st_correct_setup = np.zeros(len(trials))
    sw_correct_setup = np.zeros(len(trials))
    for count_t, trial in enumerate(trials):
        [st_correct_trial, sw_correct_trial] = otrack_class.setup_accuracy(otracks, otracks_st, otracks_sw, th_st, th_sw, 0)
        st_correct_setup[count_t] = st_correct_trial
        sw_correct_setup[count_t] = st_correct_trial
    fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 10))
    ax = ax.ravel()
    ax[0].bar(trials, st_correct_setup, color='orange')
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].bar(trials, sw_correct_setup, color='green')
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_setup_accuracy.png')

if single_animal_analysis == 0:
    # GAIT PARAMETERS ACROSS TRIALS
    param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stance_speed']
    param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
    stance_speed = np.zeros((4, len(animal_list), Ntrials))
    st_strides_trials = []
    for count_animal, animal in enumerate(animal_list):
        session = int(session_list[count_animal])
        filelist = loco.get_track_files(animal, session)
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            st_strides_trials.append(st_strides_mat)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            for count_p, param in enumerate(param_sym_name):
                param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                if param == 'stance_speed':
                    for p in range(4):
                        stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
                elif param == 'step_length':
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
                else:
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])

    # BASELINE SUBTRACTION OF GAIT PARAMETERS
    if bs_bool:
        param_sym_bs = np.zeros(np.shape(param_sym))
        for p in range(np.shape(param_sym)[0]-1):
            for a in range(np.shape(param_sym)[1]):
                if stim_start == 9:
                    bs_mean = np.nanmean(param_sym[p, a, :stim_start-1])
                if stim_start == 5:
                    bs_mean = np.nanmean(param_sym[p, a, stim_start-1:8])
                param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
    else:
        param_sym_bs = param_sym

    # PLOT GAIT PARAMETERS - BASELINE SUBTRACTED
    for p in range(np.shape(param_sym)[0] - 1):
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs[p, :, :].flatten())), stim_duration,
                                  np.max(param_sym_bs[p, :, :].flatten()) - np.min(param_sym_bs[p, :, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs[p, a, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym)[1]):
            plt.plot(np.linspace(1, len(param_sym_bs[p, a, :]), len(param_sym_bs[p, a, :])), param_sym_bs[p, a, :],
                     label=animal_list[a], linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.legend(frameon=False)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym_bs', dpi=128)
    plt.close('all')

    # PLOT GAIT PARAMETERS - NOT-BASELINE SUBTRACTED
    for p in range(np.shape(param_sym)[0] - 1):
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym[p, :, :].flatten())), stim_duration,
                                  np.max(param_sym[p, :, :].flatten()) - np.min(param_sym[p, :, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym[p, a, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym)[1]):
            plt.plot(np.linspace(1, len(param_sym[p, a, :]), len(param_sym[p, a, :])), param_sym[p, a, :],
                     label=animal_list[a], linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.legend(frameon=False)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym', dpi=128)
    plt.close('all')

    # PLOT ANIMAL AVERAGE
    for p in range(np.shape(param_sym)[0] - 1):
        # param_sym_bs_ave = param_sym_bs[p, np.array([0, 2, 3, 4]), :]
        param_sym_bs_ave = param_sym_bs[p, :, :]
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), stim_duration,
                                  np.max(param_sym_bs_ave[:, :].flatten()) - np.min(param_sym_bs_ave[:, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym_bs_ave)[0]):
            plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])), param_sym_bs_ave[a, :], color='darkgray', linewidth=1)
        plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), np.nanmean(param_sym_bs_ave, axis=0), color='black', linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym_bs_average', dpi=128)
    plt.close('all')

    if control_bool:
        param_sym_bs_plot = param_sym_bs[:, :, :]
        param_sym_bs_control = np.load(control_path + 'split_' + control_ses + '_fast_control_params_sym_bs_noMC16846.npy')
        for p in range(np.shape(param_sym)[0] - 1):
            param_sym_bs_ave = param_sym_bs_plot[p, :, :]
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), stim_duration,
                                      np.max(param_sym_bs_ave[:, :].flatten()) - np.min(
                                          param_sym_bs_ave[:, :].flatten()),
                                      fc='dimgrey', alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym_bs_ave)[0]):
                plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])),
                         param_sym_bs_ave[a, :], color='darkgray', linewidth=1)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                     np.nanmean(param_sym_bs_ave, axis=0), color='orange', linewidth=2)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                     np.nanmean(param_sym_bs_control[p, :, :], axis=0), color='black', linewidth=2)
            ax.set_xlabel('Trial', fontsize=20)
            ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
            if p == 2:
                plt.gca().invert_yaxis()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if not os.path.exists(path_save):
                    os.mkdir(path_save)
                plt.savefig(path_save + param_sym_name[p] + '_sym_bs_average_with_control', dpi=128)
        plt.close('all')

    # PLOT STANCE SPEED
    for a in range(np.shape(stance_speed)[1]):
        data = stance_speed[:, a, :]
        fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start-0.5, stim_duration), stim_duration, 210-(-10), fc='dimgrey',alpha=0.3)
        for p in range(4):
            ax.axvline(x = stim_start + 0.5, color='dimgray', linestyle='--')
            ax.axvline(x = stim_start + 0.5 + stim_duration, color='dimgray', linestyle='--')
            ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Trial', fontsize = 20)
            ax.set_ylabel('Stance speed', fontsize = 20)
            ax.tick_params(axis='x',labelsize = 16)
            ax.tick_params(axis='y',labelsize = 16)
            ax.set_title(animal_list[a],fontsize=18)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + animal_list[a] + '_stancespeed', dpi=96)
    plt.close('all')

        if len(control_path)>0:
            param_sym_bs_plot = param_sym_bs[:, included_animals_id, :]
            if len(control_path)>0:
                # Load control parameters if they have been computed and saved
                if os.path.exists(control_path[0] + control_filename):
                    control_param_sym_bs = np.load(control_path[0] + control_filename)
                else:           # Compute control parameters and save them
                    otrack_classes =  []
                    locos = []
                    paths_save = []
                    param_sym_multi = {}
                    path_index = 0
                    for path in paths:
                        control_otrack_class = online_tracking_class.otrack_class(control_path[0])
                        control_loco = locomotion_class.loco_class(control_path[0])
                                                                                                #paths_save.append(path + 'grouped output\\')
                                                                                               # if not os.path.exists(path + 'grouped output'):
                                                                                                  #  os.mkdir(path + 'grouped output')
                        # GET THE NUMBER OF ANIMALS AND THE SESSION ID
                        control_animal_session_list = control_loco.animals_within_session()
                        control_animal_list = []
                        for a in range(len(control_animal_session_list)):
                            control_animal_list.append(control_animal_session_list[a][0])
                        
                        control_session_list = []
                        for a in range(len(control_animal_session_list)):
                            control_session_list.append(control_animal_session_list[a][1])
                    control_param_sym_bs, control_stance_speed, control_st_strides_trials = control_loco.prepare_and_compute_gait_param(control_animal_list, Ntrials, param_sym_name, control_session_list, bs_bool, stim_start)

                    with open(control_path[0] + control_filename, 'wb') as f:
                        np.save(f, control_param_sym_bs)

                if len(included_animal_list)>0:
                    control_param_sym_bs = control_param_sym_bs[:, included_animals_id, :]

            for p in range(np.shape(param_sym)[0] - 1):   
                param_sym_bs_ave = param_sym_bs_plot[p, :, :]
                
                fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)  
                
                plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
                for a in range(np.shape(param_sym_bs_ave)[0]):
                    plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])),
                                param_sym_bs_ave[a, :], color=animal_colors_dict[included_animal_list[a]], linewidth=1, label=animal_list[a])
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_bs_ave, axis=0), color=experiment_colors_dict[experiment_name], linewidth=2, label=experiment_names[path_index])
               # Add control average
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(control_param_sym_bs[p, :, :], axis=0), color='black', linewidth=2, label='control')
                # Add control SE
                ax.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                        np.nanmean(control_param_sym_bs[p, :, :], axis=0)+np.nanstd(control_param_sym_bs[p, :, :], axis=0)/np.sqrt(len(included_animal_list)), 
                        np.nanmean(control_param_sym_bs[p, :, :], axis=0)-np.nanstd(control_param_sym_bs[p, :, :], axis=0)/np.sqrt(len(included_animal_list)), 
                        facecolor='black', alpha=0.5)

                if uniform_ranges:
                    ax.set(ylim=axes_ranges[param_sym_name[p]])
                    rectangle = plt.Rectangle((split_start - 0.5, axes_ranges[param_sym_name[p]][0]), split_duration,
                                            axes_ranges[param_sym_name[p]][1] - axes_ranges[param_sym_name[p]][0],
                                            fc='lightblue', alpha=0.3)
                else:
                    rectangle = plt.Rectangle((split_start - 0.5, min(np.nanmin(param_sym_bs_ave[:, :].flatten()), np.nanmin(np.nanmean(control_param_sym_bs[p, :, :], axis=0)))), split_duration,
                                                max(np.nanmax(param_sym_bs_ave[:, :].flatten()), np.nanmax(np.nanmean(control_param_sym_bs[p, :, :], axis=0))) - min(np.nanmin(
                                                    param_sym_bs_ave[:, :].flatten()),np.nanmin(np.nanmean(control_param_sym_bs[p, :, :], axis=0))),
                                                fc='lightblue', alpha=0.3)
                plt.gca().add_patch(rectangle)
                plt.legend()
                ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
                ax.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
                ax.set_xlabel('Trial', fontsize=20)
                ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
                if p == 2:
                    plt.gca().invert_yaxis()
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if print_plots:
                    if not os.path.exists(paths_save[path_index]):
                        os.mkdir(paths_save[path_index])
                    if bs_bool:
                        plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs_average_with_control', dpi=128)
                    else:
                        plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_non_bs_average_with_control', dpi=128)
                if len(paths)>1:
                    param_sym_multi[path][p] = param_sym_bs_ave
                    
            plt.close('all')

        # PLOT STANCE SPEED for ALL ANIMALS
        for a in range(np.shape(stance_speed)[1]):
            data = stance_speed[:, a, :]
            fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
            rectangle = plt.Rectangle((split_start-0.5, -0.5), split_duration, 1, fc='lightblue',alpha=0.3)
            for p in range(4):
                ax.axvline(x = split_start, color='dimgray', linestyle='--')
                ax.axvline(x = split_start+ split_duration, color='dimgray', linestyle='--')
                ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Trial', fontsize = 20)
                ax.set_ylabel('Stance speed', fontsize = 20)
                ax.tick_params(axis='x',labelsize = 16)
                ax.tick_params(axis='y',labelsize = 16)
                ax.set_title(animal_list[a],fontsize=18)
            
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                plt.savefig(paths_save[path_index] + animal_list[a] + '_stancespeed', dpi=96)
        plt.close('all')
    
    
        # CONTINUOUS STEP LENGTH WITH LASER ON
        if plot_continuous:
            trials = np.arange(1, Ntrials+1)
            for count_animal, animal in enumerate(animal_list):
                param_sl = []
                st_strides_trials = []
                session = int(session_list[count_animal])
                filelist = locos[path_index].get_track_files(animal, session)
                for count_trial, f in enumerate(filelist):
                    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                    [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                    paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
                    param_sl.append(param_mat)
                    st_strides_trials.append(st_strides_mat)
                [stride_idx, trial_continuous, sl_time, sl_values] = locos[path_index].param_continuous_sym(param_sl, st_strides_trials, trials, 'FR', 'FL', 1, 1)
                fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
                sl_time_start = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
                sl_time_duration = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(locos[path_index].trial_time*stim_duration)
                rectangle = plt.Rectangle((sl_time_start, np.nanmin(sl_values)), sl_time_duration-sl_time_start, np.nanmax(sl_values)+np.abs(np.nanmin(sl_values)), fc=experiment_colors_dict[experiment_name], alpha=0.3)
                plt.gca().add_patch(rectangle)
                ax.plot(sl_time, sl_values, color='black')
                ax.set_xlabel('time (s)')
                ax.set_title('continuous step length')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if print_plots:
                    if not os.path.exists(paths_save[path_index]):
                        os.mkdir(paths_save[path_index])
                    plt.savefig(paths_save[path_index] + animal_list[a] + '_sl_sym_continuous', dpi=96)

    path_index = path_index+1
    
included_animal_list = ['VIV41344','VIV41345','VIV40958']           #['VIV41329','VIV41330']         #[ 'MC19022','MC19082','MC19123','MC19124','MC19214']  

included_animals_id = [animal_list.index(i) for i in included_animal_list]
# MULTI-SESSION PLOT
if single_animal_analysis==0 and (len(paths)>1 or len(control_path)>0):
    if len(control_path)>0:
        current_experiment_names = ['control']
        current_experiment_colors = ['black']
        current_bar_labels = ['control']
    
    for p in range(np.shape(param_sym)[0] - 1):
        fig_multi, ax_multi = plt.subplots(figsize=(7, 10), tight_layout=True)
        min_rect = 0
        max_rect = 0
        path_index = 0
        for path in paths:
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_multi[path][p], axis = 0), color=experiment_colors_dict[experiment_names[path_index]], linewidth=2, label=experiment_names[path_index])
            # Add SE of each session
            ax_multi.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                        np.nanmean(param_sym_multi[path][p], axis = 0)+np.nanstd(param_sym_multi[path][p], axis = 0)/np.sqrt(len(included_animal_list)), 
                        np.nanmean(param_sym_multi[path][p], axis = 0)-np.nanstd(param_sym_multi[path][p], axis = 0)/np.sqrt(len(included_animal_list)), 
                        facecolor=experiment_colors_dict[experiment_names[path_index]], alpha=0.5)
            min_rect = min(min_rect,np.nanmin(np.nanmean(param_sym_multi[path][p], axis = 0)-np.nanstd(param_sym_multi[path][p], axis = 0)))
            max_rect = max(max_rect,np.nanmax(np.nanmean(param_sym_multi[path][p], axis=0)+np.nanstd(param_sym_multi[path][p], axis = 0)))
            path_index += 1
        # Add mean control (if you have it)
        if len(control_path)>0:
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0), color='k', linewidth=2, label='control')
            ax_multi.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)+np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(len(included_animal_list)), 
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)-np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(len(included_animal_list)), 
                            facecolor='k', alpha=0.5)
            min_rect = min(min_rect,np.nanmin(np.nanmean(control_param_sym_bs[p], axis = 0)-np.nanstd(control_param_sym_bs[p], axis = 0)))
            max_rect = max(max_rect,np.nanmax(np.nanmean(control_param_sym_bs[p], axis=0)+np.nanstd(control_param_sym_bs[p], axis = 0)))
        
        ax_multi.legend()
        if uniform_ranges:
            rectangle = plt.Rectangle((split_start - 0.5, axes_ranges[param_sym_name[p]][0]), split_duration,
                                                axes_ranges[param_sym_name[p]][1] - axes_ranges[param_sym_name[p]][0],
                                                fc='lightblue', alpha=0.3)
        else:
            rectangle = plt.Rectangle((split_start - 0.5, min_rect), split_duration,
                                                max_rect - min_rect,
                                                fc='lightblue', alpha=0.3)
        plt.gca().add_patch(rectangle)
        ax.plot(sl_time, sl_values, color='black')
        ax.set_xlabel('time (s)')
        ax.set_title('continuous step length')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + animal_list[a] + '_sl_sym_continuous', dpi=96)

        # First add control, if any
        if len(control_path)>0:
            if (param_sym_name[p] == 'double_support' and control_ses == 'right') or ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'left'):
                initial_error.append(np.nanmean(control_param_sym_bs[p][:,split_start-1:split_start+1], axis=1))          
                learning.append(-(np.nanmean(control_param_sym_bs[p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(control_param_sym_bs[p][:,split_start-1:split_start+1], axis=1)))
                aftereffect.append(-np.nanmean(control_param_sym_bs[p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
            else:
                initial_error.append(-np.nanmean(control_param_sym_bs[p][:,split_start-1:split_start+1], axis=1))          
                learning.append(np.nanmean(control_param_sym_bs[p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(control_param_sym_bs[p][:,split_start-1:split_start+1], axis=1))
                aftereffect.append(np.nanmean(control_param_sym_bs[p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))

        path_index = 0  
        current_experiment_names = []
        current_experiment_colors = []
        current_bar_labels = []
        for path in paths:
            for exp in experiment_colors_dict.keys():
                if exp in path:
                    experiment_name = exp
                    current_experiment_names.append(experiment_name)
                    current_experiment_colors.append(experiment_colors_dict[exp])
                    current_bar_labels.append(exp)
            
            # Flip signs to have good control learning always positive
            if (param_sym_name[p] == 'double_support' and control_ses == 'right') or ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'left'):
                initial_error.append(np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))  
                learning.append(-(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1)))
                aftereffect.append(-np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
            elif ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'right') or (param_sym_name[p] == 'double_support' and control_ses == 'left'):
                initial_error.append(-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))         
                learning.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))
                aftereffect.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
          
            # Compare to control - Statistics
            if compute_statistics:
                stat_initial_error.append(st.ttest_rel(initial_error[0], initial_error[path_index]).pvalue<significance_threshold)
                stat_learning.append(st.ttest_rel(learning[0], learning[path_index]).pvalue<significance_threshold)
                stat_aftereffect.append(st.ttest_rel(aftereffect[0], aftereffect[path_index]).pvalue<significance_threshold)
                path_index+=1

        learning_sym_change=100*np.divide(np.array(learning),np.array(initial_error))
        aftereffect_sym_change=100*np.divide(np.array(aftereffect),np.array(initial_error))
        if compute_statistics:
            for path_index in range(len(paths)):
                stat_learning_sym_change.append(st.ttest_rel(learning_sym_change[0], learning_sym_change[path_index]).pvalue<significance_threshold)
                stat_aftereffect_sym_change.append(st.ttest_rel(aftereffect_sym_change[0], aftereffect_sym_change[path_index]).pvalue<significance_threshold)
                '''
        fig_bar, ax_bar = plt.subplots(2,3)
        bars = ax_bar[0,0].bar(list(range(len(paths)+len(control_path))), np.nanmean(initial_error, axis=1), yerr=np.nanstd(initial_error, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,0].plot(initial_error,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,0].set_ylabel(param_sym_name[p]+' (mm)')
        ax_bar[0,0].set_title('init. error', size=9)
        ax_bar[0,1].bar(list(range(len(paths)+len(control_path))), np.nanmean(learning, axis=1), yerr=np.nanstd(learning, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        ax_bar[0,1].set_title('late-early', size=9)
        ax_bar[0,2].bar(list(range(len(paths)+len(control_path))), np.nanmean(aftereffect, axis=1), yerr=np.nanstd(aftereffect, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,2].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(aftereffect, axis=1)+np.nanstd(aftereffect, axis=1)),0)*i if i==1 else math.nan*i for i in stat_aftereffect ],'*', color='black')
        #ax_bar[0,2].plot(aftereffect,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,2].set_title('aftereffect', size=9)
        ax_bar[1,1].bar(list(range(len(paths)+len(control_path))), np.nanmean(learning_sym_change, axis=1), yerr=np.nanstd(learning_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,1].plot(learning_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,1].set_title('% change late-early', size=9)
        ax_bar[1,1].set_ylabel('% sym change')
        ax_bar[1,2].bar(list(range(len(paths)+len(control_path))), np.nanmean(aftereffect_sym_change, axis=1), yerr=np.nanstd(aftereffect_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,2].plot(aftereffect_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,2].set_title('% change aftereffect', size=9)
        # Add single animal data
        for a in included_animals_id:
            ax_bar[0,0].plot(np.array(initial_error)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[a]], color=animal_colors_dict[animal_list[a]], linewidth=1)
            ax_bar[0,1].plot(np.array(learning)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[a]], color=animal_colors_dict[animal_list[a]], linewidth=1)
            ax_bar[0,2].plot(np.array(aftereffect)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[a]], color=animal_colors_dict[animal_list[a]], linewidth=1)
            ax_bar[1,1].plot(np.array(learning_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[a]], color=animal_colors_dict[animal_list[a]], linewidth=1)
            ax_bar[1,2].plot(np.array(aftereffect_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[a]], color=animal_colors_dict[animal_list[a]], linewidth=1)
               
        # Add zero line
        for ax in ax_bar.flatten():
            ax.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
            if uniform_ranges:
                ax.set(ylim= bars_ranges[param_sym_name[p]])      #   [-4.5,max(abs(np.array(axes_ranges[param_sym_name[p]])))])

        if compute_statistics:
            ax_bar[0,0].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(initial_error, axis=1)+np.nanstd(initial_error, axis=1)),0)*i if i==1 else math.nan*i for i in stat_initial_error ],'*', color='black')
            ax_bar[0,1].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(learning, axis=1)+np.nanstd(learning, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning ],'*', color='black')
            ax_bar[1,1].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(learning_sym_change, axis=1)+np.nanstd(learning_sym_change, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_sym_change ],'*', color='black')
            ax_bar[1,2].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(aftereffect_sym_change, axis=1)+np.nanstd(aftereffect_sym_change, axis=1)),0)*i if i==1 else math.nan*i for i in stat_aftereffect_sym_change ],'*', color='black')

        
        for ax in ax_bar.flatten():
            ax.set_xticks([])
        fig_bar.delaxes(ax_bar[1,0])
        fig_bar.suptitle(param_sym_name[p])
        fig_bar.tight_layout()
        fig_bar.legend(bars, current_bar_labels,
           loc="lower left",   
           borderaxespad=3 
           )
'''
                               
        if print_plots_multi_session:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session_barplot', dpi=96)


      

