import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'swing'
single_animal_analysis = 0
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours, excluding 1,2,4

# List of paths - it is possible to have only one element
paths = ['D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\05062023 tied trial stim\\',
         'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\06062023 tied stance stim\\',
         'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\07062023 tied swing stim\\']
# ['E:\\090523 split right fast stance stim only split\\']
#['E:\\tied trial stim\\','E:\\tied stance wide stim\\','E:\\tied swing wide stim\\']
#['D:\\Ali\\19042023 split left fast trial stim\\']
#['D:\\Ali\\tied belt stim trial\\','D:\\Ali\\tied belt stance wide stim\\','D:\\Ali\\tied belt swing wide stim\\']
#paths = ['D:\\Ali\\19042023 split left fast trial stim\\', 'D:\\Ali\\21042023 split left fast stance large stim\\', 'D:\\Ali\\25042023 split left fast swing large stim\\']
#paths = ['D:\\Ali\\18042023 split right fast trial stim\\', 'D:\\Ali\\20042023 split right fast stance large stim\\', 'D:\\Ali\\24042023 split right fast swing large stim\\']
#['D:\\Ali\\25042023 split left fast swing large stim\\']
#
experiment_colors = ['purple', 'orange', 'green']      # stim on: trial stance swing


#['C:\\Users\\alice\\Documents\\25042023 split left fast swing large stim\\']
# ['C:\\Users\\alice\\Carey Lab Dropbox\\Tracking Movies\\AnaG+Alice\\090523 split right fast stance stim only split\\']
#['C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\18042023 split right fast trial stim (copied MC16848 T3 to mimic T2)\\']
bs_bool = 1
session = 1
Ntrials = 28
stim_start = 9
split_start = 9
stim_duration = 10      #8
split_duration = 10         #8
plot_rig_signals = 0
print_plots = 1
bs_bool = 1
control_ses = 'right'
control_path = []         # This should be a list; if empty, we have no control (e.g. in tied sessions)
#'E:\\tied trial stim\\'
#'D:\\Ali\\170423 split left ipsi fast control\\'
#'D:\\Ali\\tied belt stim trial\\'
#'D:\\Ali\\14042023 splitS1 right fast nostim\\'
#'D:\\Ali\\170423 split left ipsi fast control\\'
#['C:\\Users\\alice\\Carey Lab Dropbox\\Tracking Movies\\AnaG+Alice\\090523 split right fast stance stim only split\\']         #'C:\\Users\\Ana\\Documents\\PhD\\Projects\Online Stimulation Treadmill\\Experiments\\'
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
import online_tracking_class
import locomotion_class
otrack_classes =  []
locos = []
paths_save = []
param_sym_multi = {}
path_index = 0
for path in paths:
    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path))
    paths_save.append(path + 'grouped output\\')
    if not os.path.exists(path + 'grouped output'):
        os.mkdir(path + 'grouped output')


    # GET THE NUMBER OF ANIMALS AND THE SESSION ID
    animal_session_list = locos[path_index].animals_within_session()
    animal_list = []
    for a in range(len(animal_session_list)):
        animal_list.append(animal_session_list[a][0])
    if len(included_animal_list) == 0:              # All animals included
        included_animal_list = animal_list
    included_animals_id = [animal_list.index(i) for i in included_animal_list]
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])

    # Run the single animal analysis for each of the sessions in paths
    if single_animal_analysis:
        trials = otrack_classes[path_index].get_trials()
        # READ CAMERA TIMESTAMPS AND FRAME COUNTER
        [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_classes[path_index].get_session_metadata(plot_rig_signals)

        # READ SYNCHRONIZER SIGNALS
        [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_classes[path_index].get_synchronizer_data(camera_frames_kept, plot_rig_signals)

        # READ ONLINE DLC TRACKS
        otracks = otrack_classes[path_index].get_otrack_excursion_data(timestamps_session)
        [otracks_st, otracks_sw] = otrack_classes[path_index].get_otrack_event_data(timestamps_session)

        # READ OFFLINE DLC TRACKS
        [offtracks_st, offtracks_sw] = otrack_classes[path_index].get_offtrack_event_data(paw_otrack, locos[path_index], animal, session, timestamps_session)

        # READ OFFLINE PAW EXCURSIONS
        final_tracks_trials = otrack_classes[path_index].get_offtrack_paws(locos[path_index], animal, session)

        # PROCESS SYNCHRONIZER LASER SIGNALS
        laser_on = otrack_classes[path_index].get_laser_on(laser_signal_session, timestamps_session)

        # ACCURACY OF LIGHT ON
        laser_hits = np.zeros(len(trials))
        laser_incomplete = np.zeros(len(trials))
        laser_misses = np.zeros(len(trials))
        for count_t, trial in enumerate(trials):
            [full_hits, incomplete_hits, misses] = otrack_classes[path_index].get_hit_laser_synch(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
            laser_hits[count_t] = full_hits
            laser_incomplete[count_t] = incomplete_hits
            laser_misses[count_t] = misses
        # plot summaries
        fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
        ax.bar(trials, laser_hits, color='green')
        ax.bar(trials, laser_incomplete, bottom = laser_hits, color='orange')
        ax.bar(trials, laser_misses, bottom = laser_hits + laser_incomplete, color='red')
        ax.set_title(laser_event + ' misses')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(paths_save[path_index], 'laser_on_accuracy_' + laser_event + '.png'))
        fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
        rectangle = plt.Rectangle((split_start - 0.5, 0), split_duration, 50, fc=colors[path_index], alpha=0.3)
        plt.gca().add_patch(rectangle)
        ax.plot(trials, (laser_hits/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='green')
        ax.plot(trials, (laser_incomplete/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='orange')
        ax.set_title(laser_event + ' accuracy')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(paths_save[path_index], 'laser_on_accuracy_values_' + laser_event + '.png'))
    

    if single_animal_analysis == 0:
        # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
        # GAIT PARAMETERS ACROSS TRIALS
        param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stance_speed']
        param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
        param_sym[:] = np.NaN
        stance_speed = np.zeros((4, len(animal_list), Ntrials))
        stance_speed[:] = np.NaN
        st_strides_trials = []
        for count_animal, animal in enumerate(animal_list):
            session = int(session_list[count_animal])
            #TODO: check if this filelist needs to be emptied first!
            filelist = locos[path_index].get_track_files(animal, session)
            for f in filelist:
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                st_strides_trials.append(st_strides_mat)
                paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                for count_p, param in enumerate(param_sym_name):
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
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

        # PLOT GAIT PARAMETERS OF INCLUDED ANIMALS
        param_sym_bs_plot = param_sym_bs[:, included_animals_id, :]
        for p in range(np.shape(param_sym)[0] - 1):
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.min(param_sym_bs[p, :, :].flatten())), split_duration,
                                    np.max(param_sym_bs[p, :, :].flatten()) - np.min(param_sym_bs[p, :, :].flatten()),
                                    fc=colors[path_index], alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.hlines(0, 1, len(param_sym_bs[p, a, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym)[1]):         # Loop on all animals
                plt.plot(np.linspace(1, len(param_sym_bs[p, a, :]), len(param_sym_bs[p, a, :])), param_sym_bs[p, a, :], color=animal_colors[a],
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
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                if bs_bool:
                    plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs', dpi=128)
                else:
                    plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_non_bs', dpi=128)
        plt.close('all')

    # PLOT ANIMAL AVERAGE FOR EACH SESSION
    param_sym_multi[path] = {}
    if single_animal_analysis == 0:
        for p in range(np.shape(param_sym)[0] - 1):
            param_sym_bs_ave = param_sym_bs[p, included_animals_id, :]
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_sym_bs_ave[:, :].flatten())), split_duration,
                                        np.nanmax(param_sym_bs_ave[:, :].flatten()) - np.nanmin(param_sym_bs_ave[:, :].flatten()),
                                        fc=experiment_colors[path_index], alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym_bs_ave)[0]):
                plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])), param_sym_bs_ave[a, :], linewidth=1, color = animal_colors[included_animals_id[a]], label=animal_list[included_animals_id[a]])
            ax.legend(frameon=False)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), np.nanmean(param_sym_bs_ave, axis=0), color=experiment_colors[path_index], linewidth=3)
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
                    plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs_average', dpi=128)
                else:
                    plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_non_bs_average', dpi=128)
            # Save param_sym for multi-session plot (in case we have multiple sessions to analyse/plot)
            if len(paths)>1:
                param_sym_multi[path][p] = param_sym_bs_ave
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
                rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_sym_bs_ave[:, :].flatten())), split_duration,
                                            np.nanmax(param_sym_bs_ave[:, :].flatten()) - np.nanmin(
                                                param_sym_bs_ave[:, :].flatten()),
                                            fc=experiment_colors[path_index], alpha=0.3)
                plt.gca().add_patch(rectangle)
                plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
                for a in range(np.shape(param_sym_bs_ave)[0]):
                    plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])),
                                param_sym_bs_ave[a, :], color=experiment_colors[path_index], linewidth=1)
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_bs_ave, axis=0), color=experiment_colors[path_index], linewidth=2)
                # Add control average
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0), color='black', linewidth=2)
                # Add control SE
                ax.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                        np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)+np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(2), 
                        np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)-np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(2), 
                        facecolor='black', alpha=0.5)
                
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
            rectangle = plt.Rectangle((split_start-0.5, -0.5), split_duration, 1, fc=experiment_colors[path_index],alpha=0.3)
            for p in range(4):
                ax.axvline(x = split_start + 0.5, color='dimgray', linestyle='--')
                ax.axvline(x = split_start + 0.5 + split_duration, color='dimgray', linestyle='--')
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
    path_index = path_index+1
        
'''
        # CONTINUOUS STEP LENGTH WITH LASER ON
        trials = np.arange(1, 28)
        for count_animal, animal in enumerate(animal_list):
            param_sl = []
            session = int(session_list[count_animal])
            filelist = locos[path_index].get_track_files(animal, session)
            for count_trial, f in enumerate(filelist):
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                for count_p, param in enumerate(param_sym_name):
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
            param_sl.append(param_mat)
            [stride_idx, trial_continuous, sl_time, sl_values] = locos[path_index].param_continuous_sym(param_sl, st_strides_trials, trials, 'FR', 'FL', 1, 1)
            fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
            sl_time_start = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
            sl_time_duration = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(locos[path_index].trial_time*stim_duration)
            rectangle = plt.Rectangle((sl_time_start, np.nanmin(sl_values)), sl_time_duration-sl_time_start, np.nanmax(sl_values)+np.abs(np.nanmin(sl_values)), fc=colors[a], alpha=0.3)
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
'''

    

# MULTI-SESSION PLOT
if single_animal_analysis==0 and len(paths)>1:
    for p in range(np.shape(param_sym)[0] - 1):
        fig_multi, ax_multi = plt.subplots(figsize=(7, 10), tight_layout=True)
        min_rect = 0
        max_rect = 0
        path_index = 0
        for path in paths:
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_multi[path][p], axis = 0), color=experiment_colors[path_index], linewidth=2, label=experiment_names[path_index])
            # Add SE of each session
            ax_multi.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                        np.nanmean(param_sym_multi[path][p], axis = 0)+np.nanstd(param_sym_multi[path][p], axis = 0)/np.sqrt(2), 
                        np.nanmean(param_sym_multi[path][p], axis = 0)-np.nanstd(param_sym_multi[path][p], axis = 0)/np.sqrt(2), 
                        facecolor=experiment_colors[path_index], alpha=0.5)
            min_rect = min(min_rect,np.nanmin(np.nanmean(param_sym_multi[path][p], axis = 0)-np.nanstd(param_sym_multi[path][p], axis = 0)))
            max_rect = max(max_rect,np.nanmax(np.nanmean(param_sym_multi[path][p], axis=0)+np.nanstd(param_sym_multi[path][p], axis = 0)))
            path_index += 1
        # Add mean control (if you have it)
        if len(control_path)>0:
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0), color='black', linewidth=2, label='control')
            ax_multi.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)+np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(2), 
                            np.nanmean(control_param_sym_bs[p, 1:, :], axis=0)-np.nanstd(control_param_sym_bs[p, 1:, :], axis=0)/np.sqrt(2), 
                            facecolor='black', alpha=0.5)
            min_rect = min(min_rect,np.nanmin(np.nanmean(control_param_sym_bs[p], axis = 0)-np.nanstd(control_param_sym_bs[p], axis = 0)))
            max_rect = max(max_rect,np.nanmax(np.nanmean(control_param_sym_bs[p], axis=0)+np.nanstd(control_param_sym_bs[p], axis = 0)))
        
        ax_multi.legend()
        rectangle = plt.Rectangle((split_start - 0.5, min_rect), split_duration,
                                            max_rect - min_rect,
                                            fc='lightblue', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
        ax_multi.set_xlabel('Trial', fontsize=20)
        ax_multi.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax_multi.spines['right'].set_visible(False)
        ax_multi.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            if bs_bool:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session', dpi=128)
            else:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_non_bs_average_with_control_multi_session', dpi=128)
           
        # Do bar plot of learning parameters
        initial_error_mean = []
        initial_error_std = []
        after_effect_mean = []
        after_effect_std = []
        for path in paths:
            initial_error_mean.append(np.nanmean(param_sym_multi[path][p][:,split_start:split_start+1]))
            initial_error_std.append(np.nanmean(param_sym_multi[path][p][:,split_start:split_start+1]))
            after_effect_mean.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration:split_start+split_duration+1]))
            after_effect_std.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration:split_start+split_duration+1]))
        fig_bar, ax_bar = plt.subplots(2,1)
        ax_bar[0].bar(list(range(len(paths))), initial_error_mean,
            yerr=[abs(ie) for ie in initial_error_std],
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10)
        ax[0].set_xticklabels([''])
        ax_bar[1].bar(list(range(len(paths))), after_effect_mean,
            yerr=after_effect_std,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10)
        if print_plots:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session_barplot', dpi=96)