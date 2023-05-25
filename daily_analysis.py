import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'stance'
single_animal_analysis = 0
# List of paths - it can have only one element
session = 1
Ntrials = 28
stim_start = 5
stim_duration = 14
plot_rig_signals = 0
print_plots = 1
bs_bool = 1
control_ses = 'right'
control_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\Online Stimulation Treadmill\\Experiments\\'
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
        stance_speed = np.zeros((4, len(animal_list), Ntrials))
        st_strides_trials = []
        for count_animal, animal in enumerate(animal_list):
            session = int(session_list[count_animal])
            #TODO: check if this filelist needs to be emptied first!
            filelist = locos[path_index].get_track_files(animal, session)
            for count_trial, f in enumerate(filelist):
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

        # PLOT GAIT PARAMETERS WITH CONTROL WITHOUT MC16846
        param_sym_bs_plot = param_sym_bs[:, 2:, :]
        for p in range(np.shape(param_sym)[0] - 1):
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.min(param_sym_bs[p, :, :].flatten())), split_duration,
                                    np.max(param_sym_bs[p, :, :].flatten()) - np.min(param_sym_bs[p, :, :].flatten()),
                                    fc=colors[path_index], alpha=0.3)
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
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs', dpi=128)
        plt.close('all')

    # PLOT ANIMAL AVERAGE FOR EACH SESSION
    param_sym_multi[path] = {}
    if single_animal_analysis == 0:
        for p in range(np.shape(param_sym)[0] - 1):
            param_sym_bs_ave = param_sym_bs[p, :, :]
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), split_duration,
                                        np.max(param_sym_bs_ave[:, :].flatten()) - np.min(param_sym_bs_ave[:, :].flatten()),
                                        fc=colors[path_index], alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.hlines(0, 1, len(param_sym_bs_ave[a, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym_bs_ave)[0]):
                plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])), param_sym_bs_ave[a, :], color=colors[path_index], linewidth=1)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), np.nanmean(param_sym_bs_ave, axis=0), color=colors[path_index], linewidth=2)
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
                plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs_average', dpi=128)
            # Save param_sym for multi-session plot (in case we have multiple sessions to analyse/plot)
            #if len(paths)>1:
            #    param_sym_multi[path][p] = param_sym_bs_ave
        plt.close('all')

        if len(control_path)>0:
            param_sym_bs_plot = param_sym_bs[:, 2:, :]
            if len(control_path)>0:
                param_sym_bs_control = np.load(control_path[0] + 'split_' + control_ses + '_fast_control_params_sym_bs_noMC16846.npy')
            for p in range(np.shape(param_sym)[0] - 1):   
                param_sym_bs_ave = param_sym_bs_plot[p, :, :]
                fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
                rectangle = plt.Rectangle((split_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), split_duration,
                                            np.max(param_sym_bs_ave[:, :].flatten()) - np.min(
                                                param_sym_bs_ave[:, :].flatten()),
                                            fc=colors[path_index], alpha=0.3)
                plt.gca().add_patch(rectangle)
                plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
                for a in range(np.shape(param_sym_bs_ave)[0]):
                    plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])),
                                param_sym_bs_ave[a, :], color=colors[path_index], linewidth=1)
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_bs_ave, axis=0), color=colors[path_index], linewidth=2)
                # Add control average
                plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                            np.nanmean(param_sym_bs_control[p, 1:, :], axis=0), color='black', linewidth=2)
                # Add control SE
                ax.fill_between(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), 
                        np.nanmean(param_sym_bs_control[p, 1:, :], axis=0)+np.nanstd(param_sym_bs_control[p, 1:, :], axis=0)/np.sqrt(2), 
                        np.nanmean(param_sym_bs_control[p, 1:, :], axis=0)-np.nanstd(param_sym_bs_control[p, 1:, :], axis=0)/np.sqrt(2), 
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
                    plt.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs_average_with_control', dpi=128)
                if len(paths)>1:
                    param_sym_multi[path][p] = param_sym_bs_ave
                    
            plt.close('all')

        # PLOT STANCE SPEED
        for a in range(np.shape(stance_speed)[1]):
            data = stance_speed[:, a, :]
            fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
            rectangle = plt.Rectangle((split_start-0.5, -0.5), split_duration, 1, fc=colors[path_index],alpha=0.3)
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
