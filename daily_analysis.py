import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import math


# Inputs
laser_event = 'swing'
single_animal_analysis = 0
if single_animal_analysis:
    animal = 'MC19022'
plot_continuous = 0
compare_baselines = 0
compute_statistics = 0
significance_threshold = 0.05

#axes_ranges = {'coo': [-5, 3], 'step_length': [-12, 5], 'double_support': [-7, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
#bars_ranges = {'coo': [-2, 5], 'step_length': [-3, 12], 'double_support': [-5, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
# Opto
axes_ranges = {'coo': [-5, 5], 'step_length': [-12, 12], 'double_support': [-7, 7], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
bars_ranges = {'coo': [-4, 4], 'step_length': [-12, 12], 'double_support': [-7, 7], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
# ChR2 right
axes_ranges = {'coo': [-6, 2], 'step_length': [-11, 5], 'double_support': [-8, 15], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
bars_ranges = {'coo': [-2, 2], 'step_length': [-2, 5], 'double_support': [-7, 1], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
# ChR2 left
#axes_ranges = {'coo': [-4, 6], 'step_length': [-7, 10], 'double_support': [-12, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
#bars_ranges = {'coo': [-2, 2], 'step_length': [-5, 2], 'double_support': [-1, 6], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
uniform_ranges = 1

# List of paths for each experiment - it is possible to have only one element
experiment_names = ['chr2']           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'

#paths = ['D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\28092023 split right fast trial stim 30s\\']

#paths = [
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\18092023 split right fast trial stim\\',
 #      'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\20092023 split right fast stance stim\\',
  #   'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\22092023 split right fast swing stim\\'
   #   ]

#paths = [
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240130 tied stance stim IOcontrol\\'
        #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240126 tied stance stim\\',
        #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240125 tied swing stim\\'
       # ]

#paths = [
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240130 tied stance stim IOcontrol\\'
 #       'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240209 tied stance stim REPLAY\\',
  #      'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240208 tied swing stim REPLAY\\'
   #     ]

paths = [
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240311 tied swing stim redone\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim\\',
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\ALL_ANIMALS\\split left fast control\\',
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\ALL_ANIMALS\\split left fast stance stim\\',
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\ALL_ANIMALS\\split left fast swing stim\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\LocoCF-internal\\Tout_data\\20230606 tied stance stim\\'
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\20230608 split right fast control\\'
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240409 tied stance stim CTXchr2\\',
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240307 tied swing stim IOchr2\\'
       # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\20240202 split left fast stance stim\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\Susd4KO project\\20240212 20240221 20240304 split right fast Susd4KO\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\Susd4KO project\\20240216 20240226 20240308 split left fast Susd4KO\\'
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split left fast no-stim S3\\",
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast no-stim S2\\",
      #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast stim S4\\",
     # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast perturb S5\\",
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split left fast perturb S6\\"
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split left fast no-stim\\",
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split left fast perturb\\"
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split right fast perturb\\original protocol\\"
     "D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240531 tied swing stim IOchr2 50ms\\"
      #"D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240503 tied stance stim IOchr2 50ms\\"
     ]


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
experiment_colors_dict = {'trial stim':'purple', 'stance stim':'orange','swing stim': 'green', 'control':'black', 'chr2': 'cyan',
                          'right fast no-stim': 'gray',     # 'blue', 
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen'}      # stim on: trial stance swing    'trial stim':'purple', 
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
animal_colors_dict = {'MC16846': "#FFD700",'MC16848':"#BBF90F",'MC16850': "#15B01A",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
                      'MC17665': '#CCCCFF','MC17670': '#660033','MC17666': animal_colors[4], 'MC17668': animal_colors[5],'MC17669': animal_colors[6], 
                      'MC19022': animal_colors[7],'MC19082': animal_colors[8],'MC19123': animal_colors[9], 'MC19124': '#FF00FF', 'MC19130': '#00FFFF',
                      'MC19132': '#0000FF','MC19214': '#00FF00', 'MC18737': '#F08080', 'MC19107': '#FA8072', 'VIV41330': animal_colors[2], 
                      'VIV41329': animal_colors[3], 'VIV41375': '#5C62D6', 'VIV41376': '#FF0000', 'VIV41428': '#BC8F8F', 'VIV41429': '#A9932CC',
                      'VIV41430': '#FF4500',
                      #IO fiber control
                      'VIV40958':animal_colors[4], 'VIV41344':animal_colors[5], 'VIV41345':animal_colors[6], 
                      #ChR2
                      'VIV42375': animal_colors[4],'VIV42376': animal_colors[5],'VIV42428': animal_colors[7],'VIV42429': animal_colors[8],
                      'VIV42430': animal_colors[9], 'VIV42906': animal_colors[2], 'VIV42907': animal_colors[3],'VIV42908':animal_colors[4], 'VIV42974':animal_colors[5],
                      'VIV42985':animal_colors[6], 'VIV42992': animal_colors[7],'VIV42987': animal_colors[8],
                      'VIV44771': '#CCCCFF', 'VIV44765': '#00FF00', 'VIV44766': '#FF4500', 'VIV45372': '#BC8F8F', 'VIV45373': '#F08080', 
                      #HGM
                      'MC11231': "#FFD700",'MC11232':"#BBF90F",'MC11234': "#15B01A",'MC11235': animal_colors[0], 'MC24409': animal_colors[1],
                      'MC24410': '#CCCCFF','MC24411': '#660033','MC24412': animal_colors[4], 'MC24413': animal_colors[5],
                      'MC1262': animal_colors[0],'MC1263':  animal_colors[1],'MC1328': animal_colors[2],'MC1329': animal_colors[3],'MC1330':  animal_colors[4],
                      'A1': "#FFD700",'A2':"#BBF90F",'A3': "#15B01A",'A4': '#0000FF','A5': '#00FF00', 'MC1705': '#F08080', 'V1': '#FA8072',
                      'V2': '#5C62D6', 'V3': '#FF0000', 'V4': '#BC8F8F', 'MC1659': '#BC8F8F', 'MC1660': '#FF4500','MC1661': '#CCCCFF','MC1663': '#660033','MC1664': '#00FFFF',
                      }
'''
# Not histo confirmed in gray
animal_colors_dict = {'MC16846': "#BBBBBB",'MC16848':"#BBBBBB",'MC16850': "#BBBBBB",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
                      'MC17665': '#CCCCFF','MC17670': '#660033','MC17666': animal_colors[4], 'MC17668': animal_colors[5],'MC17669': '#BBBBBB', 
                      'MC19022': '#BBBBBB','MC19082': animal_colors[8],'MC19123': '#BBBBBB', 'MC19124': '#FF00FF', 'MC19130': '#00FFFF',
                      'MC19132': '#BBBBBB','MC19214': '#00FF00', 'MC18737': '#BBBBBB', 'MC19107': '#FA8072', 'VIV41330': '#777777', 
                      'VIV41329': '#777777', 'VIV41375': '#777777', 'VIV41376': '#777777', 'VIV41428': '#777777', 'VIV41429': '#777777',
                      'VIV41430': '#777777',
                      #IO fiber control
                      'VIV40958':animal_colors[4], 'VIV41344':animal_colors[5], 'VIV41345':animal_colors[6], 
                      #ChR2
                      'VIV42375': animal_colors[4],'VIV42376': animal_colors[5],'VIV42428': animal_colors[7],'VIV42429': animal_colors[8],
                      'VIV42430': animal_colors[9], 'VIV42906': animal_colors[2], 'VIV42907': animal_colors[3],'VIV42908':animal_colors[4], 'VIV42974':animal_colors[5],
                      'VIV42985':animal_colors[6], 'VIV42992': animal_colors[7],'VIV42987': animal_colors[8]}

'''
#included_animal_list = [ 'MC17319','MC17665','MC17666','MC17668','MC17669','MC17670']

included_animal_list = ['VIV44771', 'VIV44766', 'VIV45372', 'VIV45373']
#'MC11231','MC11234','MC11235','MC24410','MC24413'] # ChR2 LE
            #['MC1262','MC1263','MC1328','MC1329','MC1330']     # ChR2 HE                #'VIV42906', 'VIV42974', 'VIV42908','VIV42985','VIV42987']  
'''
['MC16846','MC16848','MC16850','MC16851', 'MC17319',
   'MC17665','MC17670','MC17666', 'MC17668','MC17669', 
   'MC19022','MC19082','MC19123', 'MC19124', 'MC19130',
   'MC19132','MC19214', 'MC18737', 'MC19107', 'VIV41330', 
   'VIV41329']          # 21 animals
  
included_animal_list =  ['MC16851', 'MC17319','MC17665','MC17670','MC17666', 'MC17668', 
    'MC19082','MC19124', 'MC19130','MC19214']         # 11 animals
      '''
    #'MC19022','MC19082','MC19123','MC19130','MC19132','MC19124','MC19214']             #  
#included_animal_list = ['MC16851','MC17319','MC17665','MC17666','MC17669','MC17670']#, 'MC19022','MC19082','MC19123','MC19124','MC19214']
#['C:\\Users\\alice\\Documents\\25042023 split left fast swing large stim\\']
# ['C:\\Users\\alice\\Carey Lab Dropbox\\Tracking Movies\\AnaG+Alice\\090523 split right fast stance stim only split\\']
#['C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\18042023 split right fast trial stim (copied MC16848 T3 to mimic T2)\\']

session = 1
Ntrials = 28  #28    #56       # 28
stim_start = 9 #9  #18 #9
split_start = 9   #9 #18        #9
stim_duration = 10  #10  #20      #8
split_duration = 10 #10 #20         #8
plot_rig_signals = 0
print_plots = 1
print_plots_multi_session = 1
bs_bool = 1
control_ses = 'left'
control_path = []       #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']      #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\14092023 split right fast control\\'] #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']   #]         # This should be a list; if empty, we have no control (e.g. in tied sessions)
control_filename = 'split_'+control_ses+'_fast_control_params_sym_bs.npy'
#'E:\\tied trial stim\\'
#'D:\\Ali\\170423 split left ipsi fast control\\'
#'D:\\Ali\\tied belt stim trial\\'
#'D:\\Ali\\14042023 splitS1 right fast nostim\\'
#'D:\\Ali\\170423 split left ipsi fast control\\'
#['C:\\Users\\alice\\Carey Lab Dropbox\\Tracking Movies\\AnaG+Alice\\090523 split right fast stance stim only split\\']         #'C:\\Users\\Ana\\Documents\\PhD\\Projects\Online Stimulation Treadmill\\Experiments\\'
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
paws = ['FR', 'HR', 'FL', 'HL']
import online_tracking_class
import locomotion_class
otrack_classes =  []
locos = []
paths_save = []
param_sym_multi = {}
path_index = 0
for path in paths:
    print("Analysing..........................", path)
    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path))
    paths_save.append(path + 'grouped output\\')
    if not os.path.exists(path + 'grouped output'):
        os.mkdir(path + 'grouped output')

    for exp in experiment_names:
        if exp in path:
            experiment_name = exp

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
        rectangle = plt.Rectangle((split_start - 0.5, 0), split_duration, 50, fc='lightblue', alpha=0.3)
        ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
        ax.axvline(x = stim_start+stim_duration+0.5, color = 'k', linestyle = '-', linewidth=0.5)
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
        param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'phase_st', 'stance_speed']
        param_gait_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stride_duration', 'swing_duration', 'stance_duration', 'swing_velocity','stance_speed','body_center_x_stride','body_speed_x','duty_factor','candence','phase_st']
        param_label = ['Center of\noscillation (mm)', 'Step length (mm)', '% of double support', 'Spatial motor\noutput (mm)', 'Swing length(mm)', 'Stance phase', 'Stance speed']
        param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
        param_sym[:] = np.NaN
        param_paw = np.zeros((len(param_sym_name), len(animal_list), 4, Ntrials))
        param_paw[:] = np.nan
        param_phase = np.zeros((4, len(animal_list), Ntrials))
        param_phase[:] = np.nan
        stance_speed = np.zeros((4, len(animal_list), Ntrials))
        stance_speed[:] = np.NaN
        st_strides_trials = []
        param_gait = np.zeros((len(param_gait_name), len(animal_list), Ntrials))
        param_gait[:] = np.NaN
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
                    '''if param == 'stance_speed':
                        for p in range(4):
                            stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
                    elif param == 'step_length':
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
                    else:
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])'''

                    if param == 'phase_st':
                        for p in range(4):
                            param_phase[p, count_animal, count_trial] = st.circmean(param_mat[0][p], nan_policy='omit')
                    elif param == 'stance_speed':
                        for p in range(4):
                            stance_speed[p, count_animal,count_trial] = np.nanmean(param_mat[p])
                    else:
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                    for count_paw, paw in enumerate(paws):
                        param_paw[count_p, count_animal, count_paw,count_trial] = np.nanmean(param_mat[count_paw])

                if compare_baselines:
                    for count_p, param in enumerate(param_gait_name):
                        param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                        param_gait[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])

        # BASELINE SUBTRACTION OF PARAMETERS
        if bs_bool:
            param_sym_bs = np.zeros(np.shape(param_sym))
            param_paw_bs = np.zeros(np.shape(param_paw))
            for p in range(np.shape(param_sym)[0]-1):
                for a in range(np.shape(param_sym)[1]):
                    # Compute baseline
                    if stim_start == split_start:
                        bs_mean = np.nanmean(param_sym[p, a, :stim_start-1])
                        bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_start-1])
                    if stim_start < split_start:
                        bs_mean = np.nanmean(param_sym[p, a, stim_start-1:split_start-1])
                        bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, stim_start-1:split_start-1])
                    # Subtract
                    param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
                    for count_paw in range(4):
                        param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean
        else:
            param_sym_bs = param_sym

        if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:      # If we are comparing left and right we will have them both in experiment names
           param_sym_bs = -param_sym_bs

        # Compare baseline symmetry with and without stim
        if compare_baselines:
            param_no_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))          # Stance speed is not considered in the params
            param_no_stim[:] = np.NaN
            param_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))
            param_stim[:] = np.NaN
            # Symmetry parameters
            for p in range(np.shape(param_sym)[0]-1):
                for a in included_animals_id:            #range(np.shape(param_sym)[1]):
                    param_no_stim[p, a, :] = param_sym[p, a, :stim_start-1]
                    param_stim[p, a, :] = param_sym[p, a, stim_start-1:stim_start+stim_duration-1]        # :8]
            # Gait parameters
            start_ind=np.shape(param_sym)[0]-1
            for p in range(np.shape(param_gait)[0]):
                for a in included_animals_id:           #range(np.shape(param_gait)[1]):
                    param_no_stim[start_ind+p, a, :] = param_gait[p, a, :stim_start-1]
                    param_stim[start_ind+p, a, :] = param_gait[p, a, stim_start-1:stim_start+stim_duration-1]           # :8]
            

            # Plot and compare
            import math
            param_names = param_sym_name[:-1]+param_gait_name
            fig_stim_cmp, ax_stim_cmp = plt.subplots(3,5)           # len(param_sym_name)+len(param_gait_name))
            rc = [[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[1,1],[1,2],[1,3],[1,4],[2,0],[2,1],[2,2],[2,3],[2,4]]
            for p in range(np.shape(param_stim)[0]):
                ax_stim_cmp[rc[p][0],rc[p][1]].bar([0,1], [np.nanmean(np.nanmean(param_no_stim[p,:,:], axis=1)), np.nanmean(np.nanmean(param_stim[p,:,:], axis=1))], yerr=[np.nanstd(np.nanmean(param_no_stim[p,:,:], axis=1)),np.nanstd(np.nanmean(param_stim[p,:,:], axis=1))], align='center', color=['gray', experiment_colors_dict[experiment_name]], alpha=0.5, ecolor='black', capsize=6)
                ax_stim_cmp[rc[p][0],rc[p][1]].plot([np.nanmean(param_no_stim[p,:,:], axis=1), np.nanmean(param_stim[p,:,:], axis=1)],'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
                ax_stim_cmp[rc[p][0],rc[p][1]].set_title(param_names[p], size=9)
                ax_stim_cmp[rc[p][0],rc[p][1]].set_xticks([])
            fig_stim_cmp.tight_layout()
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                
                plt.savefig(paths_save[path_index] + 'compare_baselines', dpi=128)
                





        for p in range(np.shape(param_sym)[0] - 1):
            # Plot learning curve for individual animals
            fig = pf.plot_learning_curve_ind_animals(param_sym_bs, p, param_label, animal_list, animal_colors_dict, split_intervals=[split_start, split_duration], stim_intervals=[stim_start, stim_duration])
            # Save plot
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                if bs_bool:
                    fig.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_bs', dpi=128)
                else:
                    fig.savefig(paths_save[path_index] + param_sym_name[p] + '_sym_non_bs', dpi=128)
        plt.close('all')

    # PLOT ANIMAL AVERAGE FOR EACH SESSION
    param_sym_multi[path] = {}
    if single_animal_analysis == 0:
        for p in range(np.shape(param_sym)[0]):
            param_sym_bs_ave = param_sym_bs[p, included_animals_id, :]
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            if uniform_ranges:
                ax.set(ylim=axes_ranges[param_sym_name[p]])
                rectangle = plt.Rectangle((split_start - 0.5, axes_ranges[param_sym_name[p]][0]), split_duration,
                                            axes_ranges[param_sym_name[p]][1] - axes_ranges[param_sym_name[p]][0],
                                            fc='lightblue', alpha=0.3)
            else:
                rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_sym_bs_ave[:, :].flatten())), split_duration,
                                            np.nanmax(param_sym_bs_ave[:, :].flatten()) - np.nanmin(param_sym_bs_ave[:, :].flatten()),
                                            fc='lightblue', alpha=0.3)
            plt.gca().add_patch(rectangle)
            ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym_bs_ave)[0]):
                plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])), param_sym_bs_ave[a, :], linewidth=1, color = animal_colors_dict[included_animal_list[a]], label=animal_list[included_animals_id[a]])
            ax.legend(frameon=False)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), np.nanmean(param_sym_bs_ave, axis=0), color=experiment_colors_dict[experiment_name], linewidth=3)
            ax.set_xlabel('Trial', fontsize=24)
            ax.set_ylabel(param_label[p]+' symmetry', fontsize=24)          #   param_sym_name[p].replace('_', ' '),
            #if p == 2:
            #    plt.gca().invert_yaxis()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
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
                ax.set_xlabel('Trial', fontsize=24)
                ax.set_ylabel(param_label[p]+' asymmetry', fontsize=24)            #      param_sym_name[p].replace('_', ' ')
                #if p == 2:
                #    plt.gca().invert_yaxis()
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
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
                ax.set_xlabel('Trial', fontsize = 24)
                ax.set_ylabel('Stance speed', fontsize = 24)
                ax.tick_params(axis='x',labelsize = 20)
                ax.tick_params(axis='y',labelsize = 20)
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
    
#included_animal_list = ['VIV41330', 'VIV41329']   #               'MC18737','MC19107']         #[ 'MC19022','MC19082','MC19123','MC19124','MC19214']  

#included_animals_id = [animal_list.index(i) for i in included_animal_list]
# MULTI-SESSION PLOT
if single_animal_analysis==0 and (len(paths)>0 or len(control_path)>0):
    if len(control_path)>0:
        current_experiment_names = ['control']
        current_experiment_colors = ['black']
        current_bar_labels = ['control']
    
    for p in range(np.shape(param_sym)[0]):
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
        plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
        ax_multi.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
        ax_multi.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
        ax_multi.set_xlabel('1-min trial', fontsize=24)
        ax_multi.set_ylabel(param_label[p]+' asymmetry', fontsize=24)
        if uniform_ranges:
            ax_multi.set(ylim=axes_ranges[param_sym_name[p]])
        #if p == 2:
           # plt.gca().invert_yaxis()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax_multi.spines['right'].set_visible(False)
        ax_multi.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            if bs_bool:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session', dpi=128)
            else:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_non_bs_average_with_control_multi_session', dpi=128)
 
        # LEARNING PARAMETERS (bar plots) - each one will be num_experiments x num_animals
        initial_error = []                      
        learning = []
        aftereffect = []
        learning_sym_change = []
        aftereffect_sym_change = []
        stat_initial_error = []
        stat_learning = []
        stat_aftereffect = []
        stat_learning_sym_change = []
        stat_aftereffect_sym_change = []
        if split_duration==0:
            split_start = stim_start
            split_duration = stim_duration

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
            for exp in experiment_names:
                if exp in path:
                    experiment_name = exp
                    current_experiment_names.append(experiment_name)
                    current_experiment_colors.append(experiment_colors_dict[exp])
                    current_bar_labels.append(exp)
            '''
            # Flip signs to have good control learning always positive
                    
            if (param_sym_name[p] == 'double_support' and control_ses == 'right') or ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'left'):
                initial_error.append(np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))  
                learning.append(-(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1)))
                aftereffect.append(-np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
            elif ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'right') or (param_sym_name[p] == 'double_support' and control_ses == 'left'):
                initial_error.append(-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))         
                learning.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))
                aftereffect.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
            '''
            
            initial_error.append(np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))  
            learning.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))
            aftereffect.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
               
            # Compare to first column (if there is control, it should go to first column)
            if compute_statistics and path_index>0:
                print(aftereffect[0], aftereffect[path_index])
                print(['param ', param_sym_name[p], st.wilcoxon(aftereffect[0], aftereffect[path_index])])
                stat_initial_error.append(st.wilcoxon(initial_error[0], initial_error[path_index]).pvalue<significance_threshold)
                stat_learning.append(st.wilcoxon(learning[0], learning[path_index]).pvalue<significance_threshold)
                stat_aftereffect.append(st.wilcoxon(aftereffect[0][~np.isnan(aftereffect[0]) and ~np.isnan(aftereffect[path_index])], aftereffect[path_index][~np.isnan(aftereffect[0]) and ~np.isnan(aftereffect[path_index])]).pvalue<significance_threshold)
            path_index+=1

        learning_sym_change=100*np.divide(np.array(learning),np.array(initial_error))
        aftereffect_sym_change=100*np.divide(np.array(aftereffect),np.array(initial_error))
        #if compute_statistics and path_index>0:
         #   for path_index in range(len(paths)):
          #      stat_learning_sym_change.append(st.wilcoxon(learning_sym_change[0], learning_sym_change[path_index]).pvalue<significance_threshold)
          #      stat_aftereffect_sym_change.append(st.wilcoxon(aftereffect_sym_change[0], aftereffect_sym_change[path_index]).pvalue<significance_threshold)
        fig_bar, ax_bar = plt.subplots(2,3)
        bars = ax_bar[0,0].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(initial_error, axis=1), yerr=np.nanstd(initial_error, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,0].plot(initial_error,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,0].set_ylabel(param_sym_name[p]+' (mm)')
        ax_bar[0,0].set_title('init. error', size=9)
        ax_bar[0,1].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(learning, axis=1), yerr=np.nanstd(learning, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        ax_bar[0,1].set_title('late-early', size=9)
        ax_bar[0,2].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(aftereffect, axis=1), yerr=np.nanstd(aftereffect, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,2].plot(list(range(1,len(paths)+len(control_path))),[max(max(np.nanmean(aftereffect, axis=1)+np.nanstd(aftereffect, axis=1)),0)*i if i==1 else math.nan*i for i in stat_aftereffect ],'*', color='black')
        #ax_bar[0,2].plot(aftereffect,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,2].set_title('aftereffect', size=9)
        ax_bar[1,1].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(learning_sym_change, axis=1), yerr=np.nanstd(learning_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,1].plot(learning_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,1].set_title('% change late-early', size=9)
        ax_bar[1,1].set_ylabel('% sym change')
        ax_bar[1,2].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(aftereffect_sym_change, axis=1), yerr=np.nanstd(aftereffect_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,2].plot(aftereffect_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,2].set_title('% change aftereffect', size=9)
        # Add single animal data
        for a in range(len(initial_error[0])):
            ax_bar[0,0].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(initial_error)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
            ax_bar[0,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
            ax_bar[0,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
            ax_bar[1,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
            ax_bar[1,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
               
        # Add zero line
        for ax in ax_bar.flatten():
            ax.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
            if uniform_ranges:
                ax.set(ylim= bars_ranges[param_sym_name[p]])      #   [-4.5,max(abs(np.array(axes_ranges[param_sym_name[p]])))])

        if compute_statistics and path_index>0:
            ax_bar[0,0].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[(bars_ranges[param_sym_name[p]][1]-0.5)*i if i==1 else math.nan*i for i in stat_initial_error ],'*', color='black')
            ax_bar[0,1].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[(bars_ranges[param_sym_name[p]][1]-0.5)*i if i==1 else math.nan*i for i in stat_learning ],'*', color='black')
            ax_bar[0,2].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[(bars_ranges[param_sym_name[p]][1]-0.5)*i if i==1 else math.nan*i for i in stat_aftereffect ],'*', color='black')
            #ax_bar[1,1].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[bars_ranges[param_sym_name[p][1]]*i if i==1 else math.nan*i for i in stat_learning_sym_change ],'*', color='black')
            #ax_bar[1,2].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[bars_ranges[param_sym_name[p][1]]*i if i==1 else math.nan*i for i in stat_aftereffect_sym_change ],'*', color='black')
            print('stat '+param_sym_name[p]+': '+str(stat_aftereffect))
        
        for ax in ax_bar.flatten():
            ax.set_xticks([])
        fig_bar.delaxes(ax_bar[1,0])
        fig_bar.suptitle(param_sym_name[p])
        fig_bar.tight_layout()
        fig_bar.legend(bars, current_bar_labels,
           loc="lower left",   
           borderaxespad=3 
           )

                               
        if print_plots_multi_session:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session_barplot', dpi=96)       #_gray_not_confirmed


     