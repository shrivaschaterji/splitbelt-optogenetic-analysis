import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import math
import pandas as pd
import plotting_functions as pf
import pickle

# Set the default font
plt.rcParams['font.family'] = 'Arial'


# Inputs
plot_continuous = 0
compare_baselines = 0
compute_statistics = 1
scatter_single_animals = 1
significance_threshold = 0.05

# Ranges for pre-defined range of axes and bars (if uniform_ranges = 1)
uniform_ranges = 0

# Tied
#axes_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#bars_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
axes_ranges = {'coo': [-4, 6], 'step_length': [-9, 5], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-7, 7], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
bars_ranges = {'coo': [-4, 6], 'step_length': [-9, 5], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-7, 7], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
# Rfast
#axes_ranges = {'coo': [-6, 2], 'step_length': [-12, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#bars_ranges = {'coo': [-2, 4], 'step_length': [-5, 9], 'double_support': [-9, 5], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#axes_ranges = {'coo': [-8, 12], 'step_length': [-9, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-7, 7], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#bars_ranges = {'coo': [-2.5, 5], 'step_length': [-9, 5], 'double_support': [-9, 5], 'coo_stance': [-5, 5], 'swing_length': [-7, 7], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
# Lfast
#axes_ranges = {'coo': [-2, 4], 'step_length': [-3, 9], 'double_support': [-10, 5], 'coo_stance': [-5, 5], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#bars_ranges = {'coo': [-4, 2], 'step_length': [-9, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#axes_ranges = {'coo': [-3.5, 5.5], 'step_length': [-6, 9], 'double_support': [-10, 5], 'coo_stance': [-8, 3], 'swing_length': [-13, 3], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
#bars_ranges = {'coo': [-4, 2.5], 'step_length': [-5, 4], 'double_support': [-5, 10], 'coo_stance': [-5, 3], 'swing_length': [-4, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}


# Lists of experiment names and paths for each experiment - it is possible to have only one element
experiment_names = ['100sw', '200st']  #['ChR2']       #['control','stance stim','swing stim']           #  ['control', 'stance onset', 'swing onset']             #'ChR2']           #'right fast', 'left fast']          #,'stance stim', 'swing stim']           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'

paths = [
        #'C:\\Users\\User\\Desktop\\test_animals\\Tied belt sessions\\tied stance stim HISTOsel\\',
        #'C:\\Users\\User\\Desktop\\test_animals\\Tied belt sessions\\tied swing stim HISTOsel\\'
        #'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments JAWS RT\\Tied belt sessions\\tied stance stim HISTOsel\\',
        #'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments JAWS RT\\Tied belt sessions\\tied swing stim HISTOsel\\'
        #'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments JAWS RT\\Split belt sessions\\split left fast control HISTOsel\\',
        #'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments JAWS RT\\Split belt sessions\\split left fast stance stim HISTOsel\\',
        #'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments JAWS RT\\Split belt sessions\\split left fast swing stim HISTOsel\\'
        'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments ChR2LE RT\\Tied belt sessions\\tied 100sw50ms stim HISTOsel\\',
        'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Tati&Ali&INDP2025\\Behavior\\Experiments ChR2LE RT\\Tied belt sessions\\tied 200st50ms stim HISTOsel\\'
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241111 split right fast control batch#4C\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241112 split right fast stance onset stim 200st IO batch#4C\\',

     ]

experiment_colors_dict = {'trial stim':'purple', 'stance stim':'orange','swing stim': 'green', 'control':'black', 'ChR2': 'cyan',
                          'stance onset':'green','swing onset': 'orange',
                          'right fast no-stim': 'gray',     # 'blue', 
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen',
                          'right fast': 'blue',
                          'left fast': 'black',
                          'WT': 'green',
                          '100sw': 'skyblue',
                          '200st': 'royalblue'
                          }      # stim on: trial stance swing    'trial stim':'purple', 
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
animal_colors_dict = {
    # jaws
    'MC16846': "#FFD700",'MC16848':"#BBF90F",'MC16850': "#15B01A",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
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
    'VIV49571': "#FFD700",'VIV49572':"#BBF90F",'VIV49604': "#15B01A",'VIV49605': animal_colors[0],
    #HGM
    'MC11231': "#FFD700",'MC11232':"#BBF90F",'MC11234': "#15B01A",'MC11235': animal_colors[0], 'MC24409': animal_colors[1],
    'MC24410': '#CCCCFF','MC24411': '#660033','MC24412': animal_colors[4], 'MC24413': animal_colors[5],
    'MC1262': animal_colors[0],'MC1263':  animal_colors[1],'MC1328': animal_colors[2],'MC1329': animal_colors[3],'MC1330':  animal_colors[4],
    'A1': "#FFD700",'A2':"#BBF90F",'A3': "#15B01A",'A4': '#0000FF','A5': '#00FF00', 'MC1705': '#F08080', 'V1': '#FA8072',
    'V2': '#5C62D6', 'V3': '#FF0000', 'V4': '#BC8F8F', 'MC1659': '#BC8F8F', 'MC1660': '#FF4500','MC1661': '#CCCCFF','MC1663': '#660033','MC1664': '#00FFFF',
    # extra-zombies
    # Linj
    'VIV47094': "#FFD700",'VIV47095':"#BBF90F",'VIV47147': "#0000FF",'VIV47116': animal_colors[0], 'VIV47212': animal_colors[1],
    'VIV49409': animal_colors[2], 'VIV49410': animal_colors[3], 'VIV49411': animal_colors[4], 'VIV49412': animal_colors[5], 
    # Rinj
    'VIV49574':animal_colors[6], 'VIV49931': animal_colors[7],'VIV49933': animal_colors[8],
    'VIV49934': '#CCCCFF', 'VIV49939': '#00FF00', 'VIV49940': '#FF4500', 'VIV49935': '#BC8F8F', 'VIV49941': '#F08080', 
    'VIV50033': "#FFD700",'VIV50034':"#BBF90F",'VIV50051': "#15B01A",'VIV50052': '#660033',
    # WT no fiber and no injection
    'MC2166': "#FFD700",'MC2168':"#BBF90F",'MC2585': "#15B01A",'MC2586': animal_colors[0], 'MC2587': animal_colors[1],
    'MC2588': '#CCCCFF','MC2589': '#660033','MC2590': animal_colors[4], 'MC2591': animal_colors[5],'MC2592': animal_colors[6]
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


included_animal_list = []


session = 1
Ntrials = 28
stim_start = 9
split_start = 9
stim_duration = 10
split_duration = 10
plot_rig_signals = 0
print_plots = 1
print_plots_multi_session = 1
bs_bool = 1
control_ses = 'left'
control_path = []       #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']      #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\14092023 split right fast control\\'] #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']   #]         # This should be a list; if empty, we have no control (e.g. in tied sessions)
control_filename = 'split_'+control_ses+'_fast_control_params_sym_bs.npy'

paw_colors = {'FR': 'red', 'HR': 'magenta', 'FL': 'blue', 'HL': 'cyan'}
paw_otrack = 'FR'
paws = ['FR', 'HR', 'FL', 'HL']
import online_tracking_class
import locomotion_class
otrack_classes =  []
locos = []
paths_save = []
param_sym_multi = {}
param_paw_multi = {}
path_index = 0
for path in paths:
    to_save_param_mat = False
    # If param_mat_saved exists in the path, load it
    if os.path.exists(path + 'param_mat_saved.pkl'):
        with open(path + 'param_mat_saved.pkl', 'rb') as f:
            param_mat_saved = pickle.load(f)
    else: # Create it
        param_mat_saved = {}
        to_save_param_mat = True
    
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
    included_animal_id = [animal_list.index(i) for i in included_animal_list]
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])


    # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
    # GAIT PARAMETERS ACROSS TRIALS
    param_gait_name = ['coo', 'coo_stance', 'swing_length', 'step_length']
    #['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stride_duration', 'swing_duration', 'stance_duration', 'swing_velocity','stance_speed','body_center_x_stride','body_speed_x','duty_factor','candence','phase_st']
    param_sym_name_label_map = {'coo': 'Center of\noscillation (mm)', 'step_length': 'Step length (mm)', 
                    'coo_stance': 'Spatial motor\noutput (mm)', 'swing_length': 'Swing length(mm)'}     #, 'phase_st': 'Stance phase', 'stance_speed': 'Stance speed'}
    param_sym_label = list(param_sym_name_label_map.values())
    param_sym_name = list(param_sym_name_label_map.keys())
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
        if not animal in param_mat_saved.keys():
            param_mat_saved[animal] = {}
            to_save_param_mat = True
        session = int(session_list[count_animal])
        #TODO: check if this filelist needs to be emptied first!
        filelist = locos[path_index].get_track_files(animal, session)
        for count_p, param in enumerate(param_sym_name):
            if not param in param_mat_saved[animal].keys():
                param_mat_saved[animal][param] = {}
                to_save_param_mat = True
            param_trials = np.zeros((Ntrials))
            param_trials[:] = np.NaN
            st_strides_trials = []
            for f in filelist:      # Loop over all trials
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                # if we have param_mat from the saved file, load it, otherwise do analysis
                if count_trial+1 in param_mat_saved[animal][param]:
                    param_mat = param_mat_saved[animal][param][count_trial+1]
                else:
                    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                    [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                    st_strides_trials.append(st_strides_mat)
                    paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                    # Add to param_mat_saved variable
                    param_mat_saved[animal][param][count_trial+1] = param_mat
                    to_save_param_mat = True
                if param == 'stance_speed':
                    for p in range(4):
                        stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
                elif param == 'phase_st':
                    for p in range(4):
                        param_phase[p, count_animal, count_trial] = st.circmean(param_mat[0][p], nan_policy='omit')
                else:
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                
                for count_paw, paw in enumerate(paws):
                    param_paw[count_p, count_animal, count_paw,count_trial] = np.nanmean(param_mat[count_paw])

                # Compute and plot also the continuous (not separated by trial) version of the symmetry parameter
                if plot_continuous:
                    param_trials[count_trial]=param_mat
                    st_strides_trials[count_trial]=st_strides_mat

                    # TODO: the following lines should go outside the loop over trials
                    [stride_idx, trial_continuous, sym_param_time, sym_param_values] = locos[path_index].param_continuous_sym(param_trials, st_strides_trials, trials, 'FR', 'FL', 1, 1)
                    fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
                    sym_param_time_start = sym_param_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
                    sym_param_time_duration = sym_param_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(locos[path_index].trial_time*stim_duration)
                    rectangle = plt.Rectangle((sym_param_time_start, np.nanmin(sym_param_values)), sym_param_time_duration-sym_param_time_start, np.nanmax(sym_param_values)+np.abs(np.nanmin(sym_param_values)), fc=experiment_colors_dict[experiment_name], alpha=0.3)
                    plt.gca().add_patch(rectangle)
                    ax.plot(sym_param_time, sym_param_values, color='black')
                    ax.set_xlabel('time (s)')
                    ax.set_title('continuous step length')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    if print_plots:
                        if not os.path.exists(paths_save[path_index]):
                            os.mkdir(paths_save[path_index])
                        plt.savefig(paths_save[path_index] + animal_list[a] + '_sl_sym_continuous', dpi=96)
                    '''
                    param_mat_sym = locos[path_index].compute_continuous_sym_gaitparam(param_mat, st_strides_mat, 'FR', 'FL')
                    
                    if (count_trial==7 or count_trial==8) and param_sym_name[count_p]=='double_support':
                        
                        fig1, ax1 = plt.subplots(figsize=(7, 10), tight_layout=True)
                        plt.plot(param_mat_sym, linewidth=2)
                        plt.title([param_sym_name[count_p], ' trial ', str(count_trial)])
                        plt.show()
                        # With moving avg
                        series = pd.Series(param_mat_sym)
                        # Compute a moving average with a window of 3, ignoring NaNs
                        window_size = 2
                        moving_avg = series.rolling(window=window_size, min_periods=1).mean()
                        if count_trial==7:
                            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
                        ax.plot(moving_avg, linewidth=2)
                        ax.set_title([param_sym_name[count_p], ' trial ', str(count_trial), ' moving avg'])
                        ax.set_ylim([-25,30])
                        print("avg DS!!!!!!!!!!!!!!!!!",np.nanmedian(param_mat_sym))
                        plt.show()
                        '''
                

            if compare_baselines:
                for count_p, param in enumerate(param_gait_name):
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                    param_gait[count_p, count_animal, count_trial] = np.nanmedian(param_mat[0])

    # Save the param_mat_saved variable
    if to_save_param_mat:
        with open(path + 'param_mat_saved.pkl', 'wb') as f:
            pickle.dump(param_mat_saved, f)

    # BASELINE SUBTRACTION OF PARAMETERS
    if bs_bool:
        param_sym_bs = np.zeros(np.shape(param_sym))
        param_paw_bs = np.zeros(np.shape(param_paw))
        for p in range(np.shape(param_sym)[0]):
            for a in range(np.shape(param_sym)[1]):
                # Compute baseline
                if stim_start == split_start:
                    bs_mean = np.nanmean(param_sym[p, a, :stim_start-1])
                    
                if stim_start < split_start:
                    bs_mean = np.nanmean(param_sym[p, a, stim_start-1:split_start-1])
                    bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, stim_start-1:split_start-1])
                # Subtract
                param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
                for count_paw in range(4):
                    bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_start-1])
                    param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean
    else:
        param_sym_bs = param_sym

    # Transpose param_paw_bs to switch dimensions to (parameter, paw, animal, trial)
    param_paw_bs = np.transpose(param_paw_bs, (2, 0, 1, 3))

    if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:      # If we are comparing left and right we will have them both in experiment names
        param_sym_bs = -param_sym_bs

    # Compare baseline symmetry with and without stim
    if compare_baselines:
        param_no_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))          # Stance speed is not considered in the params
        param_no_stim[:] = np.NaN
        param_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))
        param_stim[:] = np.NaN
        # Symmetry parameters
        for p in range(np.shape(param_sym)[0]):
            for a in included_animal_id:            #range(np.shape(param_sym)[1]):
                param_no_stim[p, a, :] = param_sym[p, a, :stim_start-1]
                param_stim[p, a, :] = param_sym[p, a, stim_start-1:stim_start+stim_duration-1]        # :8]
        # Gait parameters
        start_ind=np.shape(param_sym)[0]
        for p in range(np.shape(param_gait)[0]):
            for a in included_animal_id:           #range(np.shape(param_gait)[1]):
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
            

    for p in range(np.shape(param_sym)[0]):
        if param_sym_name[p] == 'stance_speed':
            continue
        # Plot learning curve for individual animals
        fig = pf.plot_learning_curve_ind_animals(param_sym_bs, p, param_sym_name_label_map, animal_list, animal_colors_dict, {'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]})
        # Save plot
        if print_plots:
            pf.save_plot(fig, paths_save[path_index], param_sym_name[p], plot_name='ind_animals', bs_bool=bs_bool)
            
    plt.close('all')

    # Plot learning curve for individual limbs per animal -- PAW
    for paw in range(np.shape(param_paw_bs)[0]):   # loop over each paw
        for p in range(np.shape(param_paw_bs)[1]): # loop over each parameter
            fig_paw = pf.plot_learning_curve_ind_animals(param_paw_bs[paw], p, param_sym_name_label_map, animal_list, animal_colors_dict, {'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]})
            # Save plot
            if print_plots:
                pf.save_plot_with_paw(fig_paw, paths_save[path_index], param_sym_name[p], paws[paw], plot_name='ind_animals', bs_bool=bs_bool)

    plt.close('all')


    # PLOT ANIMAL AVERAGE with INDIVIDUAL ANIMALS FOR EACH SESSION
    param_sym_multi[path] = {}
    for p in range(np.shape(param_sym)[0]):
        if param_sym_name[p] == 'stance_speed':
            continue
        param_sym_bs_ave = param_sym_bs[p, included_animal_id, :]
        fig = pf.plot_learning_curve_ind_animals_avg(param_sym_bs_ave, p, param_sym_name_label_map, animal_list, [included_animal_list, included_animal_id],
                                                        [animal_colors_dict, experiment_colors_dict], experiment_name, intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]}, 
                                                        ranges=[uniform_ranges, axes_ranges])
        # Save plot
        if print_plots:
            pf.save_plot(fig, paths_save[path_index], param_sym_name[p], plot_name='average', bs_bool=bs_bool)
            
        # Save param_sym for multi-session plot (in case we have multiple sessions to analyse/plot), with only the included animals
        param_sym_multi[path][p] = param_sym_bs_ave
    plt.close('all')


     # PLOT PAW AVERAGE with INDIVIDUAL ANIMALS FOR EACH SESSION   -- PAW
    param_paw_multi[path] = {}
    for paw_idx, paw_name in enumerate(paws):
        if paw_name not in param_paw_multi[path]:
            param_paw_multi[path][paw_name] = {}
        
        for p in range(np.shape(param_paw_bs)[1]):
                
            param_paw_bs_ave = param_paw_bs[paw_idx][p, included_animal_id, :]
            figpaw = pf.plot_learning_curve_ind_animals_avg_paw(param_paw_bs_ave, p, param_sym_name_label_map, animal_list, [included_animal_list, included_animal_id],
                                                            [animal_colors_dict, experiment_colors_dict], paw_colors, paw_name, intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]}, 
                                                            ranges=[uniform_ranges, axes_ranges])
            # Save plot
            if print_plots:
                pf.save_plot_with_paw(figpaw, paths_save[path_index], param_sym_name[p], paw_name, plot_name='average', bs_bool=bs_bool)
                
            # Save param data with a nested dictionary structure: path -> paw_name -> parameter
            param_paw_multi[path][paw_name][p] = param_paw_bs_ave
    plt.close('all')

    # PLOT STANCE SPEED for ALL ANIMALS
    if 'stance_speed' in param_sym_name:
        for a in range(np.shape(stance_speed)[1]):
            data = stance_speed[:, a, :]
            fig_stance_speed = pf.plot_stance_speed(data, animal_list[a], paw_colors, {'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]})
            
            if print_plots:
                pf.save_plot(fig_stance_speed, paths_save[path_index], animal_list[a], plot_name='_stancespeed', dpi=96)
    plt.close('all')




    path_index = path_index+1
    
# Changing shape of param_paw_multi to [paws, paths, parameters, animals, trials]
paw_path_multi = {}

# Initialize dictionary structure
for paw_name in paws:
    paw_path_multi[paw_name] = {}
    for path in paths:
        paw_path_multi[paw_name][path] = {}

# Fill the new dictionary from the existing param_paw_multi
for path in paths:
    for paw_name in paws:
        if path in param_paw_multi and paw_name in param_paw_multi[path]:
            for p in param_paw_multi[path][paw_name]:
                if p not in paw_path_multi[paw_name][path]:
                    paw_path_multi[paw_name][path][p] = param_paw_multi[path][paw_name][p]


# MULTI-SESSION PLOT
for p in range(np.shape(param_sym)[0]):
    if param_sym_name[p] == 'stance_speed':
            continue
    
    fig_multi = pf.plot_learning_curve_avg_compared(param_sym_multi, p, param_sym_name_label_map, [included_animal_list, included_animal_id], experiment_colors_dict, experiment_names, intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]}, ranges=[uniform_ranges, axes_ranges])
    
    if print_plots:
        pf.save_plot(fig_multi, paths_save[0], param_sym_name[p], plot_name='average_multi_session', bs_bool=bs_bool)


    # LEARNING PARAMETERS
    current_experiment_colors = [experiment_colors_dict[key] for key in experiment_names if key in experiment_names]

    # Compute learning params and statistics compared to first column (if there is control, it should go to first column) 
    # learning_params_dict and stat_learning_params_dict variables
    learning_params_dict = {}
    stat_learning_params_dict = {}

    for path_index, path in enumerate(paths):
        locos[path_index].compute_learning_params(learning_params_dict, param_sym_multi[path][p], intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]})
        
    if compute_statistics and path_index>0:
        locos[path_index].compute_stat_learning_param(learning_params_dict, stat_learning_params_dict, param_sym_name[p], thr=significance_threshold)

    # Bar plot of ALL learning parameters
    fig_bar_all = pf.plot_all_learning_params(learning_params_dict, [param_sym_name[p], param_sym_label[p]], included_animal_list, experiment_names, current_experiment_colors, animal_colors_dict, stat_learning_params=stat_learning_params_dict, scatter_single_animals=scatter_single_animals, ranges=[uniform_ranges, bars_ranges])
            
    if print_plots_multi_session:
        pf.save_plot(fig_bar_all, paths_save[0], param_sym_name[p], plot_name='multi_session_barplot_all', bs_bool=bs_bool)


    # Single learning parameter alone. Can be 'initial error', 'adaptation', 'after-effect', '% change adaptation', '% change after-effect'
    to_plot_separately = ['adaptation', 'after-effect']

    for lp_name in to_plot_separately:
        # Bar plot
        fig_separate = pf.plot_learning_param(learning_params_dict[lp_name], [param_sym_name[p], param_sym_label[p]], lp_name, included_animal_list, experiment_names, current_experiment_colors, animal_colors_dict, stat_learning_params=stat_learning_params_dict, scatter_single_animals=scatter_single_animals, ranges=[uniform_ranges, bars_ranges])
        # Scatter and avg plot
        fig_scatter = pf.plot_learning_param_scatter(learning_params_dict[lp_name], [param_sym_name[p], param_sym_label[p]], lp_name, included_animal_list, experiment_names, current_experiment_colors, stat_learning_params=stat_learning_params_dict, ranges=[uniform_ranges, bars_ranges])
        if print_plots_multi_session:
            pf.save_plot(fig_separate, paths_save[0], param_sym_name[p], plot_name='bar_scatterplot_'+lp_name, bs_bool=bs_bool, dpi=1200)  
            pf.save_plot(fig_scatter, paths_save[0], param_sym_name[p], plot_name='avg_scatterplot_'+lp_name, bs_bool=bs_bool, dpi=120) 


    # Multi-session plot for each paw -- PAW --------------------------
for paw_name in paws:
        
    # Restructure data to match expected format for plot_learning_curve_avg_compared
    paw_data_for_plotting = {}
    for path in paths:
        if path in paw_path_multi[paw_name]:
            # Get all parameters for this paw and path
            parameters_data = []
            parameters_indices = []

            # Collect all parameters data and their indices
            for p in sorted(paw_path_multi[paw_name][path].keys()):
                parameters_data.append(paw_path_multi[paw_name][path][p])
                parameters_indices.append(p)
                
            if parameters_data:
                # Stack all parameters data along a new dimension to get (parameters, animals, trials)
                all_params_data = np.stack(parameters_data)
                # Add to the dictionary
                paw_data_for_plotting[path] = all_params_data

    # Now iterate through each parameter for this paw
    for p_idx, p in enumerate(parameters_indices):
        # Only proceed if we have data for this paw and parameter
        if paw_data_for_plotting:
            fig_multi_paw = pf.plot_learning_curve_avg_compared(paw_data_for_plotting, p, param_sym_name_label_map, [included_animal_list, included_animal_id], 
                experiment_colors_dict, experiment_names, intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]}, 
                ranges=[uniform_ranges, axes_ranges]
            )
            
            if print_plots:
                pf.save_plot_with_paw(fig_multi_paw, paths_save[0], param_sym_name[p], paw_name, 
                                    plot_name='average_multi_session', bs_bool=bs_bool)

            # LEARNING PARAMETERS for this paw
            current_experiment_colors = [experiment_colors_dict[key] for key in experiment_names if key in experiment_colors_dict]

            # Compute learning params and statistics for this paw
            learning_params_dict_paw = {}
            stat_learning_params_dict_paw = {}

            for path_index, path in enumerate(paths):
                if path in paw_data_for_plotting:
                    locos[path_index].compute_learning_params(learning_params_dict_paw, paw_data_for_plotting[path][p], 
                        intervals={'split': [split_start, split_duration], 'stim': [stim_start, stim_duration]})
            
            if compute_statistics and len(paths) > 1:
                locos[0].compute_stat_learning_param(learning_params_dict_paw, stat_learning_params_dict_paw, 
                    param_sym_name[p], thr=significance_threshold)

            # Bar plot of ALL learning parameters for this paw
            if learning_params_dict_paw:
                fig_bar_all_paw = pf.plot_all_learning_params(learning_params_dict_paw, [param_sym_name[p], param_sym_label[p]], included_animal_list, 
                    experiment_names, current_experiment_colors, animal_colors_dict, stat_learning_params=stat_learning_params_dict_paw, 
                    scatter_single_animals=scatter_single_animals, ranges=[uniform_ranges, bars_ranges])
                
                if print_plots_multi_session:
                    pf.save_plot_with_paw(fig_bar_all_paw, paths_save[0], param_sym_name[p], paw_name, 
                                        plot_name='multi_session_barplot_all', bs_bool=bs_bool)

                # Single learning parameter plots for this paw
                to_plot_separately = ['adaptation', 'after-effect']
                
                for lp_name in to_plot_separately:
                    if lp_name in learning_params_dict_paw:
                        # Bar plot
                        fig_separate_paw = pf.plot_learning_param(learning_params_dict_paw[lp_name], [param_sym_name[p], param_sym_label[p]], lp_name, included_animal_list, 
                            experiment_names, current_experiment_colors, animal_colors_dict, stat_learning_params=stat_learning_params_dict_paw, 
                            scatter_single_animals=scatter_single_animals, ranges=[uniform_ranges, bars_ranges])
                        
                        # Scatter and avg plot
                        fig_scatter_paw = pf.plot_learning_param_scatter(learning_params_dict_paw[lp_name], [param_sym_name[p], param_sym_label[p]], lp_name, 
                            included_animal_list, experiment_names, current_experiment_colors, stat_learning_params=stat_learning_params_dict_paw, 
                            ranges=[uniform_ranges, bars_ranges])
                        
                        if print_plots_multi_session:
                            pf.save_plot_with_paw(fig_separate_paw, paths_save[0], param_sym_name[p], paw_name, plot_name='bar_scatterplot_'+lp_name, bs_bool=bs_bool, dpi=1200)
                            pf.save_plot_with_paw(fig_scatter_paw, paths_save[0], param_sym_name[p], paw_name, plot_name='avg_scatterplot_'+lp_name, bs_bool=bs_bool, dpi=120)
 