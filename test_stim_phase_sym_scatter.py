import os
import numpy as np
import matplotlib.pyplot as plt

#path inputs
path_loco = 'C:\\Users\\Ana\\Desktop\\Opto Data\\tied stance stim\\'
event = 'swing'

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path_loco)
path_save = path_loco+'grouped output\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])
Ntrials = 28
stim_trials = np.arange(9, 17)

param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
animals = ['MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17670']
offtracks_phase_stim_animals = []
for count_animal, animal in enumerate(animals):
    session = int(session_list[count_animal])
    trials = otrack_class.get_trials(animal)
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, 1)
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    offtracks_phase = loco.get_symmetry_laser_phase_offtracks_df(animal, session, trials, final_tracks_phase, event, laser_on,
                timestamps_session, offtracks_st, offtracks_sw, param_sym_name)
    offtracks_phase_stim = offtracks_phase.loc[(offtracks_phase['trial']>stim_trials[0]-1) & (offtracks_phase['trial']<stim_trials[-1]+1)]
    offtracks_phase_stim_animals.append(offtracks_phase_stim)

plt.figure()
for count_animal, animal in enumerate(animals):
    plt.scatter(offtracks_phase_stim_animals[count_animal]['offset'], np.abs(offtracks_phase_stim_animals[count_animal]['coo']), s=5, color='black')
    plt.scatter(offtracks_phase_stim_animals[count_animal]['onset'], np.abs(offtracks_phase_stim_animals[count_animal]['double_support']), s=5, color='darkgray')

