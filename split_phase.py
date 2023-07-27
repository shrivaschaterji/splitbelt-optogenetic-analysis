# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:37:45 2020

@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# path inputs
path_loco = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\tests 200323\\split_ipsi_fast_200323\\'
print_plots = 1
frames_dFF = 0  # black frames removed before ROI segmentation

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class

loco = locomotion_class.loco_class(path_loco)
path_save = path_loco + 'grouped output\\'

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

# summary gait parameters
Ntrials = 23
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'phase_st', 'stance_speed']
param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))

trials = np.arange(1, Ntrials+1)
final_tracks_trials_phase_animals = []
for count_animal, animal in enumerate(animal_list):
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    st_strides_trials = []
    sw_strides_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF)
        final_tracks_trials.append(final_tracks)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        count_p = 0
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')
    final_tracks_trials_phase_animals.append(final_tracks_trials_phase)

#%% Plot
#baseline subtracion of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
for p in range(np.shape(param_sym)[0]-2):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p,a,:3])
        param_sym_bs[p,a,:] = param_sym[p,a,:] - bs_mean #will give an error if animals did sessions of different sizes

#plot symmetry baseline subtracted
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
    for a in range(np.shape(param_sym)[1]):
        plt.plot(np.linspace(1,len(param_sym_bs[p,a,:]),len(param_sym_bs[p,a,:])),param_sym_bs[p,a,:], label = animal_list[a], linewidth = 2)
    ax.set_xlabel('Trial', fontsize = 20)
    ax.legend(frameon = False)
    ax.set_ylabel(param_sym_name[p].replace('_',' '), fontsize = 20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if print_plots:
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save+param_sym_name[p]+'_sym_bs', dpi=128)