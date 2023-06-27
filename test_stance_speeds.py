# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:37:45 2020

@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# path inputs
path_loco = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\170423 split left ipsi fast control\\'
print_plots = 1
frames_dFF = 0  # black frames removed before ROI segmentation
paw_colors = ['red', 'magenta', 'blue', 'cyan']
plot_trials = np.array([3, 4, 13, 14, 23])
trials_name = ['last baseline', 'early split', 'late split', 'early washout', 'late washout']
color_trials = ['black', 'red', 'salmon', 'dodgerblue', 'turquoise']

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
count_animal = 0
stance_speed = np.zeros((4, len(animal_list), 28))
for animal in animal_list:
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal, session)
    count_trial = 0
    for f in filelist:
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'stance_speed')
        count_p = 0
        for p in range(4):
            stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
            count_p += 1
        count_trial += 1
    count_animal += 1

# plot stance speed
fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
a = 0
data = stance_speed[:, a, :]
for p in range(4):
    ax.axvline(x=8 + 0.5, color='dimgray', linestyle='--')
    ax.axvline(x=18 + 0.5, color='dimgray', linestyle='--')
    ax.plot(np.linspace(1, len(data[p, :]), len(data[p, :])), data[p, :], color=paw_colors[p], linewidth=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)