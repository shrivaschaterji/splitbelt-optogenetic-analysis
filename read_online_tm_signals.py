# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023

@author: Ana
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

paw_otrack = 'FR'
frame_width = 1248
frame_height = 420
path_main = 'C:\\Users\\Ana\\Documents\\PhD\Projects\\Online Stimulation Treadmill\\OTrack_tests_slicing_batchsizes_170223\\'
subdir = 'Test condition Otrack nose beyond paws slice 1 batchsize8 probably'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16946'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

#READ CAMERA TIMESTAMPS AND FRAME COUNTER
[timestamps_session, frame_counter_session, frame_counter_0_session, timestamps_0_session] = otrack_class.get_session_metadata(plot_data)

#READ SYNCHRONIZER SIGNALS
[trial_signal_session, sync_signal_session] = otrack_class.get_synchronizer_data(plot_data)

#READ ONLINE DLC TRACKS
[otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(frame_counter_0_session, timestamps_0_session)

#READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session)

# MATCH FRAME NR WITH FRAME NR OF OTRACKS, SHOULD BE THE SAME IN BONSAI ALREADY
# otracks_frame_counter_globalframe_st = []
# otracks_frame_counter_globalframe_sw = []
# for c in range(np.shape(otracks)[0]):
#     otracks_frame_counter_globalframe_st.append(frame_counter_session[0][np.where(frame_counter_session[0]-otracks_frame_counter_st[c])[0]])
#     otracks_frame_counter_globalframe_sw.append(frame_counter_session[0][np.where(frame_counter_session[0]-otracks_frame_counter_sw[c])[0]])

trial = 0
plt.figure()
plt.scatter(offtracks_st[trial][:, 1], np.repeat(1, len(offtracks_st[trial][:, 0])), color='black')
for i in range(len(otracks_st[trial][:, 1])):
    plt.axvline(otracks_st[trial][i, 1], color='red')

# MEASURE LATENCY BETWEEN OFFTRACK AND ONTRACK
trial = 0
latency = np.zeros(len(otracks_st[trial][:, 1]))
for i in range(len(otracks_st[trial][:, 1])):
    closer_offtrack_idx = np.argmin(np.abs(otracks_st[trial][i, 1]-offtracks_st[trial][:, 1]))
    latency[i] = offtracks_st[trial][closer_offtrack_idx, 1]-otracks_st[trial][i, 1]
plt.scatter(otracks_st[trial][:, 0], latency)
plt.xlabel('Online tracking frames')
plt.ylabel('Latency between online and offline tracking (s)')

#READ MP4 AND OVERLAY ONLINE DLC TRACKS
mp4_file = 'MC16946_60_25_0.1_0.1_tied_1_1.mp4'
otracks_file = 'MC16946_60_25_0.1_0.1_tied_1_1_otrack.csv'
vidObj = cv2.VideoCapture(os.path.join(path, mp4_file))
paw = 'FR'
frameNr = 100
if paw == 'FR':
    paw_color = (255, 0, 0)
if paw == 'FL':
    paw_color = (0, 0, 255)
vidObj.set(1, frameNr)
cap, frame = vidObj.read()
frame_otracks = cv2.circle(frame, (np.int64(otracks.iloc[frameNr, 2]), np.int64(otracks.iloc[frameNr, 3])),
                           radius=5, color=paw_color, thickness=0)
fig, ax = plt.subplots(tight_layout=True)
ax.imshow(frame_otracks, cmap='gray')

#READ MP4 AND OVERLAY OFFLINE DLC TRACKS
mp4_file = 'MC16946_60_25_0.1_0.1_tied_1_1.mp4'
otracks_file = 'MC16946_60_25_0.1_0.1_tied_1_1_otrack.csv'
otracks = pd.read_csv(os.path.join(path, otracks_file))
bg = cv2.imread('\\'.join(main_dir)+'\\Camera settings\\Camera frame.png')
vidObj = cv2.VideoCapture(os.path.join(path, mp4_file))
frame_otracks = otrack_class.overlayDLCtracks(vidObj, otracks, np.int64(otracks_st[0][0,0]), 'FR', frame_width, frame_height)
fig, ax = plt.subplots(tight_layout=True)
ax.imshow(frame_otracks, cmap='gray')




