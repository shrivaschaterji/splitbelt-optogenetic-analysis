# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023

@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

paw_otrack = 'FR'
frame_width = 1088
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

# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[timestamps_session, frame_counter_session, frame_counter_0_session, timestamps_0_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[trial_signal_session, sync_signal_session] = otrack_class.get_synchronizer_data(plot_data)

# READ ONLINE DLC TRACKS
[otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(frame_counter_0_session, timestamps_0_session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session)

# MEASURE LATENCY BETWEEN OFFTRACK AND ONTRACK
trial = 1
plt.figure()
plt.scatter(offtracks_st.loc[offtracks_st['trial'] == trial, 'time'], np.repeat(1, len(offtracks_st.loc[offtracks_st['trial'] == trial, 'time'])), color='black')
otracks_time = otracks_st.loc[otracks_st['trial'] == trial, 'time']
for i in range(len(otracks_time)):
    plt.axvline(otracks_time[i], color='red')

trial = 1
latency = np.zeros(len(otracks_st.loc[otracks_st['trial'] == trial, 'time']))
for i in range(len(otracks_st.loc[otracks_st['trial'] == trial, 'time'])):
    closer_offtrack_idx = np.argmin(np.abs(otracks_st.loc[otracks_st['trial'] == trial, 'time'][i]-offtracks_st.loc[offtracks_st['trial'] == trial, 'time']))
    latency[i] = offtracks_st.loc[offtracks_st['trial'] == trial, 'time'][closer_offtrack_idx]-otracks_st.loc[otracks_st['trial'] == trial, 'time'][i]
plt.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'frames'], latency)
plt.xlabel('Online tracking frames')
plt.ylabel('Latency between online and offline tracking (s)')

# # MEASURE WHEN LIGHT WAS ON
# trial = 1
# mp4_file = 'MC16946_60_25_0.1_0.1_tied_1_1.mp4'
# vidObj = cv2.VideoCapture(os.path.join(path, mp4_file))
# frames_total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
# st_led = []
# sw_led = []
# for frameNr in range(frames_total):
#     vidObj.set(1, frameNr)
#     cap, frame = vidObj.read()
#     if cap:
#         st_led.append(np.mean(frame[:60, 980:1050, :].flatten()))
#         sw_led.append(np.mean(frame[:60, 1050:, :].flatten()))
# vidObj.release()
# st_led_on = np.where(np.diff(st_led) > 0)[0]
# sw_led_on = np.where(np.diff(st_led) > 0)[0]
# st_led_on_time = np.array(timestamps_session[trial-1])[st_led_on]
# sw_led_on_time = np.array(timestamps_session[trial-1])[sw_led_on]

# READ MP4 AND OVERLAY OFFLINE DLC TRACKS
trial = 1
mp4_file = 'MC16946_60_25_0.1_0.1_tied_1_1.mp4'
vidObj = cv2.VideoCapture(os.path.join(path, mp4_file))
frames_total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_height, frame_width), True)
paw = 'FR'
if paw == 'FR':
    paw_color = (255, 0, 0)
if paw == 'FL':
    paw_color = (0, 0, 255)
for frameNr in range(frames_total):
    vidObj.set(1, frameNr)
    if frameNr in np.int64(offtracks_st.loc[offtracks_st['trial']==trial, 'frames']):
        cap1, frame1 = vidObj.read()
        if cap1:
            st_x = np.array(offtracks_st.loc[(offtracks_st['trial'] == trial) & (offtracks_st['frames']==frameNr), 'x'])
            st_y = np.array(offtracks_st.loc[(offtracks_st['trial'] == trial) & (offtracks_st['frames']==frameNr), 'y'])
            if np.all([~np.isnan(st_x),~np.isnan(st_y)]):
                frame_offtracks = cv2.circle(frame1, (np.int64(st_x)[0], np.int64(st_y)[0]), radius=5, color=paw_color, thickness=0)
                out.write(frame_offtracks)
    if frameNr in np.int64(otracks_st.loc[otracks_st['trial']==trial, 'frames']):
        cap2, frame2 = vidObj.read()
        if cap2:
            st_x = np.array(otracks_st.loc[(otracks_st['trial'] == trial) & (otracks_st['frames']==frameNr), 'x'])
            st_y = np.array(otracks_st.loc[(otracks_st['trial'] == trial) & (otracks_st['frames']==frameNr), 'y'])
            if np.all([~np.isnan(st_x),~np.isnan(st_y)]):
                frame_otracks = cv2.circle(frame2, (np.int64(st_x)[0], np.int64(st_y)[0]), radius=5, color=paw_color, thickness=1)
                out.write(frame_otracks)
    cap3, frame3 = vidObj.read()
    if cap3:
        out.write(frame3)
vidObj.release()
out.release()


