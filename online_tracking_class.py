#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:23:29 2020

@author: anagigoncalves
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob

class otrack_class:
    def __init__(self, path):
        self.path = path
        self.delim = self.path[-1]
        # self.pixel_to_mm = ?????
        self.sr = 333  # sampling rate of behavior camera for treadmill

    @staticmethod
    def converttime(time):
        """Converts the number given by Bonsai for the timestamps to seconds.
        Seen in Bonsai google group
        Input:
        time (int)"""
        # offset = time & 0xFFF
        cycle1 = (time >> 12) & 0x1FFF
        cycle2 = (time >> 25) & 0x7F
        seconds = cycle2 + cycle1 / 8000.
        return seconds

    @staticmethod
    def get_port_data(sync, port):
        """Converts the synchronizer csv data into the square pulses that it generated.
        Input:
        sync: dataframe from .._synch.csv
        port: name of the channel (int)"""
        sync_signal = sync.iloc[np.where(sync.iloc[:, 1] == 32)[0], :]
        sync_dev = []
        for i in range(len(sync_signal.iloc[:, 3])):
            bin_value = bin(sync_signal.iloc[i, 3])[2:]
            if bin_value == '1':
                sync_dev.append('00000000000001')
            else:
                sync_dev.append(bin_value)
        sync_dev_array = np.array(sync_dev)
        y = np.zeros(len(sync_dev_array))
        for i in range(len(sync_dev_array)-1):
            y[i] = int(sync_dev_array[i][-1 - port], 2)
        sync_signal_full = np.zeros(len(sync_signal) + 1)
        sync_signal_full[np.nonzero(np.diff(sync_signal.iloc[:, 2]))[0] + 1] = y[:-1]
        sync_timestamps = (sync_signal.iloc[:, 2] - sync_signal.iloc[0, 2]) * 1000  # in ms
        sync_timestamps_full = np.zeros(len(sync_timestamps) + 1)
        sync_timestamps_full[1:] = sync_timestamps
        sync_timestamps_full[0] = -0.1
        return sync_timestamps_full, sync_signal_full

    def get_trials(self):
        metadata_files = glob.glob(os.path.join(self.path, '*_meta.csv'))
        trial_order = []
        for f in metadata_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial_order.append(int(filename_split[7]))
        trials = np.sort(np.array(trial_order))
        return trials

    def get_session_metadata(self, plot_data):
        """From the meta csv get the timestamps and frame counter.
        Input:
        plot_data: boolean"""
        timestamps_session = []
        frame_counter_session = []
        frame_counter_0 = []
        timestamps_0 = []
        metadata_files = glob.glob(os.path.join(self.path,'*_meta.csv'))
        trial_order = []
        filelist = []
        for f in metadata_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order)) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        for trial, f in enumerate(files_ordered):
            metadata = pd.read_csv(os.path.join(self.path, f))
            cam_timestamps = [0]
            for t in np.arange(1, len(metadata.iloc[:,9])):
                cam_timestamps.append(self.converttime(metadata.iloc[t,9]-metadata.iloc[0,9]))
            print('FPS for camera acquisition ' + str(np.round(1/np.nanmedian(np.diff(cam_timestamps)), 2)))
            frame_counter = metadata.iloc[:,3]-metadata.iloc[0,3]
            frame_counter_0.append(metadata.iloc[0,3])
            timestamps_session.append(list(cam_timestamps))
            timestamps_0.append(metadata.iloc[0,9])
            frame_counter_session.append(list(frame_counter))
            if plot_data:
                plt.figure()
                plt.plot(list(cam_timestamps), metadata.iloc[:,3]-metadata.iloc[0,3])
                plt.title('Camera metadata for trial '+str(trial+1))
                plt.xlabel('Camera timestamps (s)')
                plt.ylabel('Frame counter')
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'timestamps_session.npy'), timestamps_session)
        np.save(os.path.join(self.path, 'processed files', 'frame_counter_session.npy'), frame_counter_session)
        np.save(os.path.join(self.path, 'processed files', 'frame_counter_0.npy'), frame_counter_0)
        np.save(os.path.join(self.path, 'processed files', 'timestamps_0.npy'), timestamps_0)
        return timestamps_session, frame_counter_session, frame_counter_0, timestamps_0

    def get_synchronizer_data(self, plot_data):
        """From the sync csv get the pulses generated from synchronizer.
        Input:
        plot_data: boolean"""
        trial_signal_session = []
        sync_signal_session = []
        sync_files = glob.glob(os.path.join(self.path, '*_synch.csv'))
        trial_order = []
        filelist = []
        for f in sync_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        for t, f in enumerate(files_ordered):
            sync_csv = pd.read_csv(os.path.join(self.path, f))
            [sync_timestamps_p0, sync_signal_p0] = self.get_port_data(sync_csv, 0)
            [sync_timestamps_p1, sync_signal_p1] = self.get_port_data(sync_csv, 1)
            trial_signal_session.append(np.concatenate((sync_timestamps_p0, sync_signal_p0)))
            sync_signal_session.append(np.concatenate((sync_timestamps_p1, sync_signal_p1)))
            if plot_data:
                plt.figure()
                plt.plot(sync_timestamps_p1, sync_signal_p1)
                plt.plot(sync_timestamps_p0, sync_signal_p0, linewidth=2)
                plt.title('Sync data for trial '+str(t+1))
                plt.xlabel('Time (ms)')
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'trial_signal_session.npy'), trial_signal_session)
        np.save(os.path.join(self.path, 'processed files', 'sync_signal_session.npy'), sync_signal_session)
        return trial_signal_session, sync_signal_session

    def get_otrack_event_data(self, frame_counter_0_session, timestamps_0_session):
        """Get the online tracking data (timestamps, frame counter, paw positionx and y).
        Use the first timestamps from the whole video to generate the sliced timestamps
        of the online tracking
        Input:
        frame_counter_0_session: list
        timestamps_0_session: list"""
        sync_files = glob.glob(os.path.join(self.path,'*_otrack.csv'))
        trial_order = []
        filelist = []
        for f in sync_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        otracks_st_time = []
        otracks_sw_time = []
        otracks_st_frames = []
        otracks_sw_frames = []
        otracks_st_trials = []
        otracks_sw_trials = []
        otracks_st_posx = []
        otracks_sw_posx = []
        otracks_st_posy = []
        otracks_sw_posy = []
        for trial, f in enumerate(files_ordered):
            otracks = pd.read_csv(os.path.join(self.path, f))
            stance_frames = np.where(otracks.iloc[:, 4]==True)[0]
            swing_frames = np.where(otracks.iloc[:, 5]==True)[0]
            otracks_frame_counter = []
            for i in range(len(otracks.iloc[:,1])):
                otracks_frame_counter.append((otracks.iloc[i,1]-frame_counter_0_session[trial]))
            otracks_timestamps = [0]
            for j in np.arange(1, len(otracks.iloc[:,0])):
                otracks_timestamps.append(self.converttime(np.array(otracks.iloc[j,0]-timestamps_0_session[trial])))
            otracks_st_time.extend(np.array(otracks_timestamps)[stance_frames])
            otracks_sw_time.extend(np.array(otracks_timestamps)[swing_frames])
            otracks_st_frames.extend(np.array(otracks_frame_counter)[stance_frames])
            otracks_sw_frames.extend(np.array(otracks_frame_counter)[swing_frames])
            otracks_st_trials.extend(np.array(np.ones(len(otracks_frame_counter))[stance_frames]*(trial+1)))
            otracks_sw_trials.extend(np.array(np.ones(len(otracks_frame_counter))[swing_frames]*(trial+1)))
            otracks_st_posx.extend(np.array(otracks.iloc[stance_frames, 2]))
            otracks_sw_posx.extend(np.array(otracks.iloc[swing_frames, 2]))
            otracks_st_posy.extend(np.array(otracks.iloc[stance_frames, 3]))
            otracks_sw_posy.extend(np.array(otracks.iloc[swing_frames, 3]))
        otracks_st = pd.DataFrame({'time': otracks_st_time, 'frames': otracks_st_frames, 'trial': otracks_st_trials,
            'x': otracks_st_posx, 'y': otracks_st_posy})
        otracks_sw = pd.DataFrame({'time': otracks_sw_time, 'frames': otracks_sw_frames, 'trial': otracks_sw_trials,
            'x': otracks_sw_posx, 'y': otracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        otracks_st.to_csv(
            os.path.join(self.path, 'processed files', 'otracks_st.csv'), sep=',',
            index=False)
        otracks_sw.to_csv(
            os.path.join(self.path, 'processed files', 'otracks_sw.csv'), sep=',',
            index=False)
        return otracks_st, otracks_sw

    def get_offtrack_paws_bottom(self, loco, animal, session):
        """Use the locomotion class to get the paw excursions from
        the post-hoc tracking.
        Input:
        loco: locomotion class
        animal: (str)
        session: (int)"""
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        final_tracks_trials = []
        for f in files_ordered:
            final_tracks = loco.read_h5_bottom(f, 0.9, 0)
            final_tracks_trials.append(final_tracks)
        return final_tracks_trials

    def get_offtrack_paws_bottomright(self, loco, animal, session):
        """Use the locomotion class to get the paw excursions from
        the post-hoc tracking.
        Input:
        loco: locomotion class
        animal: (str)
        session: (int)"""
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        final_tracks_trials = []
        for f in files_ordered:
            final_tracks = loco.read_h5_bottomright(f, 0.9, 0)
            final_tracks_trials.append(final_tracks)
        return final_tracks_trials

    def get_offtrack_event_data(self, paw, loco, animal, session):
        """Use the locomotion class to get the stance and swing points from
        the post-hoc tracking.
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        offtracks_st_time = []
        offtracks_sw_time = []
        offtracks_st_off_time = []
        offtracks_sw_off_time = []
        offtracks_st_frames = []
        offtracks_sw_frames = []
        offtracks_st_off_frames = []
        offtracks_sw_off_frames = []
        offtracks_st_trials = []
        offtracks_sw_trials = []
        offtracks_st_posx = []
        offtracks_sw_posx = []
        offtracks_st_posy = []
        offtracks_sw_posy = []
        for f in files_ordered:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial = int(filename_split[7][:-3])
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 0)  # no exclusion of strides
            offtracks_st_time.extend(np.array(st_strides_mat[p][:, 0, 0] / 1000))
            offtracks_st_off_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_off_time.extend(np.append(np.array(st_strides_mat[p][1:, 0, 0] / 1000), 0))
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1]))
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_sw_off_frames.extend(np.append(np.array(st_strides_mat[p][1:, 0, -1]), 0))
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)
            offtracks_st_posx.extend(final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posx.extend(final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])
            offtracks_st_posy.extend(final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posy.extend(final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        offtracks_st.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_st.csv'), sep=',',
            index=False)
        offtracks_sw.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_sw.csv'), sep=',',
            index=False)
        return offtracks_st, offtracks_sw

    def get_offtrack_event_data_bottom(self, paw, loco, animal, session):
        """Use the locomotion class to get the stance and swing points from
        the post-hoc tracking.
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        offtracks_st_time = []
        offtracks_sw_time = []
        offtracks_st_off_time = []
        offtracks_sw_off_time = []
        offtracks_st_frames = []
        offtracks_sw_frames = []
        offtracks_st_off_frames = []
        offtracks_sw_off_frames = []
        offtracks_st_trials = []
        offtracks_sw_trials = []
        offtracks_st_posx = []
        offtracks_sw_posx = []
        offtracks_st_posy = []
        offtracks_sw_posy = []
        for f in files_ordered:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial = int(filename_split[7][:-3])
            final_tracks = loco.read_h5_bottom(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 0)  # no exclusion of strides
            offtracks_st_time.extend(np.array(st_strides_mat[p][:, 0, 0] / 1000))
            offtracks_st_off_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_off_time.extend(np.append(np.array(st_strides_mat[p][1:, 0, 0] / 1000), 0))
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1]))
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_sw_off_frames.extend(np.append(np.array(st_strides_mat[p][1:, 0, -1]), 0))
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)
            offtracks_st_posx.extend(final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posx.extend(final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])
            offtracks_st_posy.extend(final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posy.extend(final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        offtracks_st.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_st.csv'), sep=',',
            index=False)
        offtracks_sw.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_sw.csv'), sep=',',
            index=False)
        return offtracks_st, offtracks_sw

    def get_offtrack_event_data_bottomright(self, paw, loco, animal, session):
        """Use the locomotion class to get the stance and swing points from
        the post-hoc tracking.
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        offtracks_st_time = []
        offtracks_sw_time = []
        offtracks_st_off_time = []
        offtracks_sw_off_time = []
        offtracks_st_frames = []
        offtracks_sw_frames = []
        offtracks_st_off_frames = []
        offtracks_sw_off_frames = []
        offtracks_st_trials = []
        offtracks_sw_trials = []
        offtracks_st_posx = []
        offtracks_sw_posx = []
        offtracks_st_posy = []
        offtracks_sw_posy = []
        for f in files_ordered:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial = int(filename_split[7][:-3])
            final_tracks = loco.read_h5_bottomright(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 0)  # no exclusion of strides
            offtracks_st_time.extend(np.array(st_strides_mat[p][:, 0, 0] / 1000))
            offtracks_st_off_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_sw_off_time.extend(np.append(np.array(st_strides_mat[p][1:, 0, 0] / 1000), 0))
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1]))
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_sw_off_frames.extend(np.append(np.array(st_strides_mat[p][1:, 0, -1]), 0))
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)
            offtracks_st_posx.extend(final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posx.extend(final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])
            offtracks_st_posy.extend(final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posy.extend(final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        offtracks_st.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_st.csv'), sep=',',
            index=False)
        offtracks_sw.to_csv(
            os.path.join(self.path, 'processed files', 'offtracks_sw.csv'), sep=',',
            index=False)
        return offtracks_st, offtracks_sw

    @staticmethod
    def get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw):
        """Get the otrack data for correspondent offtrack data. Also outputs the number of misses
        of sw and st for each trial.
        Input:
            trials: list of trials in the session
            otracks_st: dataframe with the otrack stance data
            otracks_sw: dataframe with the otrack swing data
            offtracks_st: dataframe with the offtrack stance data
            offtracks_sw: dataframe with the offtrack swing data"""
        otrack_st_miss = np.zeros(len(trials))
        otrack_st_frames_hits = []
        otrack_st_time_hits = []
        offtrack_st_frames_hits = []
        offtrack_st_time_hits = []
        otrack_st_trial = []
        for count_t, trial in enumerate(trials):
            offtrack_st_times = offtracks_st.loc[offtracks_st['trial'] == trial, 'time']
            offtrack_st_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            otrack_st_hits = []
            for t in np.array(offtrack_st_times):
                offtrack_frame = np.int64(
                    offtrack_st_trial.loc[offtrack_st_times.index[np.where(t == offtrack_st_times)[0][0]], 'frames'])
                offtrack_timeoff = offtrack_st_trial.loc[offtrack_st_times.index[np.where(t == offtrack_st_times)[0][0]], 'time_off']
                time_diff = t - np.array(otracks_st.loc[otracks_st['trial'] == trial, 'time'])
                idx_correspondent_offtrack = np.where((time_diff < 0) & (time_diff > -(offtrack_timeoff-t)))[0]
                if len(idx_correspondent_offtrack) > 0:
                    otrack_st_hits.append(otracks_st.loc[otracks_st['trial'] == trial].iloc[
                                              idx_correspondent_offtrack, 1])  # to get the number of correspondences
                    otrack_st_frames_hits.extend(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_offtrack, 1])
                    otrack_st_time_hits.extend(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_offtrack, 0])
                    offtrack_st_frames_hits.extend(np.repeat(offtrack_frame, len(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_offtrack, 1])))
                    offtrack_st_time_hits.extend(
                        np.repeat(t, len(otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_offtrack, 0])))
                    otrack_st_trial.extend(np.repeat(trial, len(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_offtrack, 0])))
            otrack_st_miss[count_t] = (len(offtrack_st_times) - len(otrack_st_hits))/len(offtrack_st_times)
        tracks_hits_st = pd.DataFrame({'otrack_frames': otrack_st_frames_hits, 'otrack_times': otrack_st_time_hits,
                                       'offtrack_frames': offtrack_st_frames_hits, 'offtrack_times': offtrack_st_time_hits,
                                       'trial': otrack_st_trial})
        otrack_sw_miss = np.zeros(len(trials))
        otrack_sw_frames_hits = []
        otrack_sw_time_hits = []
        offtrack_sw_frames_hits = []
        offtrack_sw_time_hits = []
        otrack_sw_trial = []
        for count_t, trial in enumerate(trials):
            offtrack_sw_times = offtracks_sw.loc[offtracks_sw['trial'] == trial, 'time']
            offtrack_sw_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            otrack_sw_hits = []
            for t in np.array(offtrack_sw_times):
                offtrack_frame = np.int64(
                    offtrack_sw_trial.loc[offtrack_sw_times.index[np.where(t == offtrack_sw_times)[0][0]], 'frames'])
                time_diff = t - np.array(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'])
                offtrack_timeoff = offtrack_sw_trial.loc[
                    offtrack_sw_times.index[np.where(t == offtrack_sw_times)[0][0]], 'time_off']
                idx_correspondent_offtrack = np.where((time_diff < 0) & (time_diff > -(offtrack_timeoff-t)))[0]
                if len(idx_correspondent_offtrack) > 0:
                    otrack_sw_hits.append(otracks_sw.loc[otracks_sw['trial'] == trial].iloc[
                                              idx_correspondent_offtrack, 1])  # to get the number of correspondences
                    otrack_sw_frames_hits.extend(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_offtrack, 1])
                    otrack_sw_time_hits.extend(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_offtrack, 0])
                    offtrack_sw_frames_hits.extend(np.repeat(offtrack_frame, len(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_offtrack, 1])))
                    offtrack_sw_time_hits.extend(
                        np.repeat(t, len(otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_offtrack, 0])))
                    otrack_sw_trial.extend(np.repeat(trial, len(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_offtrack, 0])))
            otrack_sw_miss[count_t] = (len(offtrack_sw_times) - len(otrack_sw_hits))/len(offtrack_sw_times)
        tracks_hits_sw = pd.DataFrame({'otrack_frames': otrack_sw_frames_hits, 'otrack_times': otrack_sw_time_hits,
                                       'offtrack_frames': offtrack_sw_frames_hits, 'offtrack_times': offtrack_sw_time_hits,
                                       'trial': otrack_sw_trial})
        return tracks_hits_st, tracks_hits_sw, otrack_st_miss, otrack_sw_miss

    @staticmethod
    def frames_outside_st_sw(trials, offtracks_st, offtracks_sw, otracks_st, otracks_sw):
        """Function to detect how many frames the online tracking said were relevant and
        they were outside the target swing or stance
        Inputs:
        trials: list of trials
        otracks_st: dataframe with the otrack stance data
        otracks_sw: dataframe with the otrack swing data
        offtracks_st: dataframe with the offtrack stance data
        offtracks_sw: dataframe with the offtrack swing data"""
        detected_frames_bad_st = np.zeros(len(trials))
        for count_t, trial in enumerate(trials):
            offtracks_st_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            otracks_st_trial = otracks_st.loc[otracks_st['trial'] == trial]
            frames_st = []
            for i in range(np.shape(offtracks_st_trial)[0]):
                frames_st.extend(np.arange(offtracks_st_trial.iloc[i, 2], offtracks_st_trial.iloc[i, 3]))
            detected_frames_bad_st[count_t] = len(
                np.setdiff1d(np.array(otracks_st_trial['frames']), np.array(frames_st)))
        detected_frames_bad_sw = np.zeros(len(trials))
        for count_t, trial in enumerate(trials):
            offtracks_sw_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            otracks_sw_trial = otracks_sw.loc[otracks_sw['trial'] == trial]
            frames_sw = []
            for i in range(np.shape(offtracks_sw_trial)[0]):
                frames_sw.extend(np.arange(offtracks_sw_trial.iloc[i, 2], offtracks_sw_trial.iloc[i, 3]))
            detected_frames_bad_sw[count_t] = len(
                np.setdiff1d(np.array(otracks_sw_trial['frames']), np.array(frames_sw)))
        return detected_frames_bad_st, detected_frames_bad_sw

    def overlay_tracks_video(self, trial, paw_otrack, offtracks_st, offtracks_sw, otracks_st, otracks_sw):
        """Function to overlay the post-hoc tracking (large circle) and the online
         (small filled circle) tracking on the video
         Input:
         trial: int
         paw_otrack: (str) FR or  FL
         otracks_st: dataframe with the otrack stance data
         otracks_sw: dataframe with the otrack swing data
         offtracks_st: dataframe with the offtrack stance data
         offtracks_sw: dataframe with the offtrack swing data"""
        if not os.path.exists(self.path + 'videos with tracks'):
            os.mkdir(self.path + 'videos with tracks')
        mp4_files = glob.glob(self.path + '*.mp4')
        frame_width = 1088
        frame_height = 420
        for f in mp4_files:
            filename_split = f.split(self.delim)[-1]
            trial_nr = np.int64(filename_split.split('_')[-1][:-4])
            if trial_nr == trial:
                filename = f
        vidObj = cv2.VideoCapture(filename)
        frames_total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(os.path.join(self.path, 'videos with tracks', filename.split(self.delim)[-1][:-4] + 'tracks.mp4'), cv2.VideoWriter_fourcc(*'XVID'), self.sr,
                              (frame_width, frame_height), True)
        if paw_otrack == 'FR':
            paw_color_st = (0, 0, 255)
            paw_color_sw = (255, 0, 0)
        if paw_otrack == 'FL':
            paw_color_st = (255, 0, 0)
            paw_color_sw = (0, 0, 255)
        for frameNr in range(frames_total):
            vidObj.set(1, frameNr)
            if frameNr in np.int64(offtracks_st.loc[offtracks_st['trial'] == trial, 'frames']):
                cap1, frame1 = vidObj.read()
                if cap1:
                    st_x_off = np.array(
                        offtracks_st.loc[(offtracks_st['trial'] == trial) & (offtracks_st['frames'] == frameNr), 'x'])
                    st_y_off = np.array(
                        offtracks_st.loc[(offtracks_st['trial'] == trial) & (offtracks_st['frames'] == frameNr), 'y'])
                    sw_x_off = np.array(
                        offtracks_sw.loc[(offtracks_sw['trial'] == trial) & (offtracks_sw['frames'] == frameNr), 'x'])
                    sw_y_off = np.array(
                        offtracks_sw.loc[(offtracks_sw['trial'] == trial) & (offtracks_sw['frames'] == frameNr), 'y'])
                    if np.all([~np.isnan(st_x_off), ~np.isnan(st_y_off)]) and len(st_x_off) > 0 and len(st_y_off) > 0:
                        frame_offtracks_st = cv2.circle(frame1, (np.int64(st_x_off)[0], np.int64(st_y_off)[0]),
                                                        radius=11, color=paw_color_st, thickness=2)
                        out.write(frame_offtracks_st)
                    if np.all([~np.isnan(sw_x_off), ~np.isnan(sw_y_off)]) and len(sw_x_off) > 0 and len(sw_y_off) > 0:
                        frame_offtracks_sw = cv2.circle(frame1, (np.int64(sw_x_off)[0], np.int64(sw_y_off)[0]),
                                                        radius=11, color=paw_color_sw, thickness=2)
                        out.write(frame_offtracks_sw)
            if frameNr in np.int64(otracks_st.loc[otracks_st['trial'] == trial, 'frames']):
                cap2, frame2 = vidObj.read()
                if cap2:
                    st_x_on = np.array(
                        otracks_st.loc[(otracks_st['trial'] == trial) & (otracks_st['frames'] == frameNr), 'x'])
                    st_y_on = np.array(
                        otracks_st.loc[(otracks_st['trial'] == trial) & (otracks_st['frames'] == frameNr), 'y'])
                    if np.all([~np.isnan(st_x_on), ~np.isnan(st_y_on)]):
                        frame_otracks_st = cv2.circle(frame2, (np.int64(st_x_on)[0], np.int64(st_y_on)[0]), radius=5,
                                                      color=paw_color_st, thickness=5)
                        out.write(frame_otracks_st)
            if frameNr in np.int64(otracks_sw.loc[otracks_sw['trial'] == trial, 'frames']):
                cap3, frame3 = vidObj.read()
                if cap3:
                    sw_x_on = np.array(
                        otracks_sw.loc[(otracks_sw['trial'] == trial) & (otracks_sw['frames'] == frameNr), 'x'])
                    sw_y_on = np.array(
                        otracks_sw.loc[(otracks_sw['trial'] == trial) & (otracks_sw['frames'] == frameNr), 'y'])
                    if np.all([~np.isnan(sw_x_on), ~np.isnan(sw_y_on)]):
                        frame_otracks_sw = cv2.circle(frame3, (np.int64(sw_x_on)[0], np.int64(sw_y_on)[0]), radius=5,
                                                      color=paw_color_sw, thickness=5)
                        out.write(frame_otracks_sw)
            cap4, frame4 = vidObj.read()
            if cap4:
                out.write(frame4)
        vidObj.release()
        out.release()

    def measure_light_on_videos(self, trial, timestamps_session, otracks_st, otracks_sw):
        """Function to measure when the light in the video was ON (equivalent to optogenetic
        stimulation).
         Input:
         trial: int
         timestamps_session: list with the camera timestamps for each session
         otracks_st: dataframe with the otrack stance data
         otracks_sw: dataframe with the otrack swing data"""
        mp4_files = glob.glob(self.path + '*.mp4')
        for f in mp4_files:
            filename_split = f.split(self.delim)[-1]
            trial_nr = np.int64(filename_split.split('_')[-1][:-4])
            if trial_nr == trial:
                filename = f
        vidObj = cv2.VideoCapture(filename)
        frames_total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        st_led = []
        sw_led = []
        for frameNr in range(frames_total):
            vidObj.set(1, frameNr)
            cap, frame = vidObj.read()
            if cap:
                st_led.append(np.mean(frame[:60, 980:1050, :].flatten()))
                sw_led.append(np.mean(frame[:60, 1050:, :].flatten()))
        vidObj.release()
        st_led_on_all = np.where(np.diff(st_led) > 5)[0]
        st_led_on = np.array(otrack_class.remove_consecutive_numbers(st_led_on_all))
        sw_led_on_all = np.where(np.diff(sw_led) > 5)[0]
        sw_led_on = np.array(otrack_class.remove_consecutive_numbers(sw_led_on_all))
        st_led_on_time = np.array(timestamps_session[trial - 1])[st_led_on]
        sw_led_on_time = np.array(timestamps_session[trial - 1])[sw_led_on]
        st_led_off_all = np.where(-np.diff(st_led) > 5)[0]
        st_led_off = np.array(otrack_class.remove_consecutive_numbers(st_led_off_all))
        sw_led_off_all = np.where(-np.diff(sw_led) > 5)[0]
        sw_led_off = np.array(otrack_class.remove_consecutive_numbers(sw_led_off_all))
        if len(st_led_on) != len(st_led_off):
            st_led_off = np.append(st_led_off, frameNr-1) # because python starts at 0
        if len(sw_led_on) != len(sw_led_off):
            sw_led_off = np.append(sw_led_off, frameNr-1) # because python starts at 0
        st_led_frames = np.vstack((st_led_on, st_led_off))
        sw_led_frames = np.vstack((sw_led_on, sw_led_off))
        otrack_st_trial = otracks_st.loc[otracks_st['trial'] == trial]
        otrack_sw_trial = otracks_sw.loc[otracks_sw['trial'] == trial]
        latency_trial_st = np.zeros(len(otrack_st_trial['time']))
        latency_trial_st[:] = np.nan
        for count_t, t in enumerate(otrack_st_trial['time']):
            time_diff = st_led_on_time - t
            time_diff_larger_0 = time_diff[time_diff > 0]
            if len(time_diff_larger_0)>0:
                latency_trial_st[count_t] = np.min(time_diff_larger_0) * 1000
        latency_trial_sw = np.zeros(len(otrack_sw_trial['time']))
        latency_trial_sw[:] = np.nan
        for count_t, t in enumerate(otrack_sw_trial['time']):
            time_diff = sw_led_on_time - t
            time_diff_larger_0 = time_diff[time_diff > 0]
            if len(time_diff_larger_0) > 0:
                latency_trial_sw[count_t] = np.min(time_diff_larger_0) * 1000
        return latency_trial_st, latency_trial_sw, st_led_frames, sw_led_frames

    def get_led_information_trials(self, trials, timestamps_session, otracks_st, otracks_sw):
        """Using the function to see, in wach trial when the LED were on and off loop over the session
        trials and compile this information across trials
        Inputs:
        trials: list of trials
        timestamps_session: list with the camera timestamps for each session
        otracks_st: dataframe with the otrack stance data
        otracks_sw: dataframe with the otrack swing data"""
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        latency_light_st = []
        latency_light_sw = []
        st_led_trials = []
        sw_led_trials = []
        for trial in trials:
            [latency_trial_st, latency_trial_sw, st_led_frames, sw_led_frames] = self.measure_light_on_videos(trial, timestamps_session, otracks_st, otracks_sw)
            latency_light_st.append(latency_trial_st)
            latency_light_sw.append(latency_trial_sw)
            st_led_trials.append(st_led_frames)
            sw_led_trials.append(sw_led_frames)
            np.save(os.path.join(self.path, 'processed files', 'latency_light_st_trial' + str(trial) + '.npy'), latency_trial_st)
            np.save(os.path.join(self.path, 'processed files', 'latency_light_sw_trial' + str(trial) + '.npy'), latency_trial_sw)
            np.save(os.path.join(self.path, 'processed files', 'st_led_trials_trial' + str(trial) + '.npy'), st_led_frames)
            np.save(os.path.join(self.path, 'processed files', 'sw_led_trials_trial' + str(trial) + '.npy'), sw_led_frames)
        return latency_light_st, latency_light_sw, st_led_trials, sw_led_trials

    @staticmethod
    def remove_consecutive_numbers(list_original):
        """Function that takes a list and removes any consecutive numbers, keeping the first one only
        Input:
        list_original (list)"""
        list_clean = []  # if there are consecutive numbers take the first
        last_seen = None
        for s in list_original:
            if (s - 1) != last_seen:
                list_clean.append(s)
            last_seen = s
        return list_clean

    def load_processed_files(self):
        """Function to load processed files (camera timestamps, online and offline tracking info, led light on info.
         Outputs:
         otracks_st, otracks_sw, offtracks_st, offtracks_sw, latency_light_st, latency_light_sw, st_led_frames,
         sw_led_frames, timestamps_session, final_tracks_trials"""
        otracks_st = pd.read_csv(
            os.path.join(self.path, 'processed files', 'otracks_st.csv'))
        otracks_sw = pd.read_csv(
            os.path.join(self.path, 'processed files', 'otracks_sw.csv'))
        offtracks_st = pd.read_csv(
            os.path.join(self.path, 'processed files', 'offtracks_st.csv'))
        offtracks_sw = pd.read_csv(
            os.path.join(self.path, 'processed files', 'offtracks_sw.csv'))
        timestamps_session = np.load(os.path.join(self.path, 'processed files', 'timestamps_session.npy'), allow_pickle=True)
        return otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session