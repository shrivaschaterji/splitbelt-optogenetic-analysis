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
from decord import VideoReader
from decord import cpu
np.warnings.filterwarnings('ignore')

class otrack_class:
    def __init__(self, path):
        self.path = path
        self.delim = self.path[-1]
        # self.pixel_to_mm = ????? #TODO measure this in the videos
        self.sr = 333  # sampling rate of behavior camera for treadmill

    @staticmethod
    def converttime(time):
        """Converts the number given by Bonsai for the timestamps to seconds.
        Seen in Bonsai google group: https://groups.google.com/g/bonsai-users/c/WD6mV94KAQs
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
        ChatGPT helped me convert Hugo's MATLAB code to Python
        Input:
        sync: dataframe from .._synch.csv
        port: name of the channel (int)"""
        data = sync.to_numpy()
        # select only trigger related events
        data_trigger = data[data[:, 1] == 32, :]
        # select data collected from sync_dev
        pe = np.array([np.binary_repr(np.int64(x), width=14) for x in data_trigger[:, 3]])
        # set time to the first timestamp
        x = (data_trigger[:, 2] - data_trigger[0, 2]) * 1000
        # select port events
        pe = np.array([np.binary_repr(np.int64(x), width=14) for x in data_trigger[:, 3]])
        y = np.array([int(x[-port - 1]) for x in pe])
        # in case of repeated timestamps use only the last one (when all the changing edges have been considered)
        data_trigger = data_trigger[[True] + list(np.diff(data_trigger[:, 2]) > 0), :]
        timestamps = data_trigger[:, 2]
        t0 = timestamps[0] * 1000
        timestamps_corr = (timestamps - timestamps[0]) * 1000
        # select port events
        port_events = np.array([np.binary_repr(np.int64(x), width=14) for x in data_trigger[:, 3]])
        data_port_events = np.array([np.int64(x[-port - 1]) for x in port_events])
        # this adds one data point before the first event which assumes that the port is at state 0
        timestamps_full = np.concatenate(([-0.1], timestamps_corr))
        sync_signal_full = np.concatenate(([0], data_port_events))
        peaks = np.concatenate(([0], np.diff(sync_signal_full)))
        if len(timestamps_full) > 0:
            rising_edges = timestamps_full[peaks == 1] - 0.000001
        else:
            rising_edges = []
        # since Python doesn't have a logical negation operator (~) like MATLAB, we need to convert the ttl array to a boolean array and then negate it using the astype(bool) method
        falling_edges = timestamps_full[peaks == -1] - 0.000001
        timestamps_trial = np.concatenate((timestamps_full, rising_edges, falling_edges))
        square_signal_trial = np.concatenate(
            (sync_signal_full, np.zeros(np.shape(rising_edges)), np.ones(np.shape(falling_edges))))
        ind = np.argsort(timestamps_trial, kind='mergesort')  # sort indices of timestamps mergesort is matlab style
        timestamps_done = timestamps_trial[ind]
        square_signal_done = square_signal_trial[ind]
        return timestamps_done, square_signal_done

    def get_trials(self):
        """Fom the metadata generated files compute the list of trials in the session"""
        metadata_files = glob.glob(os.path.join(self.path, '*_meta.csv'))
        trial_order = []
        for f in metadata_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial_order.append(int(filename_split[7]))
        trials = np.sort(np.array(trial_order))
        self.trials = trials
        trials_idx = np.arange(0, len(trials))
        self.trials_idx = trials_idx
        return trials

    def get_session_metadata(self, plot_data):
        """From the meta csv get the timestamps and frame counter.
        Input:
        plot_data: boolean"""
        frames_kept = []
        frame_counter_session = []
        timestamps_session = []
        metadata_files = glob.glob(os.path.join(self.path,'*_meta.csv'))
        trial_order = []
        filelist = []
        for f in metadata_files: #get the trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order)) #reorder trials
        files_ordered = []
        for f in range(len(filelist)): #order the metadata files
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        for trial, f in enumerate(files_ordered): #for each metadata file
            metadata = pd.read_csv(os.path.join(self.path, f), names=['a','b','c','d','e','f','g','h','i','j']) #ADD HEADER AND IT SHOUDL BE FINE
            cam_timestamps = [0]
            for t in np.arange(1, len(metadata.iloc[:,9])):
                cam_timestamps.append(self.converttime(metadata.iloc[t,9]-metadata.iloc[0,9])) #get the camera timestamps subtracting the first as the 0
            timestamps_session.append(cam_timestamps)
            frame_counter = np.array(metadata.iloc[:, 3] - metadata.iloc[
                0, 3])  # get the camera frame counter subtracting the first as the 0
            frame_counter_vec = np.arange(0, len(frame_counter)+1)
            frame_counter_diff = np.diff(frame_counter) #check for missing frames
            missing_frames_idx_start = frame_counter[np.where(np.diff(frame_counter) > 1)[0]]
            missing_frames = frame_counter_diff[np.diff(frame_counter) > 1] - 1
            missing_frames_idx = [] #do all the frames that are missing
            for count_i, i in enumerate(missing_frames_idx_start):
                for r in range(missing_frames[count_i]): #loop over the number of missing frames for that idx
                    i += 1
                    missing_frames_idx.append(i)
            frames_in = np.setdiff1d(np.arange(0, frame_counter[-1]+1), missing_frames_idx)
            frame_counter_session.append(frame_counter_vec)
            frames_kept.append(frames_in)
            if plot_data: #plot the camera timestamps and frame counter to see
                plt.figure()
                plt.plot(list(cam_timestamps), metadata.iloc[:,3]-metadata.iloc[0,3])
                plt.title('Camera metadata for trial '+str(self.trials[trial]))
                plt.xlabel('Camera timestamps (s)')
                plt.ylabel('Frame counter')
        return timestamps_session, frames_kept, frame_counter_session

    def get_synchronizer_data(self, frames_kept, plot_data):
        """From the sync csv get the pulses generated from synchronizer.
        Input:
        plot_data: boolean"""
        sync_files = glob.glob(os.path.join(self.path, '*_synch.csv'))
        trial_order = []
        filelist = []
        for f in sync_files: #get the trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)): #order the sync files
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        trial_p0_list_session = []
        trial_p1_list_session = []
        trial_p2_list_session = []
        trial_p3_list_session = []
        p0_signal_list = []
        p1_signal_list = []
        p2_signal_list = []
        p3_signal_list = []
        p0_time_list = []
        p1_time_list = []
        p2_time_list = []
        p3_time_list = []
        timestamps_session = []
        frame_counter_session = []
        for t, f in enumerate(files_ordered):
            sync_csv = pd.read_csv(os.path.join(self.path, f))
            [sync_timestamps_p0, sync_signal_p0] = self.get_port_data(sync_csv, 0) #read channel 0 of synchronizer - TRIAL START
            [sync_timestamps_p1, sync_signal_p1] = self.get_port_data(sync_csv, 1)  # read channel 1 of synchronizer - CAMERA TRIGGERS
            sync_signal_p0_on_idx = np.where(sync_signal_p0 > 0)[0][0]
            sync_signal_p0_off_idx = np.where(sync_signal_p0 > 0)[0][-1]
            time_beg = sync_timestamps_p0[sync_signal_p0_on_idx] #time when trial start signal started
            time_end = sync_timestamps_p0[sync_signal_p0_off_idx] #time when trial start signal ended
            timestamps_p1 = np.arange(time_beg, time_end, 3) #since cam is triggered all triggers should appear every 3ms between trial start ON
            [sync_timestamps_p1, sync_signal_p1] = self.get_port_data(sync_csv, 1) #read channel 1 of synchronizer - CAMERA TRIGGERS
            [sync_timestamps_p2, sync_signal_p2] = self.get_port_data(sync_csv, 2)  # read channel 2 of synchronizer - LASER SYNCH
            [sync_timestamps_p3, sync_signal_p3] = self.get_port_data(sync_csv, 3)  # read channel 3 of synchronizer - LASER TRIAL SYNCH
            trial_p0_list_session.extend(np.repeat(self.trials[t], len(sync_timestamps_p0)))
            trial_p1_list_session.extend(np.repeat(self.trials[t], len(sync_timestamps_p1)))
            trial_p2_list_session.extend(np.repeat(self.trials[t], len(sync_timestamps_p2)))
            trial_p3_list_session.extend(np.repeat(self.trials[t], len(sync_timestamps_p3)))
            p0_signal_list.extend(sync_signal_p0)
            p1_signal_list.extend(sync_signal_p1)
            p2_signal_list.extend(sync_signal_p2)
            p3_signal_list.extend(sync_signal_p3)
            p0_time_list.extend(sync_timestamps_p0/1000)
            p1_time_list.extend(sync_timestamps_p1/1000)
            p2_time_list.extend(sync_timestamps_p2/1000)
            p3_time_list.extend(sync_timestamps_p3/1000)
            if plot_data: # plot channel 0 and 1 from synchronizer to see
                plt.figure()
                plt.plot(sync_timestamps_p1/1000, sync_signal_p1)
                plt.plot(sync_timestamps_p0/1000, sync_signal_p0, linewidth=2)
                plt.title('Sync data for trial '+str(t+1))
                plt.xlabel('Time (ms)')
                plt.figure()
                plt.plot(sync_timestamps_p2/1000, sync_signal_p2)
                plt.title('Laser sync data for trial ' + str(self.trials[t]))
                plt.xlabel('Time (ms)')
                plt.figure()
                plt.plot(sync_timestamps_p3/1000, sync_signal_p3)
                plt.title('Laser trial sync data for trial ' + str(self.trials[t]))
                plt.xlabel('Time (ms)')
            camera_timestamps_in = timestamps_p1[frames_kept[self.trials_idx[t]]] / 1000
            timestamps_session.append(camera_timestamps_in)
            frame_counter_session.append(frames_kept[self.trials_idx[t]])
        if not os.path.exists(self.path + 'processed files'):  # save camera timestamps and frame counter in processed files
            os.mkdir(self.path + 'processed files')
        trial_signals = pd.DataFrame({'time': p0_time_list, 'trial': trial_p0_list_session, 'signal': p0_signal_list})
        cam_signals = pd.DataFrame({'time': p1_time_list, 'trial': trial_p1_list_session, 'signal': p1_signal_list})
        laser_signals = pd.DataFrame({'time': p2_time_list, 'trial': trial_p2_list_session, 'signal': p2_signal_list})
        laser_trial_signals = pd.DataFrame({'time': p3_time_list, 'trial': trial_p3_list_session, 'signal': p3_signal_list})
        trial_signals.to_csv(os.path.join(self.path, 'processed files', 'trial_signals.csv'), sep=',', index=False)
        cam_signals.to_csv(os.path.join(self.path, 'processed files', 'cam_signals.csv'), sep=',', index=False)
        laser_signals.to_csv(os.path.join(self.path, 'processed files', 'laser_signals.csv'), sep=',', index=False)
        laser_trial_signals.to_csv(os.path.join(self.path, 'processed files', 'laser_trial_signals.csv'), sep=',', index=False)
        np.save(os.path.join(self.path, 'processed files', 'timestamps_session.npy'), np.array(timestamps_session, dtype=object), allow_pickle=True)
        np.save(os.path.join(self.path, 'processed files', 'frame_counter_session.npy'), np.array(frame_counter_session, dtype=object), allow_pickle=True)
        return timestamps_session, frame_counter_session, trial_signals, cam_signals, laser_signals, laser_trial_signals

    def get_otrack_excursion_data(self, timestamps_session):
        """Get the online tracking data (timestamps, frame counter, paw position x and y).
        Use the first timestamps from the whole video to generate the sliced timestamps
        of the online tracking. Keep the all the tracked excursions of the paw
        Input:
        timestamps_session: list of timestamps (from synchronizer) for each trial"""
        otrack_files = glob.glob(os.path.join(self.path, '*_otrack.csv'))
        trial_order = []
        filelist = []
        for f in otrack_files:  # get the trial list sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):  # get otrack files in order
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        otracks_time = []
        otracks_frames = []
        otracks_trials = []
        otracks_posx = []
        otracks_st = []
        otracks_sw = []
        for trial, f in enumerate(files_ordered):
            otracks = pd.read_csv(os.path.join(self.path, f), names = ['bonsai time', 'bonsai frame', 'x', 'st', 'sw'])
            otracks_frame_counter = np.array(otracks.iloc[:, 1] - otracks.iloc[0, 1])# get the frame counter of otrack, first line is the same as first frame of the trial
            len_timestamps_trial = len(timestamps_session[self.trials_idx[trial]])
            if otracks_frame_counter[-1] >= len_timestamps_trial:
                frames_not_valid = np.where(otracks_frame_counter > len_timestamps_trial-1)[0]
                frames_valid = np.setdiff1d(np.arange(0, len(otracks_frame_counter)), frames_not_valid)
                otracks_frame_counter = otracks_frame_counter[frames_valid]
                otracks = otracks.iloc[frames_valid, :]
                print('otracks trial ' + str(trial + 1) + ' has more ' + str(len(frames_not_valid)) + ' frames than video')
            otracks_timestamps = np.array(timestamps_session[self.trials_idx[trial]])[otracks_frame_counter] #get timestamps of synchronizer for each otrack frame
            # create lists to add them to a dataframe
            otracks_time.extend(np.array(otracks_timestamps))  # list of timestamps
            otracks_frames.extend(np.array(otracks_frame_counter))  # list of frame counters
            otracks_trials.extend(
                np.array(np.ones(len(otracks_frame_counter)) * (self.trials[trial])))  # list of trial value
            otracks_posx.extend(
                np.array(otracks.iloc[:, 2]))  # list of otrack paw x position
            otracks_st.extend(
                np.array(otracks.iloc[:, 3]))  # list of otrack when in stance
            otracks_sw.extend(
                np.array(otracks.iloc[:, 4]))  # list of otrack when in swing
        # creating the dataframe
        otracks = pd.DataFrame({'time': otracks_time, 'frames': otracks_frames, 'trial': otracks_trials,
                                   'x': otracks_posx, 'st_on': otracks_st, 'sw_on': otracks_sw})
        if not os.path.exists(self.path + 'processed files'):  # saving the csv
            os.mkdir(self.path + 'processed files')
        otracks.to_csv(
            os.path.join(self.path, 'processed files', 'otracks.csv'), sep=',',
            index=False)
        return otracks

    def get_otrack_event_data(self, timestamps_session):
        """Get the online tracking data (timestamps, frame counter, paw position x and y).
        Use the first timestamps from the whole video to generate the sliced timestamps
        of the online tracking. Keep only the times where swing or stance was detected.
        Input:
        timestamps_session: list of timestamps (from synchronizer) for each trial"""
        otrack_files = glob.glob(os.path.join(self.path,'*_otrack.csv'))
        trial_order = []
        filelist = []
        for f in otrack_files: #get the trial list sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)): #get otrack files in order
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
            otracks = pd.read_csv(os.path.join(self.path, f), names = ['bonsai time', 'bonsai frame', 'x', 'y', 'st', 'sw'])
            otracks_frame_counter = np.array(otracks.iloc[:, 1] - otracks.iloc[0, 1])# get the frame counter of otrack, first line is the same as first frame of the trial
            len_timestamps_trial = len(timestamps_session[self.trials_idx[trial]])
            if otracks_frame_counter[-1] >= len_timestamps_trial:
                frames_not_valid = np.where(otracks_frame_counter > len_timestamps_trial-1)[0]
                frames_valid = np.setdiff1d(np.arange(0, len(otracks_frame_counter)), frames_not_valid)
                otracks_frame_counter = otracks_frame_counter[frames_valid]
                otracks = otracks.iloc[frames_valid, :]
                print('otracks trial ' + str(trial + 1) + ' has more ' + str(len(frames_not_valid)) + ' frames than video')
            stance_frames = np.where(otracks.iloc[:, 3]==True)[0] #get all the otrack where it detected a stance (above the threshold set in bonsai)
            swing_frames = np.where(otracks.iloc[:, 4]==True)[0] #get all the otrack where it detected a swing (above the threshold set in bonsai)
            otracks_timestamps = np.array(timestamps_session[self.trials_idx[trial]])[otracks_frame_counter] #get timestamps of synchronizer for each otrack frame
            # create lists to add them to a dataframe
            otracks_st_time.extend(np.array(otracks_timestamps)[stance_frames]) #list of timestamps
            otracks_sw_time.extend(np.array(otracks_timestamps)[swing_frames]) #list of timestamps
            otracks_st_frames.extend(np.array(otracks_frame_counter)[stance_frames]) #list of frame counters
            otracks_sw_frames.extend(np.array(otracks_frame_counter)[swing_frames]) #list of frame counters
            otracks_st_trials.extend(np.array(np.ones(len(otracks_frame_counter))[stance_frames]*(self.trials[trial]))) #list of trial value
            otracks_sw_trials.extend(np.array(np.ones(len(otracks_frame_counter))[swing_frames]*(self.trials[trial]))) #list of trial value
            otracks_st_posx.extend(np.array(otracks.iloc[stance_frames, 2])) #list of otrack paw x position when in stance
            otracks_sw_posx.extend(np.array(otracks.iloc[swing_frames, 2])) #list of otrack paw x position when in swing
            # otracks_st_posy.extend(np.array(otracks.iloc[stance_frames, 3])) #list of otrack paw y position when in stance
            # otracks_sw_posy.extend(np.array(otracks.iloc[swing_frames, 3])) #list of otrack paw y position when in swing
        #creating the dataframe
        otracks_st = pd.DataFrame({'time': otracks_st_time, 'frames': otracks_st_frames, 'trial': otracks_st_trials,
            'x': otracks_st_posx})    #, 'y': otracks_st_posy})
        otracks_sw = pd.DataFrame({'time': otracks_sw_time, 'frames': otracks_sw_frames, 'trial': otracks_sw_trials,
            'x': otracks_sw_posx})     #, 'y': otracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'): #saving the csv
            os.mkdir(self.path + 'processed files')
        otracks_st.to_csv(
            os.path.join(self.path, 'processed files', 'otracks_st.csv'), sep=',',
            index=False)
        otracks_sw.to_csv(
            os.path.join(self.path, 'processed files', 'otracks_sw.csv'), sep=',',
            index=False)
        return otracks_st, otracks_sw

    def get_offtrack_paws(self, loco, animal, session):
        """Use the locomotion class to get the paw excursions from
        the post-hoc tracking. full DLC NETWORK
        Input:
        loco: locomotion class
        animal: (str)
        session: (int)"""
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #get the DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #get the trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order h5 filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        final_tracks_trials = []
        for f in files_ordered:
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0) #read h5 using the full network features
            final_tracks_trials.append(final_tracks)
        return final_tracks_trials

    def get_offtrack_paws_bottom(self, loco, animal, session):
        """Use the locomotion class to get the paw excursions from
        the post-hoc tracking. BOTTOM VIEW DLC NETWORK
        Input:
        loco: locomotion class
        animal: (str)
        session: (int)"""
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #get the DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #get the trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order h5 filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        final_tracks_trials = []
        for f in files_ordered:
            final_tracks = loco.read_h5_bottom(f, 0.9, 0) #read h5 using the bottom view network features
            final_tracks_trials.append(final_tracks)
        return final_tracks_trials

    def get_offtrack_paws_bottomright(self, loco, animal, session):
        """Use the locomotion class to get the paw excursions from
        the post-hoc tracking. BOTTOM VIEW RIGHT BELT NETWORK
        Input:
        loco: locomotion class
        animal: (str)
        session: (int)"""
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #get the DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #get the trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order h5 filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        final_tracks_trials = []
        for f in files_ordered:
            final_tracks = loco.read_h5_bottomright(f, 0.9, 0) #read h5 using the bottom view right belt network features
            final_tracks_trials.append(final_tracks)
        return final_tracks_trials

    def get_offtrack_event_data(self, paw, loco, animal, session, timestamps_session):
        """Use the locomotion class to get the stance and swing points from
        the post-hoc tracking. FULL NETWORK (BOTH VIEWS)
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)
        timestamps_session: list of timestamps for each trial"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #get the DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #get trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)): #get files sorted by trial
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
        for count_t, f in enumerate(files_ordered):
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial = int(filename_split[7][:-3])
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0) #read h5 using the full network features
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)  # swing and stance detection, exclusion of strides
            #get lists for dataframe
            offtracks_st_time.extend(timestamps_session[count_t][np.int64(np.array(st_strides_mat[p][:, 0, -1]))]) #stance onset time in seconds
            offtracks_st_off_time.extend(timestamps_session[count_t][np.int64(np.array(sw_pts_mat[p][:, 0, -1]))]) #stance offset time in seconds, same as swing onset
            offtracks_sw_time.extend(timestamps_session[count_t][np.int64(np.array(sw_pts_mat[p][:, 0, -1]))]) #swing onset time in seconds
            offtracks_sw_off_time.extend(timestamps_session[count_t][np.int64(np.array(st_strides_mat[p][:, 1, -1]))]) #swing offset time in seconds, same as stride offset or the next stride stance onset
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1])) #stance onset idx
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1])) #stance offset idx
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1])) #swing onset idx
            offtracks_sw_off_frames.extend(np.array(st_strides_mat[p][:, 1, -1])) #swing offset idx
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial) #trial number
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial) #trial number
            offtracks_st_posx.extend(final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])]) #paw x position for stance onset
            offtracks_sw_posx.extend(final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])]) #paw x position for swing onset
            offtracks_st_posy.extend(final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])]) #paw y position for stance onset
            offtracks_sw_posy.extend(final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])]) #paw y position for swing onset
        #create dataframe
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'): #save csv
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
        the post-hoc tracking. BOTTOM VIEW NETWORK
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #get DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)): #get list of files in order
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
            final_tracks = loco.read_h5_bottom(f, 0.9, 0) #read h5 DLC output bottom view network
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)  # swing and stance detection, exclusion of strides
            # get lists for dataframe
            offtracks_st_time.extend(np.array(st_strides_mat[p][:, 0, 0] / 1000))  # stance onset time in seconds
            offtracks_st_off_time.extend(
                np.array(sw_pts_mat[p][:, 0, 0] / 1000))  # stance offset time in seconds, same as swing onset
            offtracks_sw_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))  # swing onset time in seconds
            offtracks_sw_off_time.extend(np.append(np.array(st_strides_mat[p][1:, 0, 0] / 1000),
                                                   0))  # swing offset time in seconds, same as stride offset or the next stride stance onset
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1]))  # stance onset idx
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))  # stance offset idx
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))  # swing onset idx
            offtracks_sw_off_frames.extend(np.append(np.array(st_strides_mat[p][1:, 0, -1]), 0))  # swing offset idx
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)  # trial number
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)  # trial number
            offtracks_st_posx.extend(
                final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])  # paw x position for stance onset
            offtracks_sw_posx.extend(
                final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])  # paw x position for swing onset
            offtracks_st_posy.extend(
                final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])  # paw y position for stance onset
            offtracks_sw_posy.extend(
                final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])  # paw y position for swing onset
        # create dataframe
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'): #save csv
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
        the post-hoc tracking. BOTTOM VIEW RIGHT BELT NETWORK
        Input:
        paw: 'FR' or 'FL'
        loco: locomotion class
        animal: (str)
        session: (int)"""
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5')) #read DLC offline tracking files
        filelist = []
        trial_order = []
        for f in h5files: #get trial order sorted
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)): #get h5 files sorted
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
            final_tracks = loco.read_h5_bottomright(f, 0.9, 0) #read h5 bottom view right belt network
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks,
                                                                   1)  # swing and stance detection, exclusion of strides
            # get lists for dataframe
            offtracks_st_time.extend(np.array(st_strides_mat[p][:, 0, 0] / 1000))  # stance onset time in seconds
            offtracks_st_off_time.extend(
                np.array(sw_pts_mat[p][:, 0, 0] / 1000))  # stance offset time in seconds, same as swing onset
            offtracks_sw_time.extend(np.array(sw_pts_mat[p][:, 0, 0] / 1000))  # swing onset time in seconds
            offtracks_sw_off_time.extend(np.append(np.array(st_strides_mat[p][1:, 0, 0] / 1000),
                                                   0))  # swing offset time in seconds, same as stride offset or the next stride stance onset
            offtracks_st_frames.extend(np.array(st_strides_mat[p][:, 0, -1]))  # stance onset idx
            offtracks_sw_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))  # stance offset idx
            offtracks_st_off_frames.extend(np.array(sw_pts_mat[p][:, 0, -1]))  # swing onset idx
            offtracks_sw_off_frames.extend(np.append(np.array(st_strides_mat[p][1:, 0, -1]), 0))  # swing offset idx
            offtracks_st_trials.extend(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)  # trial number
            offtracks_sw_trials.extend(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)  # trial number
            offtracks_st_posx.extend(
                final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])  # paw x position for stance onset
            offtracks_sw_posx.extend(
                final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])  # paw x position for swing onset
            offtracks_st_posy.extend(
                final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])  # paw y position for stance onset
            offtracks_sw_posy.extend(
                final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])  # paw y position for swing onset
        # create dataframe
        offtracks_st = pd.DataFrame(
            {'time': offtracks_st_time, 'time_off': offtracks_st_off_time, 'frames': offtracks_st_frames, 'frames_off': offtracks_st_off_frames,
             'trial': offtracks_st_trials,
             'x': offtracks_st_posx, 'y': offtracks_st_posy})
        offtracks_sw = pd.DataFrame(
            {'time': offtracks_sw_time, 'time_off': offtracks_sw_off_time, 'frames': offtracks_sw_frames, 'frames_off': offtracks_sw_off_frames,
             'trial': offtracks_sw_trials,
             'x': offtracks_sw_posx, 'y': offtracks_sw_posy})
        if not os.path.exists(self.path + 'processed files'): #save csv
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
        """Get the otrack data for correspondent offtrack data, by looking within that stride when otrack happened.
        Also outputs the number of misses of sw and st for each trial. The hits are the absolute number
        Input:
            trials: list of trials in the session
            otracks_st: dataframe with the otrack stance data
            otracks_sw: dataframe with the otrack swing data
            offtracks_st: dataframe with the offtrack stance data
            offtracks_sw: dataframe with the offtrack swing data"""
        otrack_st_hits = np.zeros(len(trials))
        otrack_st_frames_hits = []
        otrack_st_time_hits = []
        offtrack_st_frames_hits = []
        offtrack_st_time_hits = []
        otrack_st_trial = []
        for count_t, trial in enumerate(trials):
            offtrack_st_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            #get offtrack and otrack times
            offtrack_st_times = offtracks_st.loc[offtracks_st['trial'] == trial, 'time']
            otrack_st_time = np.array(otracks_st.loc[otracks_st['trial'] == trial, 'time'])
            otrack_st_hits_trial = 0
            for t in np.array(offtrack_st_times[:-1]):
                offtrack_frame = np.int64(
                    offtrack_st_trial.loc[offtrack_st_times.index[np.where(t == offtrack_st_times)[0][0]], 'frames'])
                offtrack_timeoff = offtrack_st_trial.loc[
                    offtrack_st_times.index[np.where(t == offtrack_st_times)[0][0]], 'time_off']
                #find otrack between the stance offtrack
                idx_correspondent_otrack = np.where((otrack_st_time > t) & (otrack_st_time < offtrack_timeoff))[0]
                #if found is a hit
                if len(idx_correspondent_otrack) > 0:
                    otrack_st_hits_trial += 1
                    otrack_st_frames_hits.extend(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_otrack, 1])
                    otrack_st_time_hits.extend(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_otrack, 0])
                    offtrack_st_frames_hits.extend(np.repeat(offtrack_frame, len(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_otrack, 1])))
                    offtrack_st_time_hits.extend(
                        np.repeat(t, len(otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_otrack, 0])))
                    otrack_st_trial.extend(np.repeat(trial, len(
                        otracks_st.loc[otracks_st['trial'] == trial].iloc[idx_correspondent_otrack, 0])))
            otrack_st_hits[count_t] = otrack_st_hits_trial
        tracks_hits_st = pd.DataFrame({'otrack_frames': otrack_st_frames_hits, 'otrack_times': otrack_st_time_hits,
                                       'offtrack_frames': offtrack_st_frames_hits, 'offtrack_times': offtrack_st_time_hits,
                                       'trial': otrack_st_trial})
        otrack_sw_hits = np.zeros(len(trials))
        otrack_sw_frames_hits = []
        otrack_sw_time_hits = []
        offtrack_sw_frames_hits = []
        offtrack_sw_time_hits = []
        otrack_sw_trial = []
        for count_t, trial in enumerate(trials):
            offtrack_sw_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            #get offtrack and otrack times
            offtrack_sw_times = offtracks_sw.loc[offtracks_sw['trial'] == trial, 'time']
            otrack_sw_time = np.array(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'])
            otrack_sw_hits_trial = 0
            for t in np.array(offtrack_sw_times[:-1]):
                offtrack_frame = np.int64(
                    offtrack_sw_trial.loc[offtrack_sw_times.index[np.where(t == offtrack_sw_times)[0][0]], 'frames'])
                offtrack_timeoff = offtrack_sw_trial.loc[
                    offtrack_sw_times.index[np.where(t == offtrack_sw_times)[0][0]], 'time_off']
                #find otrack between the swing offtrack
                idx_correspondent_otrack = np.where((otrack_sw_time > t) & (otrack_sw_time < offtrack_timeoff))[0]
                #if found is a hit
                if len(idx_correspondent_otrack) > 0:
                    otrack_sw_hits_trial += 1
                    otrack_sw_frames_hits.extend(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_otrack, 1])
                    otrack_sw_time_hits.extend(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_otrack, 0])
                    offtrack_sw_frames_hits.extend(np.repeat(offtrack_frame, len(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_otrack, 1])))
                    offtrack_sw_time_hits.extend(
                        np.repeat(t, len(otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_otrack, 0])))
                    otrack_sw_trial.extend(np.repeat(trial, len(
                        otracks_sw.loc[otracks_sw['trial'] == trial].iloc[idx_correspondent_otrack, 0])))
            otrack_sw_hits[count_t] =otrack_sw_hits_trial
        tracks_hits_sw = pd.DataFrame({'otrack_frames': otrack_sw_frames_hits, 'otrack_times': otrack_sw_time_hits,
                                       'offtrack_frames': offtrack_sw_frames_hits, 'offtrack_times': offtrack_sw_time_hits,
                                       'trial': otrack_sw_trial})
        return tracks_hits_st, tracks_hits_sw, otrack_st_hits, otrack_sw_hits

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
            #gets all the frames between offtrack stance onset and offset
            frames_st = []
            for i in range(np.shape(offtracks_st_trial)[0]):
                frames_st.extend(np.arange(offtracks_st_trial.iloc[i, 2], offtracks_st_trial.iloc[i, 3]))
            #detects the frames of otrack that fall outside of it
            detected_frames_bad_st[count_t] = len(
                np.setdiff1d(np.array(otracks_st_trial['frames']), np.array(frames_st)))
        #same for swing
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

    @staticmethod
    def inpaint_nans(A):
        """Interpolates NaNs in numpy arrays
        Input: A (numpy array)"""
        ok = ~np.isnan(A)
        xp = ok.ravel().nonzero()[0]
        fp = A[~np.isnan(A)]
        x  = np.isnan(A).ravel().nonzero()[0]
        A[np.isnan(A)] = np.interp(x, xp, fp)
        return A

    def overlay_tracks_video(self, trial, align, final_tracks_trials, laser_on, st_led_on, sw_led_on):
        """Function to overlay when the light came (laser or LED) on the video
        and on top of the FR paw
         Input:
         trial: int
         align: (str) laser, swing or stance - which signal to plot
         final_tracks_trials: paw excursions for each trial in the session
         laser_on: dataframe with the times where laser was on
         st_led_on: dataframe with the times where LED stance was on
         sw_led_on: dataframe with the times where LED swing was on"""
        if not os.path.exists(self.path + 'videos with tracks'):
            os.mkdir(self.path + 'videos with tracks')
        mp4_files = glob.glob(self.path + '*.mp4')  # gets all mp4 filenames
        frame_width = 1088
        frame_height = 420
        filename = []
        for f in mp4_files:
            filename_split = f.split(self.delim)[-1]
            trial_nr = np.int64(filename_split.split('_')[-1][:-4])
            if trial_nr == trial:
                filename = f  # reads the mp4 corresponding to the trial you want to do the video
        vidObj = VideoReader(filename, ctx=cpu(0))  # read the video
        frames_total = len(vidObj)
        # sets the specs of the output video
        out = cv2.VideoWriter(os.path.join(self.path, 'videos with tracks',
                                           filename.split(self.delim)[-1][:-4] + 'tracks.mp4'),
                              cv2.VideoWriter_fourcc(*'XVID'), self.sr,
                              (frame_width, frame_height), True)
        paw_color = (0, 0, 255)  # red
        if align == 'laser':
            trials = laser_on['trial'].unique()
            frames_on = np.array(laser_on.loc[laser_on['trial'] == trial, 'frames_on'])
            frames_off = np.array(laser_on.loc[laser_on['trial'] == trial, 'frames_off'])
        if align == 'stance':
            trials = st_led_on['trial'].unique()
            frames_on = np.array(st_led_on.loc[st_led_on['trial'] == trial, 'frames_on'])
            frames_off = np.array(st_led_on.loc[st_led_on['trial'] == trial, 'frames_off'])
        if align == 'swing':
            trials = sw_led_on['trial'].unique()
            frames_on = np.array(sw_led_on.loc[sw_led_on['trial'] == trial, 'frames_on'])
            frames_off = np.array(sw_led_on.loc[sw_led_on['trial'] == trial, 'frames_off'])
        frames_light_on = []
        for count_f, f in enumerate(frames_on):
            frames_light_on.extend(np.arange(frames_on[count_f], frames_off[count_f]))
        paw_x = self.inpaint_nans(final_tracks_trials[self.trials_idx[np.where(trials == trial)[0][0]]][2, 0, :])
        paw_z = self.inpaint_nans(final_tracks_trials[self.trials_idx[np.where(trials == trial)[0][0]]][3, 0, :])
        for frameNr in range(frames_total):
            frame = vidObj[frameNr]
            frame_np = frame.asnumpy()
            if frameNr in frames_light_on:
                x_position = np.int64(paw_x[frameNr])
                z_position = np.int64(paw_z[frameNr])
                frame_light = cv2.circle(frame_np, (x_position, z_position), radius=5, color=paw_color, thickness=5)
                out.write(frame_light)
            if frameNr not in frames_light_on:
                out.write(frame_np)
        out.release()

    def measure_light_on_videos(self, trial, timestamps_session, otracks_st, otracks_sw):
        """Function to measure when the light in the video was ON (equivalent to optogenetic
        stimulation).
         Input:
         trial: int
         timestamps_session: list with the camera timestamps for each session
         otracks_st: dataframe with the otrack stance data
         otracks_sw: dataframe with the otrack swing data"""
        mp4_files = glob.glob(self.path + '*.mp4') #read mp4 filenames
        filename = []
        for f in mp4_files: #get the mp4 for that specific trial
            filename_split = f.split(self.delim)[-1]
            trial_nr = np.int64(filename_split.split('_')[-1][:-4])
            if trial_nr == trial:
                filename = f
        vidObj = VideoReader(filename, ctx=cpu(0))  # read the video
        frames_total = len(vidObj)
        st_led = []
        sw_led = []
        for frameNr in range(frames_total):
            frame = vidObj[frameNr]
            frame_np = frame.asnumpy()
            # st_led.append(np.mean(frame_np[:60, 980:1050, :].flatten())) #get the mean intensity of the location where left LED is
            st_led.append(np.mean(
                frame_np[:60, 1050:, :].flatten()))  # get the mean intensity of the location where right LED is
            sw_led.append(np.mean(frame_np[:60, 1050:, :].flatten())) #get the mean intensity of the location where right LED is
        #if it starts on
        if st_led[0]>15:
            st_led[0] = 0
        if sw_led[0]>15:
            sw_led[0] = 0
        st_led_on_all = np.where(np.diff(st_led) > 5)[0]+1 #find when the left light turned on (idx)
        st_led_on = np.array(self.remove_consecutive_numbers(st_led_on_all)) #sometimes it takes a bit to turn on so get only the first value (weird but there's different intensities at times)
        sw_led_on_all = np.where(np.diff(sw_led) > 5)[0]+1 #find when the right light turned on (idx)
        sw_led_on = np.array(self.remove_consecutive_numbers(sw_led_on_all)) #sometimes it takes a bit to turn on so get only the first value
        st_led_on_time = np.array(timestamps_session[trial-1])[st_led_on] #find when the left light turned on
        sw_led_on_time = np.array(timestamps_session[trial-1])[sw_led_on] #find when the right light turned on
        st_led_off_all = np.where(-np.diff(st_led) > 5)[0]+1 #find when the left light turned off (idx)
        st_led_off = np.array(self.remove_consecutive_numbers(st_led_off_all)) #sometimes it takes a bit to turn off so get only the first value
        sw_led_off_all = np.where(-np.diff(sw_led) > 5)[0]+1 #find when the right light turned off (idx)
        sw_led_off = np.array(self.remove_consecutive_numbers(sw_led_off_all)) #sometimes it takes a bit to turn off so get only the first value
        if len(st_led_on) != len(st_led_off):
            if len(st_led_on) == len(st_led_off)+2: #rare case
                st_led_on = st_led_on[:-1]
            st_led_off = np.append(st_led_off, frameNr-1) #if trial ends with light on add the last frame, do -1 because python starts at 0
        if len(sw_led_on) != len(sw_led_off):
            if len(sw_led_on) == len(sw_led_off)+2: #rare case
                sw_led_on = sw_led_on[:-1]
            sw_led_off = np.append(sw_led_off, frameNr-1) #if trial ends with light on add the last frame, do -1 because python starts at 0
        st_led_frames = np.vstack((st_led_on, st_led_off)) #concatenate
        sw_led_frames = np.vstack((sw_led_on, sw_led_off))
        otrack_st_trial = otracks_st.loc[otracks_st['trial'] == trial]
        otrack_sw_trial = otracks_sw.loc[otracks_sw['trial'] == trial]
        latency_trial_st = np.zeros(len(otrack_st_trial['time']))
        latency_trial_st[:] = np.nan
        otrack_st_beg_times = np.array(otrack_st_trial['time'])[np.where(np.diff(otrack_st_trial['frames']) > 4)[0]+1]  # subsampling of 4th frame to do otrack
        for count_t, t in enumerate(otrack_st_beg_times):
            time_diff = st_led_on_time - t  # measure the time difference between otrack time and light turning on for stance
            latency_trial_st[count_t] = np.min(np.abs(time_diff)) * 1000
        latency_trial_sw = np.zeros(len(otrack_sw_trial['time']))
        latency_trial_sw[:] = np.nan
        otrack_sw_beg_times = np.array(otrack_sw_trial['time'])[np.where(np.diff(otrack_sw_trial['frames']) > 4)[0]+1]  # subsampling of 4th frame to do otrack
        for count_t, t in enumerate(otrack_sw_beg_times):
            time_diff = sw_led_on_time - t #measure the time difference between otrack time and light turning on for swing
            latency_trial_sw[count_t] = np.min(np.abs(time_diff)) * 1000
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
        st_led_on = []
        sw_led_on = []
        st_led_off = []
        sw_led_off = []
        st_led_time_on = []
        st_led_time_off = []
        st_led_trial = []
        sw_led_time_on = []
        sw_led_time_off = []
        sw_led_trial = []
        for count_t, trial in enumerate(trials):
            [latency_trial_st, latency_trial_sw, st_led_frames, sw_led_frames] = self.measure_light_on_videos(trial, timestamps_session, otracks_st, otracks_sw)
            latency_light_st.append(latency_trial_st)
            latency_light_sw.append(latency_trial_sw)
            st_led_on.extend(st_led_frames[0, :])
            sw_led_on.extend(sw_led_frames[0, :])
            st_led_off.extend(st_led_frames[1, :])
            sw_led_off.extend(sw_led_frames[1, :])
            st_led_time_on.extend(np.array(timestamps_session[self.trials_idx[count_t]])[st_led_frames[0, :]])
            st_led_time_off.extend(np.array(timestamps_session[self.trials_idx[count_t]])[st_led_frames[1, :]])
            sw_led_time_on.extend(np.array(timestamps_session[self.trials_idx[count_t]])[sw_led_frames[0, :]])
            sw_led_time_off.extend(np.array(timestamps_session[self.trials_idx[count_t]])[sw_led_frames[1, :]])
            st_led_trial.extend(np.repeat(trial, len(st_led_frames[0, :])))
            sw_led_trial.extend(np.repeat(trial, len(sw_led_frames[0, :])))
        st_led_on = pd.DataFrame({'time_on': st_led_time_on, 'time_off': st_led_time_off, 'frames_on': st_led_on, 'frames_off': st_led_off,
             'trial': st_led_trial})
        sw_led_on = pd.DataFrame({'time_on': sw_led_time_on, 'time_off': sw_led_time_off, 'frames_on': sw_led_on, 'frames_off': sw_led_off,
             'trial': sw_led_trial})
        if not os.path.exists(self.path + 'processed files'): #save csv
            os.mkdir(self.path + 'processed files')
        st_led_on.to_csv(os.path.join(self.path, 'processed files', 'st_led_on.csv'), sep=',', index=False)
        sw_led_on.to_csv(os.path.join(self.path, 'processed files', 'sw_led_on.csv'), sep=',', index=False)
        np.save(os.path.join(self.path, 'processed files', 'latency_light_st.npy'), np.array(latency_light_st, dtype=object), allow_pickle=True)
        np.save(os.path.join(self.path, 'processed files', 'latency_light_sw.npy'), np.array(latency_light_sw, dtype=object), allow_pickle=True)
        return latency_light_st, latency_light_sw, st_led_on, sw_led_on

    def get_laser_on(self, laser_signal_session, timestamps_session):
        """Get in a dataframe format for each trial the time the laser was on and off from the synchronizer.
        Save as csv
        Input:
            laser_signal_session: (csv)
            timestamps_session: list with the frame timestamps for each trial"""
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        laser_time_on = []
        laser_time_off = []
        laser_trial = []
        frame_time_on_all = []
        frame_time_off_all = []
        for count_t, trial in enumerate(self.trials):
            laser_time = np.array(laser_signal_session.loc[laser_signal_session['trial'] == trial, 'time'])
            laser_signal = np.array(laser_signal_session.loc[laser_signal_session['trial'] == trial, 'signal'])
            laser_signal_onset = laser_time[np.where(np.diff(laser_signal) > 0)[0]]
            laser_signal_offset = laser_time[np.where(np.diff(laser_signal) < 0)[0]]
            if len(laser_signal_offset)<len(laser_signal_onset):
                laser_signal_offset = np.append(laser_signal_offset, 0)
            laser_time_on.extend(laser_signal_onset)
            laser_time_off.extend(laser_signal_offset)
            laser_trial.extend(np.repeat(trial, len(laser_signal_offset)))
            frame_time_on = []
            for i in laser_signal_onset:
                frame_time_on.append(np.argmin(np.abs(i - timestamps_session[self.trials_idx[count_t]])))
            frame_time_on_all.extend(frame_time_on)
            frame_time_off = []
            for j in laser_signal_offset:
                frame_time_off.append(np.argmin(np.abs(j - timestamps_session[self.trials_idx[count_t]])))
            frame_time_off_all.extend(frame_time_off)
        laser_on = pd.DataFrame(
            {'time_on': laser_time_on, 'time_off': laser_time_off, 'frames_on': frame_time_on_all,
             'frames_off': frame_time_off_all, 'trial': laser_trial})
        if not os.path.exists(self.path + 'processed files'):  # save csv
            os.mkdir(self.path + 'processed files')
        laser_on.to_csv(os.path.join(self.path, 'processed files', 'laser_on.csv'), sep=',', index=False)
        return laser_on

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
        otracks = pd.read_csv(
            os.path.join(self.path, 'processed files', 'otracks.csv'))
        otracks_st = pd.read_csv(
            os.path.join(self.path, 'processed files', 'otracks_st.csv'))
        otracks_sw = pd.read_csv(
            os.path.join(self.path, 'processed files', 'otracks_sw.csv'))
        offtracks_st = pd.read_csv(
            os.path.join(self.path, 'processed files', 'offtracks_st.csv'))
        offtracks_sw = pd.read_csv(
            os.path.join(self.path, 'processed files', 'offtracks_sw.csv'))
        st_led_on = pd.read_csv(
            os.path.join(self.path, 'processed files', 'st_led_on.csv'))
        sw_led_on = pd.read_csv(
            os.path.join(self.path, 'processed files', 'sw_led_on.csv'))
        timestamps_session = np.load(os.path.join(self.path, 'processed files', 'timestamps_session.npy'), allow_pickle=True)
        return otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, st_led_on, sw_led_on

    def plot_led_on_paws(self, timestamps_session, st_led_on, sw_led_on, final_tracks_trials, otracks, st_th, sw_th, trial, event, online_bool):
        """Plot when LED was on paw excursions, either online or offline
        Inputs:
            timestamps_session: list with the timestamps for each trial
            st_led_on: csv with frames and times when st led was on
            sw_led_on: csv with frames and times when sw led was on
            final_tracks_trials: list with the final tracks (offline paw excursions)
            otracks: csv with online paw excursion
            st_th: (int) threshold for stance detection
            sw_th: (int) threshold for swing detection
            trial: (int) trial to plot
            event: (str) stance or swing
            online_bool: boolean for plotting online excursions (1) or offline (0) or both (2)
            """
        paw_colors = ['red', 'blue', 'magenta', 'cyan']
        p = 0
        if online_bool == 0 or online_bool == 2:
            scale_max = 800
            scale_min = -400
        else:
            scale_max = 250
            scale_min = 0
        st_led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
        otrack_trial_x = np.array(otracks.loc[otracks['trial'] == trial, 'x'])
        otrack_trial_time = np.array(otracks.loc[otracks['trial'] == trial, 'time'])
        otrack_threshold_st = self.remove_consecutive_numbers(np.where(otrack_trial_x > st_th)[0])
        otrack_threshold_sw = self.remove_consecutive_numbers(np.where(otrack_trial_x < sw_th)[0])
        if event == 'stance':
            fig, ax = plt.subplots()
            for r in range(np.shape(st_led_trials)[1]):
                rectangle = plt.Rectangle((timestamps_session[trial - 1][st_led_trials[0, r]], scale_min),
                                          timestamps_session[trial - 1][st_led_trials[1, r]] -
                                          timestamps_session[trial - 1][st_led_trials[0, r]], scale_max, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            if online_bool == 0:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
            if online_bool == 1:
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_st:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            if online_bool == 2:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_st:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            ax.set_title('light on stance')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        if event=='swing':
            sw_led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
            fig, ax = plt.subplots()
            for r in range(np.shape(sw_led_trials)[1]):
                rectangle = plt.Rectangle((timestamps_session[trial - 1][sw_led_trials[0, r]], scale_min),
                                          timestamps_session[trial - 1][sw_led_trials[1, r]] -
                                          timestamps_session[trial - 1][sw_led_trials[0, r]], scale_max, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            if online_bool == 0:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
            if online_bool == 1:
                ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
                        color='black')
                for t in otrack_threshold_sw:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            if online_bool == 2:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_sw:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            ax.set_title('light on swing')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return

    def plot_led_on_paws_frames(self, timestamps_session, st_led_on, sw_led_on, final_tracks_trials, otracks, st_th, sw_th, trial, event, online_bool):
        """Plot when LED was on paw excursions, either online or offline
        Inputs:
            timestamps_session: list with the timestamps for each trial
            st_led_on: csv with frames and times when st led was on
            sw_led_on: csv with frames and times when sw led was on
            final_tracks_trials: list with the final tracks (offline paw excursions)
            otracks: csv with online paw excursion
            st_th: (int) threshold for stance detection
            sw_th: (int) threshold for swing detection
            trial: (int) trial to plot
            event: (str) stance or swing
            online_bool: boolean for plotting online excursions (1) or offline (0) or both (2)
            """
        paw_colors = ['red', 'blue', 'magenta', 'cyan']
        p = 0
        if online_bool == 0 or online_bool == 2:
            scale_max = 800
            scale_min = -400
        else:
            scale_max = 250
            scale_min = 0
        st_led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
        otrack_trial_x = np.array(otracks.loc[otracks['trial'] == trial, 'x'] - 280)
        otrack_trial_time = np.array(otracks.loc[otracks['trial'] == trial, 'frames'])
        otrack_threshold_st = self.remove_consecutive_numbers(np.where(otrack_trial_x > st_th)[0])
        otrack_threshold_sw = self.remove_consecutive_numbers(np.where(otrack_trial_x < sw_th)[0])
        if event == 'stance':
            fig, ax = plt.subplots()
            for r in range(np.shape(st_led_trials)[1]):
                rectangle = plt.Rectangle((st_led_trials[0, r], scale_min),
                                          st_led_trials[1, r] -
                                          st_led_trials[0, r], scale_max, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            if online_bool == 0:
                ax.plot(final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
            if online_bool == 1:
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_st:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            if online_bool == 2:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_st:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            ax.set_title('light on stance')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        if event=='swing':
            sw_led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
            fig, ax = plt.subplots()
            for r in range(np.shape(sw_led_trials)[1]):
                rectangle = plt.Rectangle((sw_led_trials[0, r], scale_min),
                                          sw_led_trials[1, r] -
                                          sw_led_trials[0, r], scale_max, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            if online_bool == 0:
                ax.plot(final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
            if online_bool == 1:
                ax.plot(otracks.loc[otracks['trial'] == trial, 'frames'], otracks.loc[otracks['trial'] == trial, 'x'] - 280,
                        color='black')
                for t in otrack_threshold_sw:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            if online_bool == 2:
                ax.plot(timestamps_session[trial - 1],
                        final_tracks_trials[trial - 1][0, p, :] - np.nanmean(final_tracks_trials[trial - 1][0, p, :]),
                        color=paw_colors[p], linewidth=2)
                ax.plot(otrack_trial_time, otrack_trial_x, color='black')
                for t in otrack_threshold_sw:
                    ax.scatter(otrack_trial_time[t], otrack_trial_x[t], s=10, color='red')
            ax.set_title('light on swing')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return

    @staticmethod
    def plot_led_synchronizer_signals(trial, event, st_led_on, sw_led_on, laser_signal_session, timestamps_session, plot_data):
        """Plot times of LED on and LED signal gone to the synchronizer
        Inputs:
        trial: int
        event: (str) stance or swing
        st_led_on: csv with times of stance LED on
        sw_led_on: csv with times of stance LED on
        laser_signal_session: times of LED ON measured in the synchronizer
        timestamps_session: (list) with timestamps for each trial
        plot_data: boolean
        """
        laser_time = np.array(laser_signal_session.loc[laser_signal_session['trial'] == trial, 'time'])
        laser_signal = np.array(laser_signal_session.loc[laser_signal_session['trial'] == trial, 'signal'])
        laser_signal_onset = laser_time[np.where(np.diff(laser_signal) > 0)[0]]
        laser_signal_offset = laser_time[np.where(np.diff(laser_signal) < 0 )[0]]
        signal_time_diff_onset = []
        signal_time_diff_offset = []
        if event == 'stance':
            # stance
            st_led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
            event_signal_onset = timestamps_session[trial - 1][st_led_trials[0, :]]
            event_signal_offset = timestamps_session[trial - 1][st_led_trials[1, :]]
            plt.figure()
            for r in range(np.shape(st_led_trials)[1]):
                diff_onset = timestamps_session[trial - 1][st_led_trials[0, r]]-laser_signal_onset
                diff_offset = timestamps_session[trial - 1][st_led_trials[1, r]] - laser_signal_offset
                signal_time_diff_onset.append(diff_onset[np.argmin(np.abs(diff_onset))])
                signal_time_diff_offset.append(diff_offset[np.argmin(np.abs(diff_offset))])
                rectangle = plt.Rectangle((timestamps_session[trial - 1][st_led_trials[0, r]], 0),
                                          timestamps_session[trial - 1][st_led_trials[1, r]] -
                                          timestamps_session[trial - 1][st_led_trials[0, r]], 1, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            plt.plot(laser_time, laser_signal, color='black')
        if event == 'swing':
            # swing
            sw_led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
            event_signal_onset = timestamps_session[trial - 1][sw_led_trials[0, :]]
            event_signal_offset = timestamps_session[trial - 1][sw_led_trials[1, :]]
            plt.figure()
            for r in range(np.shape(sw_led_trials)[1]):
                diff_onset = timestamps_session[trial - 1][sw_led_trials[0, r]]-laser_signal_onset
                diff_offset = timestamps_session[trial - 1][sw_led_trials[1, r]] - laser_signal_offset
                signal_time_diff_onset.append(diff_onset[np.argmin(np.abs(diff_onset))])
                signal_time_diff_offset.append(diff_offset[np.argmin(np.abs(diff_offset))])
                rectangle = plt.Rectangle((timestamps_session[trial - 1][sw_led_trials[0, r]], 0),
                                          timestamps_session[trial - 1][sw_led_trials[1, r]] -
                                          timestamps_session[trial - 1][sw_led_trials[0, r]], 1, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            plt.plot(laser_time, laser_signal, color='black')
        signal_time_diff_onset_full = np.vstack((event_signal_onset, signal_time_diff_onset))
        signal_time_diff_offset_full = np.vstack((event_signal_offset, signal_time_diff_offset))
        return [signal_time_diff_onset_full, signal_time_diff_offset_full]

    @staticmethod
    def get_hit_laser_synch(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, plot_data):
        """Gets the hits of when laser sync was on. Gets also when
        the hit was incomplete (started before or ended after)
        Inputs:
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            laser_on: dataframe with laser synch on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial
            plot_data: boolean"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
            led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
            led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
        full_hits = 0
        incomplete_hits = 0
        full_hits_st = []
        incomplete_hits_st = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            after_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                     & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                     & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                     & (offtrack_trial['time_off'].iloc[t] < light_trial['time_off']))[0]
            if len(full_hit_idx) > 0:
                full_hits += 1
                full_hits_st.extend(full_hit_idx)
            if len(before_hit_idx) > 0:
                incomplete_hits += 1
                incomplete_hits_st.extend(before_hit_idx)
        misses = len(light_trial) - full_hits - incomplete_hits
        if plot_data:
            paw_colors = ['red', 'blue', 'magenta', 'cyan']
            p = 0
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in full_hits_st:
                rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], -400),
                                          timestamps_session[trial - 1][led_trials[1, r]] -
                                          timestamps_session[trial - 1][led_trials[0, r]], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, p, :])
            ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.scatter(offtrack_trial['time'], offtrack_trial['x'] - mean_excursion, s=20, color='black')
            ax.scatter(offtrack_trial['time_off'], offtrack_trial['x'] - mean_excursion, s=20, color='black')
            ax.set_title('full hit - light on ' + event + ' trial ' + str(trial))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in incomplete_hits_st:
                rectangle = plt.Rectangle((led_trials[0, r], -400),
                                          led_trials[1, r] -
                                          led_trials[0, r], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            ax.plot(final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.set_title('incomplete hit - light on ' + event + ' trial ' + str(trial))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return full_hits, incomplete_hits, misses

    @staticmethod
    def get_hit_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, plot_data):
        """Gets the hits of when light was either in stance or swing. Gets also when
        the hit was incomplete (started before or ended after)
        Inputs:
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            st_led_on: dataframe with stance led on
            sw_led_on: dataframe with swing led on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial
            plot_data: boolean"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = st_led_on.loc[st_led_on['trial'] == trial]
            led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = sw_led_on.loc[sw_led_on['trial'] == trial]
            led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
        full_hits = 0
        incomplete_hits = 0
        full_hits_st = []
        incomplete_hits_st = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            after_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                     & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                     & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                     & (offtrack_trial['time_off'].iloc[t] < light_trial['time_off']))[0]
            if len(full_hit_idx) > 0:
                full_hits += 1
                full_hits_st.extend(full_hit_idx)
            if len(before_hit_idx) > 0:
                incomplete_hits += 1
                incomplete_hits_st.extend(before_hit_idx)
        misses = len(light_trial) - full_hits - incomplete_hits
        if plot_data:
            paw_colors = ['red', 'blue', 'magenta', 'cyan']
            p = 0
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in full_hits_st:
                rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], -400),
                                          timestamps_session[trial - 1][led_trials[1, r]] -
                                          timestamps_session[trial - 1][led_trials[0, r]], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, p, :])
            ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.set_title('full hit - light on ' + event + ' trial ' + str(trial))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in incomplete_hits_st:
                rectangle = plt.Rectangle((led_trials[0, r], -400),
                                          led_trials[1, r] -
                                          led_trials[0, r], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            ax.plot(final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.set_title('incomplete hit - light on ' + event + ' trial ' + str(trial))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return full_hits, incomplete_hits, misses

    @staticmethod
    def accuracy_laser_sync(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, plot_data):
        """Gets the accuracy of laser presentations (true positive, false positive, true negative, false negative)
        Inputs:
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            laser_on: dataframe with laser synch on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial
            plot_data: boolean"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
            led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
            offtrack_trial_otherperiod = offtracks_sw.loc[offtracks_sw['trial'] == trial]
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
            led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
            offtrack_trial_otherperiod = offtracks_st.loc[offtracks_st['trial'] == trial]
        nr_presentations = np.shape(offtracks_st.loc[offtracks_st['trial'] == trial])[0] + np.shape(offtracks_sw.loc[offtracks_sw['trial'] == trial])[0]
        full_hits = 0
        incomplete_hits = 0
        full_hits_st = []
        incomplete_hits_st = []
        all = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            if len(full_hit_idx) > 0:
                full_hits += 1
                full_hits_st.extend(full_hit_idx)
                all.extend(full_hit_idx)
            if len(before_hit_idx) > 0:
                incomplete_hits += 1
                incomplete_hits_st.extend(before_hit_idx)
                all.extend(before_hit_idx)
        all_other = []
        for t in range(len(offtrack_trial_otherperiod['time'])):
            full_hit_idx = np.where((offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_off'])
                                    & (offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_off']))[0]
            incomplete_hit_idx = np.where((offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_off'])
                                          & (offtrack_trial_otherperiod['time'].iloc[t] > light_trial['time_on'])
                                          & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_on'])
                                          & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_off']))[
                0]
            if len(full_hit_idx) > 0:
                all_other.extend(full_hit_idx)
            if len(incomplete_hit_idx) > 0:
                all_other.extend(incomplete_hit_idx)
        # definitions of accuracy for an aimed stance
        # false positive is the number of times light hit correctly swing
        fp_trial = len(all_other) / nr_presentations
        # true positive is the number of times light hit correctly stance
        tp_trial = (full_hits + incomplete_hits) / nr_presentations
        # false negative is the number of times a stance was detected and there was no light there
        fn_trial = len(np.setdiff1d(np.arange(0, len(offtrack_trial['time'])), all)) / nr_presentations
        # true negative is the number of times a swing was detected and light was not there
        tn_trial = len(np.setdiff1d(np.arange(0, len(offtrack_trial_otherperiod['time'])), all_other))/ nr_presentations
        accuracy_trial = (full_hits + incomplete_hits + len(np.setdiff1d(np.arange(0, len(offtrack_trial_otherperiod['time'])), all_other)))/ nr_presentations
        precision_trial = (full_hits + incomplete_hits) / (full_hits + incomplete_hits + len(all_other))
        recall_trial = (full_hits + incomplete_hits) / (full_hits + incomplete_hits + len(np.setdiff1d(np.arange(0, len(offtrack_trial['time'])), all)))
        f1_trial = 2 * ((precision_trial*recall_trial)/(precision_trial+recall_trial))
        if plot_data:
            paw_colors = ['red', 'blue', 'magenta', 'cyan']
            p = 0
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in range(np.shape(led_trials)[1]):
                rectangle = plt.Rectangle((led_trials[0, r], -400),
                                          led_trials[1, r] - led_trials[0, r], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, p, :])
            ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial

    @staticmethod
    def accuracy_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, plot_data):
        """Gets the accuracy of LED presentations (true positive, false positive, true negative, false negative)
        Inputs:
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            st_led_on: dataframe with stance led on
            sw_led_on: dataframe with swing led on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial
            plot_data: boolean"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = st_led_on.loc[st_led_on['trial'] == trial]
            led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
            offtrack_trial_otherperiod = offtracks_sw.loc[offtracks_sw['trial'] == trial]
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = sw_led_on.loc[sw_led_on['trial'] == trial]
            led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
            offtrack_trial_otherperiod = offtracks_st.loc[offtracks_st['trial'] == trial]
        nr_presentations = np.shape(offtracks_st.loc[offtracks_st['trial'] == trial])[0] + np.shape(offtracks_sw.loc[offtracks_sw['trial'] == trial])[0]
        full_hits = 0
        incomplete_hits = 0
        full_hits_st = []
        incomplete_hits_st = []
        all = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            if len(full_hit_idx) > 0:
                full_hits += 1
                full_hits_st.extend(full_hit_idx)
                all.extend(full_hit_idx)
            if len(before_hit_idx) > 0:
                incomplete_hits += 1
                incomplete_hits_st.extend(before_hit_idx)
                all.extend(before_hit_idx)
        all_other = []
        for t in range(len(offtrack_trial_otherperiod['time'])):
            full_hit_idx = np.where((offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_off'])
                                    & (offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_off']))[0]
            incomplete_hit_idx = np.where((offtrack_trial_otherperiod['time'].iloc[t] < light_trial['time_off'])
                                          & (offtrack_trial_otherperiod['time'].iloc[t] > light_trial['time_on'])
                                          & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_on'])
                                          & (offtrack_trial_otherperiod['time_off'].iloc[t] > light_trial['time_off']))[
                0]
            if len(full_hit_idx) > 0:
                all_other.extend(full_hit_idx)
            if len(incomplete_hit_idx) > 0:
                all_other.extend(incomplete_hit_idx)
        # definitions of accuracy for an aimed stance
        # false positive is the number of times light hit correctly swing
        fp_trial = len(all_other) / nr_presentations
        # true positive is the number of times light hit correctly stance
        tp_trial = (full_hits + incomplete_hits) / nr_presentations
        # false negative is the number of times a stance was detected and there was no light there
        fn_trial = len(np.setdiff1d(np.arange(0, len(offtrack_trial['time'])), all)) / nr_presentations
        # true negative is the number of times a swing was detected and light was not there
        tn_trial = len(np.setdiff1d(np.arange(0, len(offtrack_trial_otherperiod['time'])), all_other))/ nr_presentations
        accuracy_trial = (full_hits + incomplete_hits + len(np.setdiff1d(np.arange(0, len(offtrack_trial_otherperiod['time'])), all_other)))/ nr_presentations
        precision_trial = (full_hits + incomplete_hits) / (full_hits + incomplete_hits + len(all_other))
        recall_trial = (full_hits + incomplete_hits) / (full_hits + incomplete_hits + len(np.setdiff1d(np.arange(0, len(offtrack_trial['time'])), all)))
        f1_trial = 2 * ((precision_trial*recall_trial)/(precision_trial+recall_trial))
        if plot_data:
            paw_colors = ['red', 'blue', 'magenta', 'cyan']
            p = 0
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            for r in range(np.shape(led_trials)[1]):
                rectangle = plt.Rectangle((led_trials[0, r], -400),
                                          led_trials[1, r] - led_trials[0, r], 800, fc='grey', alpha=0.3)
                plt.gca().add_patch(rectangle)
            mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, p, :])
            ax.plot(final_tracks_trials[trial - 1][0, p, :] - mean_excursion,
                    color=paw_colors[p], linewidth=2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial

    @staticmethod
    def laser_presentation_phase(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session):
        """
        Inputs: From all the times the laser was on it checks when the laser onset and offset happened in relation to either
        the stance or swing duration
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            st_led_on: dataframe with stance led on
            sw_led_on: dataframe with swing led on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = laser_on.loc[laser_on['trial'] == trial]
        light_onset_phase = []
        light_offset_phase = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            stride_duration = offtrack_trial['time_off'].iloc[t]-offtrack_trial['time'].iloc[t]
            light_onset_arr = np.array(light_trial['time_on'])
            light_offset_arr = np.array(light_trial['time_off'])
            if len(full_hit_idx) > 0:
                light_onset_phase.append((light_onset_arr[full_hit_idx[0]] - offtrack_trial['time'].iloc[t])/stride_duration)
                light_offset_phase.append((light_offset_arr[full_hit_idx[0]] - offtrack_trial['time'].iloc[t]) / stride_duration)
            if len(before_hit_idx) > 0:
                light_onset_phase.append((light_onset_arr[before_hit_idx[0]] - offtrack_trial['time'].iloc[t])/stride_duration)
                light_offset_phase.append((light_offset_arr[before_hit_idx[0]] - offtrack_trial['time'].iloc[t]) / stride_duration)
        return light_onset_phase, light_offset_phase

    @staticmethod
    def light_presentation_phase(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session):
        """
        Inputs: From all the times the LED was on it checks when the laser onset and offset happened in relation to either
        the stance or swing duration
            trial: int
            event: (str) stance or swing
            offtracks_st: dataframe with offline tracks for stance
            offtracks_sw: dataframe with offline tracks for swing
            st_led_on: dataframe with stance led on
            sw_led_on: dataframe with swing led on
            final_tracks_trials: paw excursions
            timestamps_session: list of timestamps for each trial"""
        if event == 'stance':
            offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
            light_trial = st_led_on.loc[st_led_on['trial'] == trial]
        if event == 'swing':
            offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
            light_trial = sw_led_on.loc[sw_led_on['trial'] == trial]
        light_onset_phase = []
        light_offset_phase = []
        for t in range(len(offtrack_trial['time'])):
            full_hit_idx = np.where((offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                    & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_on'])
                                    & (offtrack_trial['time'].iloc[t] < light_trial['time_off']))[0]
            before_hit_idx = np.where((offtrack_trial['time'].iloc[t] < light_trial['time_off'])
                                      & (offtrack_trial['time'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_on'])
                                      & (offtrack_trial['time_off'].iloc[t] > light_trial['time_off']))[0]
            stride_duration = offtrack_trial['time_off'].iloc[t]-offtrack_trial['time'].iloc[t]
            light_onset_arr = np.array(light_trial['time_on'])
            light_offset_arr = np.array(light_trial['time_off'])
            if len(full_hit_idx) > 0:
                light_onset_phase.append((light_onset_arr[full_hit_idx[0]] - offtrack_trial['time'].iloc[t])/stride_duration)
                light_offset_phase.append((light_offset_arr[full_hit_idx[0]] - offtrack_trial['time'].iloc[t]) / stride_duration)
            if len(before_hit_idx) > 0:
                light_onset_phase.append((light_onset_arr[before_hit_idx[0]] - offtrack_trial['time'].iloc[t])/stride_duration)
                light_offset_phase.append((light_offset_arr[before_hit_idx[0]] - offtrack_trial['time'].iloc[t]) / stride_duration)
        return light_onset_phase, light_offset_phase

    def accuracy_scores_otrack(self, otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw):
        """Function to compute accuracy, precision, recall and F1 scores for when the
        online tracking paw subtraction crosses the threshold.
        Inputs:
            otracks: dataframe with online tracking data
            otracks_st: dataframe with online tracking data when st was reached
            otracks_sw: dataframe with online tracking data when sw was reached
            offtracks_st: dataframe with offline tracking data when stance occurred
            offtracks_sw: dataframe with offline tracking data when swing occurred"""
        accuracy_st = np.zeros(len(self.trials))
        fn_st = np.zeros(len(self.trials))
        tn_st = np.zeros(len(self.trials))
        fp_st = np.zeros(len(self.trials))
        tp_st = np.zeros(len(self.trials))
        precision_st = np.zeros(len(self.trials))
        recall_st = np.zeros(len(self.trials))
        f1_st = np.zeros(len(self.trials))
        accuracy_sw = np.zeros(len(self.trials))
        fn_sw = np.zeros(len(self.trials))
        tn_sw = np.zeros(len(self.trials))
        fp_sw = np.zeros(len(self.trials))
        tp_sw = np.zeros(len(self.trials))
        precision_sw = np.zeros(len(self.trials))
        recall_sw = np.zeros(len(self.trials))
        f1_sw = np.zeros(len(self.trials))
        for count_t, t in enumerate(self.trials):
            # stance accuracy
            frames_otrack_st = np.array(otracks_st.loc[otracks_st['trial'] == t, 'frames'])
            otrack_st_off = otracks.loc[otracks['trial'] == t].iloc[
                np.where(otracks.loc[otracks['trial'] == t, 'st_on'] == 0)[0]]
            frames_otrack_st_off = np.array(otrack_st_off['frames'])
            otracks_trial_length = len(otracks.loc[otracks['trial'] == t])
            tp_st_trial = []
            fp_st_trial = []
            tn_st_trial = []
            fn_st_trial = []
            for count_i, i in enumerate(frames_otrack_st):
                tp_st_idx = np.where((i > np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames'])) & (
                            i < np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames_off'])))[
                    0]  # true positive
                fp_st_idx = np.where((i > np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames'])) & (
                            i < np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames_off'])))[
                    0]  # false positive
                if len(tp_st_idx) > 0:
                    tp_st_trial.append(otracks_st.iloc[count_i, 1])
                if len(fp_st_idx) > 0:
                    fp_st_trial.append(otracks_st.iloc[count_i, 1])
            for count_j, j in enumerate(frames_otrack_st_off):
                fn_st_idx = np.where((j > np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames'])) & (
                            j < np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames_off'])))[
                    0]  # false negative
                tn_st_idx = np.where((j > np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames'])) & (
                            j < np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames_off'])))[
                    0]  # true negative
                if len(tn_st_idx) > 0:
                    tn_st_trial.append(otrack_st_off.iloc[count_j, 1])
                if len(fn_st_idx) > 0:
                    fn_st_trial.append(otrack_st_off.iloc[count_j, 1])
            accuracy_st[count_t] = (len(tp_st_trial) + len(tn_st_trial)) / (
                        len(tp_st_trial) + len(fp_st_trial) + len(tn_st_trial) + len(fn_st_trial))
            fn_st[count_t] = len(fn_st_trial) / (len(tp_st_trial) + len(fp_st_trial) + len(tn_st_trial) + len(fn_st_trial))
            tn_st[count_t] = len(tn_st_trial) / (len(tp_st_trial) + len(fp_st_trial) + len(tn_st_trial) + len(fn_st_trial))
            fp_st[count_t] = len(fp_st_trial) / (
                        len(tp_st_trial) + len(fp_st_trial) + len(tn_st_trial) + len(fn_st_trial))
            tp_st[count_t] = len(tp_st_trial) / (
                        len(tp_st_trial) + len(fp_st_trial) + len(tn_st_trial) + len(fn_st_trial))
            precision_st[count_t] = len(tp_st_trial) / (len(tp_st_trial) + len(fp_st_trial))
            recall_st[count_t] = len(tp_st_trial) / (len(tp_st_trial) + len(fn_st_trial))
            f1_st[count_t] = 2 * ((precision_st[count_t] * recall_st[count_t]) / (precision_st[count_t] + recall_st[count_t]))
            # swing accuracy
            frames_otrack_sw = np.array(otracks_sw.loc[otracks_sw['trial'] == t, 'frames'])
            otrack_sw_off = otracks.loc[otracks['trial'] == t].iloc[
                np.where(otracks.loc[otracks['trial'] == t, 'sw_on'] == 0)[0]]
            frames_otrack_sw_off = np.array(otrack_sw_off['frames'])
            tp_sw_trial = []
            fp_sw_trial = []
            tn_sw_trial = []
            fn_sw_trial = []
            for count_i, i in enumerate(frames_otrack_sw):
                fp_sw_idx = np.where((i > np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames'])) & (
                            i < np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames_off'])))[0]
                tp_sw_idx = np.where((i > np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames'])) & (
                            i < np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames_off'])))[0]
                if len(tp_sw_idx) > 0:
                    tp_sw_trial.append(otracks_sw.iloc[count_i, 1])
                if len(fp_sw_idx) > 0:
                    fp_sw_trial.append(otracks_sw.iloc[count_i, 1])
            for count_j, j in enumerate(frames_otrack_sw_off):
                tn_sw_idx = np.where((j > np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames'])) & (
                            j < np.array(offtracks_st.loc[offtracks_st['trial'] == t, 'frames_off'])))[0]
                fn_sw_idx = np.where((j > np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames'])) & (
                            j < np.array(offtracks_sw.loc[offtracks_sw['trial'] == t, 'frames_off'])))[0]
                if len(tn_sw_idx) > 0:
                    tn_sw_trial.append(otrack_sw_off.iloc[count_j, 1])
                if len(fn_sw_idx) > 0:
                    fn_sw_trial.append(otrack_sw_off.iloc[count_j, 1])
            accuracy_sw[count_t] = (len(tp_sw_trial) + len(tn_sw_trial)) / (
                        len(tp_sw_trial) + len(fp_sw_trial) + len(tn_sw_trial) + len(fn_sw_trial))
            fn_sw[count_t] = len(fn_sw_trial) / (len(tp_sw_trial) + len(fp_sw_trial) + len(tn_sw_trial) + len(fn_sw_trial))
            tn_sw[count_t] = len(tn_sw_trial) / (len(tp_sw_trial) + len(fp_sw_trial) + len(tn_sw_trial) + len(fn_sw_trial))
            fp_sw[count_t] = len(fp_sw_trial) / (
                        len(tp_sw_trial) + len(fp_sw_trial) + len(tn_sw_trial) + len(fn_sw_trial))
            tp_sw[count_t] = len(tp_sw_trial) / (
                        len(tp_sw_trial) + len(fp_sw_trial) + len(tn_sw_trial) + len(fn_sw_trial))
            precision_sw[count_t] = len(tp_sw_trial) / (len(tp_sw_trial) + len(fp_sw_trial))
            recall_sw[count_t] = len(tp_sw_trial) / (len(tp_sw_trial) + len(fn_sw_trial))
            f1_sw[count_t] = 2 * ((precision_sw[count_t] * recall_sw[count_t]) / (precision_sw[count_t] + recall_sw[count_t]))
        return accuracy_st, accuracy_sw, precision_st, precision_sw, recall_st, recall_sw, f1_st, f1_st, fn_st, fn_sw, tn_st, tn_sw, fp_st, fp_sw, tp_st, tp_sw

    @staticmethod
    def setup_accuracy(trial, otracks, otracks_st, otracks_sw, th_st, th_sw, plot_data):
        """Function to compute setup-accuracy (how many times in the online tracking it
        crossed a threshold and it was detected as such).
        Inputs:
            otracks: dataframe with online tracking data
            otracks_st: dataframe with online tracking data when st was reached
            otracks_sw: dataframe with online tracking data when sw was reached
            th_st: threshold for stance (int)
            th_sw: threshold for swing (int)
            plot_data: boolean"""
        st_correct_trial = (len(np.where(otracks_st.loc[otracks_st['trial'] == trial, 'x'] >= th_st)[0]) / len(
            np.where(otracks.loc[otracks['trial'] == trial, 'x'] >= th_st)[0])) * 100
        sw_correct_trial = (len(np.where(otracks_sw.loc[otracks_sw['trial'] == trial, 'x'] <= th_sw)[0]) / len(
            np.where(otracks.loc[otracks['trial'] == trial, 'x'] <= th_sw)[0])) * 100
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
            ax.plot(otracks.loc[otracks['trial'] == trial, 'time'], otracks.loc[otracks['trial'] == trial, 'x'],
                    color='black')
            ax.scatter(otracks_st.loc[otracks_st['trial'] == trial, 'time'],
                       otracks_st.loc[otracks_st['trial'] == trial, 'x'], color='orange')
            ax.scatter(otracks_sw.loc[otracks_sw['trial'] == trial, 'time'],
                       otracks_sw.loc[otracks_sw['trial'] == trial, 'x'], color='green')
            ax.axhline(th_st, color='orange')
            ax.axhline(th_sw, color='green')
            ax.set_title('trial' + str(trial))
            ax.set_ylim([-100, 300])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return st_correct_trial, sw_correct_trial

    def cross_threshold_delays(self, otracks, otracks_st, otracks_sw, th_st, th_sw, plot_data):
        """Function to compute setup-accuracy (how many times in the online tracking it
        crossed a threshold and it was detected as such).
        Inputs:
            otracks: dataframe with online tracking data
            otracks_st: dataframe with online tracking data when st was reached
            otracks_sw: dataframe with online tracking data when sw was reached
            th_st: threshold for stance (list)
            th_sw: threshold for swing (list)
            plot_data: boolean"""
        st_threshold_cross_latency = []
        sw_threshold_cross_latency = []
        th_cross_st_duration = []
        th_cross_sw_duration = []
        # measure difference since x crossed the threshold and it detected true
        for count_t, trial in enumerate(self.trials):
            print(trial)
            otracks_trial = otracks.loc[otracks['trial'] == trial]
            otracks_st_trial = otracks_st.loc[otracks_st['trial'] == trial]
            otracks_sw_trial = otracks_sw.loc[otracks_sw['trial'] == trial]
            if otracks_trial.iloc[0, 0] == 0: #bug fix
                otracks_trial = otracks_trial.iloc[1:, :]
            if otracks_st_trial.iloc[0, 0] == 0: #bug fix
                otracks_st_trial = otracks_st_trial.iloc[1:, :]
            if otracks_sw_trial.iloc[0, 0] == 0: #bug fix
                otracks_sw_trial = otracks_sw_trial.iloc[1:, :]
            otracks_trial_time = np.array(otracks_trial['time'])
            # getting the time duration that x crossed the threshold for threshold stance and threshold swing
            th_cross_st_time = otracks_trial_time[np.where(np.array(otracks_trial['x']) >= th_st[count_t])[0]]
            th_cross_st_on_time = np.insert(th_cross_st_time[np.where(np.diff(th_cross_st_time) > 0.013)[0]], 0, th_cross_st_time[0])
            th_cross_st_off_time = th_cross_st_time[np.where(np.diff(th_cross_st_time) > 0.013)[0]-1]
            plt.plot(otracks_trial['time'], otracks_trial['x'], color='black')
            plt.axhline(y = 100, color='darkgray', zorder=0, linewidth=3)
            for i in range(len(th_cross_st_on_time)):
                plt.axvline(x = th_cross_st_on_time[i], color='orange')
                plt.axvline(x=th_cross_st_off_time[i], color='green')

            # if len(th_cross_st_on_time) > len(th_cross_st_off_time):
            #    th_cross_st_on_time = th_cross_st_on_time[:-1]
            # if len(th_cross_st_on_time) < len(th_cross_st_off_time):
            #     th_cross_st_off_time = th_cross_st_off_time[:-1]
            # th_cross_st_duration.append(th_cross_st_off_time-th_cross_st_on_time)
            th_cross_sw_time = otracks_trial_time[np.where(np.array(otracks_trial['x']) < th_sw[count_t])[0]]
            th_cross_sw_on_time = np.insert(th_cross_sw_time[np.where(np.diff(th_cross_sw_time) > 0.013)[0]], 0, th_cross_sw_time[0])
            th_cross_sw_off_time = th_cross_sw_time[np.where(np.diff(th_cross_sw_time) > 0.013)[0]-1]
            # if len(th_cross_sw_on_time) > len(th_cross_sw_off_time):
            #    th_cross_sw_on_time = th_cross_sw_on_time[:-1]
            # if len(th_cross_sw_on_time) < len(th_cross_sw_off_time):
            #     th_cross_sw_off_time = th_cross_sw_off_time[:-1]
            # th_cross_sw_duration.append(th_cross_sw_off_time-th_cross_sw_on_time)

            # get times where threshold was crossed and times where it got detected
            otracks_st_trial_time = np.array(otracks_st_trial['time'])
            st_th_cross_st_on_time_true = np.insert(otracks_st_trial_time[np.where(np.diff(otracks_st_trial_time) > 0.013)[0]], 0, otracks_st_trial_time[0])
            st_th_cross_st_off_time_true = otracks_st_trial_time[np.where(np.diff(otracks_st_trial_time) > 0.013)[0]-1]



            if len(st_th_cross_st_on_time_true) > len(th_cross_st_on_time):
                st_th_cross_st_on_time_true = st_th_cross_st_on_time_true[:-1]
            if len(st_th_cross_st_on_time_true) < len(th_cross_st_on_time):
                st_th_cross_st_on_time_true = th_cross_st_on_time[:-1]
            #st_threshold_cross_latency.append(st_th_cross_st_on_time_true-th_cross_st_on_time)
            otracks_sw_trial_time = np.array(otracks_sw_trial['time'])
            sw_th_cross_sw_on_time_true = np.insert(otracks_sw_trial_time[np.where(np.diff(otracks_sw_trial_time) > 0.013)[0]], 0, otracks_sw_trial_time[0])
            if len(sw_th_cross_sw_on_time_true) > len(th_cross_sw_on_time):
                sw_th_cross_sw_on_time_true = sw_th_cross_sw_on_time_true[:-1]
            if len(sw_th_cross_sw_on_time_true) < len(th_cross_sw_on_time):
                sw_th_cross_sw_on_time_true = th_cross_sw_on_time[:-1]
            #sw_threshold_cross_latency.append(sw_th_cross_sw_on_time_true-th_cross_sw_on_time)
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
            for count_t, trial in enumerate(self.trials):
                ax.scatter(np.repeat(trial, len(st_threshold_cross_latency[count_t])) + np.random.uniform(low=0, high=0.5,
                                                                                           size=len(st_threshold_cross_latency[count_t])),
                           st_threshold_cross_latency[count_t]*1000, color='orange')
            ax.set_xlabel('Trials', fontsize=14)
            ax.set_ylabel('Time from threshold cross\nand threshold detected (ms)', fontsize=14)
            ax.set_title('Stance threshold', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
            for count_t, trial in enumerate(self.trials):
                ax.scatter(np.repeat(trial, len(sw_threshold_cross_latency[count_t])) + np.random.uniform(low=0, high=0.5,
                                                                                           size=len(sw_threshold_cross_latency[count_t])),
                           sw_threshold_cross_latency[count_t]*1000, color='green')
            ax.set_xlabel('Trials', fontsize=14)
            ax.set_ylabel('Time from threshold cross\nand threshold detected (ms)', fontsize=14)
            ax.set_title('Swing threshold', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return st_threshold_cross_latency, sw_threshold_cross_latency
