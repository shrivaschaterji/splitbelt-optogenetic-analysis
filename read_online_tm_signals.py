# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os

paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Online Tracking Treadmill\\Test Bonsai OneBelt Two Belts 100322\\MC16947\\'
subdir = 'TwoBeltsCM\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16947CM'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()
# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[timestamps_session, frame_counter_session, frame_counter_0_session, timestamps_0_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[trial_signal_session, sync_signal_session] = otrack_class.get_synchronizer_data(plot_data)

# READ ONLINE DLC TRACKS
[otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(frame_counter_0_session, timestamps_0_session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data_bottom(paw_otrack, loco, animal, session)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws_bottom(loco, animal, session)

# LATENCY OF LIGHT IN RELATION TO OTRACK
[latency_light_st, latency_light_sw, st_led_trials, sw_led_trials] = otrack_class.get_led_information_trials(trials, timestamps_session, otracks_st, otracks_sw)

# READ MP4 AND OVERLAY OFFLINE AND ONLINE DLC TRACKS
for t in trials:
    otrack_class.overlay_tracks_video(t, paw_otrack, offtracks_st, offtracks_sw, otracks_st, otracks_sw)
