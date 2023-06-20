# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

paw_otrack = 'FR'
path = 'J:\\Data OPTO\\75percent\\'
th_st_all = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
th_sw_all = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
condition = path.split('\\')[-2]
main_dir = path.split('\\')[:-2]
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))

animals = ['MC18089', 'MC18090']

th_st_cross_on = []
th_st_cross_off = []
th_st_detect_on = []
th_st_detect_off = []
th_st_laser_on = []
th_st_laser_off = []
th_st_on = []
th_st_off = []
animal_st_id = []
trial_st_id = []
condition_st_id = []
th_sw_cross_on = []
th_sw_cross_off = []
th_sw_detect_on = []
th_sw_detect_off = []
th_sw_laser_on = []
th_sw_laser_off = []
th_sw_on = []
th_sw_off = []
animal_sw_id = []
trial_sw_id = []
condition_sw_id = []
for animal in animals:
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)
    # LOAD DATA FOR BENCHMARK ANALYSIS
    [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)
    # GET BENCHMARK DATA STANCE
    [condition_id_st_singleanimal, trial_id_st_singleanimal, animal_id_st_singleanimal, th_cross_on_st_singleanimal,
     th_cross_off_st_singleanimal, th_detect_on_st_singleanimal, th_detect_off_st_singleanimal, th_laser_on_st_singleanimal,
     th_laser_off_st_singleanimal, th_st_on_st_singleanimal, th_st_off_st_singleanimal] = otrack_class.get_benchmark_data_laser('stance',
        th_st_all, condition, animal, otracks, otracks_st, otracks_sw, laser_on, offtracks_st, offtracks_sw)
    th_st_cross_on.extend(th_cross_on_st_singleanimal)
    th_st_cross_off.extend(th_cross_off_st_singleanimal)
    th_st_detect_on.extend(th_detect_on_st_singleanimal)
    th_st_detect_off.extend(th_detect_off_st_singleanimal)
    th_st_laser_on.extend(th_laser_on_st_singleanimal)
    th_st_laser_off.extend(th_laser_off_st_singleanimal)
    th_st_on.extend(th_st_on_st_singleanimal)
    th_st_off.extend(th_st_off_st_singleanimal)
    animal_st_id.extend(animal_id_st_singleanimal)
    trial_st_id.extend(trial_id_st_singleanimal)
    condition_st_id.extend(condition_id_st_singleanimal)
    # GET BENCHMARK DATA SWING
    [condition_id_sw_singleanimal, trial_id_sw_singleanimal, animal_id_sw_singleanimal, th_cross_on_sw_singleanimal,
     th_cross_off_sw_singleanimal, th_detect_on_sw_singleanimal, th_detect_off_sw_singleanimal, th_laser_on_sw_singleanimal,
     th_laser_off_sw_singleanimal, th_st_on_sw_singleanimal, th_st_off_sw_singleanimal] = otrack_class.get_benchmark_data_led('swing',
        th_st_all, condition, animal, otracks, otracks_st, otracks_sw, st_led_on, sw_led_on, offtracks_st, offtracks_sw)
    th_sw_cross_on.extend(th_cross_on_sw_singleanimal)
    th_sw_cross_off.extend(th_cross_off_sw_singleanimal)
    th_sw_detect_on.extend(th_detect_on_sw_singleanimal)
    th_sw_detect_off.extend(th_detect_off_sw_singleanimal)
    th_sw_laser_on.extend(th_laser_on_sw_singleanimal)
    th_sw_laser_off.extend(th_laser_off_sw_singleanimal)
    th_sw_on.extend(th_st_on_sw_singleanimal)
    th_sw_off.extend(th_st_off_sw_singleanimal)
    animal_sw_id.extend(animal_id_sw_singleanimal)
    trial_sw_id.extend(trial_id_sw_singleanimal)
    condition_sw_id.extend(condition_id_sw_singleanimal)
benchmark_data_st = pd.DataFrame(
    {'condition': condition_st_id, 'animal': animal_st_id, 'trial': trial_st_id, 'th_cross_on': th_st_cross_on,
     'th_cross_off': th_st_cross_off, 'th_detect_on': th_st_detect_on, 'th_detect_on': th_st_detect_off,
     'th_laser_on': th_st_laser_on, 'th_laser_off': th_st_laser_off, 'th_st_on': th_st_on, 'th_st_off': th_st_off})
benchmark_data_st.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_data_st.csv'), sep=',', index=False)

benchmark_data_sw = pd.DataFrame(
    {'condition': condition_sw_id, 'animal': animal_sw_id, 'trial': trial_sw_id, 'th_cross_on': th_sw_cross_on,
     'th_cross_off': th_sw_cross_off, 'th_detect_on': th_sw_detect_on, 'th_detect_on': th_sw_detect_off,
     'th_laser_on': th_sw_laser_on, 'th_laser_off': th_sw_laser_off, 'th_sw_on': th_sw_on, 'th_sw_off': th_sw_off})
benchmark_data_sw.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_data_sw.csv'), sep=',', index=False)





#TODO where do laser presentation errors come from?
#TODO latency between online tracking and threshold crossing
#TODO latency between software detection and light delivery
#obs: different fps in otrack trial, synchronizer, LED on measurements
#TODO st duration vs laser duration; otrack on st duration vs laser duration