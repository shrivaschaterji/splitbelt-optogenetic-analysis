import os
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\Tailbase tests\\50percent\\'
condition = path.split('\\')[-2]
network = path.split('\\')[-3]
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

animal = 'MC18089'
trial = 3
trials = otrack_class.get_trials(animal)

trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
 laser_on] = otrack_class.load_processed_files(animal)
# LOAD DATA FOR BENCHMARK ANALYSIS
[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
light_trial = laser_on.loc[laser_on['trial'] == trial]
led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
idx_th_cross = np.where(otracks.loc[otracks['trial']==trial, 'x']>=190)[0]
fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((led_trials[0, r], 100),
                              led_trials[1, r] - led_trials[0, r], 160, fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
ax.plot(otracks.loc[otracks['trial']==trial, 'time'], otracks.loc[otracks['trial']==trial, 'x'],
        color='black', linewidth=2)
ax.scatter(otracks.loc[otracks['trial']==trial, 'time'].iloc[idx_th_cross], otracks.loc[otracks['trial']==trial, 'x'].iloc[idx_th_cross],
           s=10, color='red')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
