import os
import numpy as np
import matplotlib.pyplot as plt

paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\'
subdir = '040423 mobile network crop bottom tests\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16946'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()

# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, st_led_on, sw_led_on] = otrack_class.load_processed_files()

# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_data)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

# PROCESS SYNCHRONIZER LASER SIGNALS
laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

# hits plot
[tracks_hits_st, tracks_hits_sw, otrack_st_hits, otrack_sw_hits] = otrack_class.get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
fig, ax = plt.subplots(tight_layout=True, sharey=True)
rectangle1 = plt.Rectangle((11.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
rectangle2 = plt.Rectangle((22.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle1)
plt.gca().add_patch(rectangle2)
for count_t, trial in enumerate(trials):
    ax.bar(trials, np.array(otrack_st_hits), color='green')
    ax.bar(trials, [np.shape(offtracks_st.loc[offtracks_st['trial']==i])[0] for i in trials]-np.array(otrack_st_hits), bottom = np.array(otrack_st_hits), color='red')
ax.set_xlabel('Trials', fontsize=16)
ax.set_ylabel('Stance points', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_misses_stance.png')
fig, ax = plt.subplots(tight_layout=True, sharey=True)
rectangle1 = plt.Rectangle((11.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
rectangle2 = plt.Rectangle((22.5, 0), 2, 450, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle1)
plt.gca().add_patch(rectangle2)
for count_t, trial in enumerate(trials):
    ax.bar(trials, np.array(otrack_sw_hits), color='green')
    ax.bar(trials,
              [np.shape(offtracks_sw.loc[offtracks_sw['trial'] == i])[0] for i in trials] - np.array(otrack_sw_hits),
              bottom = np.array(otrack_sw_hits), color='red')
ax.set_xlabel('Trials', fontsize=16)
ax.set_ylabel('Swing points', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\otrack_misses_swing.png')

# LASER LIGHT ON TRACKS
laser_event = 'stance'
laser_hits = np.zeros(len(trials))
laser_incomplete = np.zeros(len(trials))
laser_misses = np.zeros(len(trials))
for count_t, trial in enumerate(trials):
    [full_hits, incomplete_hits, misses] = otrack_class.get_hit_laser_synch(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
    laser_hits[count_t] = full_hits
    laser_incomplete[count_t] = incomplete_hits
    laser_misses[count_t] = misses
event='swing'
led_sw_hits = np.zeros(len(trials))
led_sw_incomplete = np.zeros(len(trials))
led_sw_misses = np.zeros(len(trials))
for count_t, trial in enumerate(trials):
    [full_hits, incomplete_hits, misses] = otrack_class.get_hit_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, plot_data)
    led_sw_hits[count_t] = full_hits
    led_sw_incomplete[count_t] = incomplete_hits
    led_sw_misses[count_t] = misses
# plot summaries
fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
rectangle = plt.Rectangle((10.5, 0), 2, 90, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
ax.plot(trials[:12], (laser_hits[:12]/(laser_hits[:12]+laser_misses[:12]+laser_incomplete[:12]))*100, '-o', color='green', linewidth=2)
ax.plot(trials[:12], (laser_incomplete[:12]/(laser_hits[:12]+laser_misses[:12]+laser_incomplete[:12]))*100, '-o', color='orange', linewidth=2)
ax.plot(trials[:12], (led_sw_hits[:12]/(led_sw_hits[:12]+led_sw_misses[:12]+led_sw_incomplete[:12]))*100, '-o', color='green', linestyle='dashed', linewidth=2)
ax.plot(trials[:12], (led_sw_incomplete[:12]/(led_sw_hits[:12]+led_sw_misses[:12]+led_sw_incomplete[:12]))*100, '-o', color='orange', linestyle='dashed', linewidth=2)
ax.legend(['stance full hit', 'stance onset hit', 'swing full hit', 'swing onset hit'], frameon=False, fontsize=14)
ax.set_xlabel('Trials', fontsize=16)
ax.set_ylabel('Percentage of \n correct hits', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\laser_accuracy_widethreshold.png')

# plot summaries
fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
rectangle = plt.Rectangle((22.5, 0), 2, 90, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
ax.plot(trials[12:], (laser_hits[12:]/(laser_hits[12:]+laser_misses[12:]+laser_incomplete[12:]))*100, '-o', color='green', linewidth=2)
ax.plot(trials[12:], (laser_incomplete[12:]/(laser_hits[12:]+laser_misses[12:]+laser_incomplete[12:]))*100, '-o', color='orange', linewidth=2)
ax.plot(trials[12:], (led_sw_hits[12:]/(led_sw_hits[12:]+led_sw_misses[12:]+led_sw_incomplete[12:]))*100, '-o', color='green', linestyle='dashed', linewidth=2)
ax.plot(trials[12:], (led_sw_incomplete[12:]/(led_sw_hits[12:]+led_sw_misses[12:]+led_sw_incomplete[12:]))*100, '-o', color='orange', linestyle='dashed', linewidth=2)
ax.set_xlabel('Trials', fontsize=16)
ax.set_ylabel('Percentage of \n correct hits', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\laser_accuracy_narrowthreshold.png')

# plot summaries
fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
data_st_narrow_all = []
data_st_wide_all = []
data_sw_narrow_all = []
data_sw_wide_all = []
for i in trials[:12]:
    data_st_wide = (laser_hits[i-1]/(laser_hits[i-1]+laser_misses[i-1]+laser_incomplete[i-1])*100) + (laser_incomplete[i-1]/(laser_hits[i-1]+laser_misses[i-1]+laser_incomplete[i-1])*100)
    data_sw_wide = (led_sw_hits[i-1]/(led_sw_hits[i-1]+led_sw_misses[i-1]+led_sw_incomplete[i-1])*100) + (led_sw_incomplete[i-1]/(led_sw_hits[i-1]+led_sw_misses[i-1]+led_sw_incomplete[i-1])*100)
    ax.scatter(0.5+np.random.rand(), data_st_wide, color='darkgrey')
    ax.scatter(2.5+np.random.rand(), data_sw_wide, color='darkgrey')
    data_st_wide_all.append(data_st_wide)
    data_sw_wide_all.append(data_sw_wide)
for i in trials[12:]:
    data_st_narrow = (laser_hits[i-1]/(laser_hits[i-1]+laser_misses[i-1]+laser_incomplete[i-1])*100) + (laser_incomplete[i-1]/(laser_hits[i-1]+laser_misses[i-1]+laser_incomplete[i-1])*100)
    data_sw_narrow = (led_sw_hits[i-1]/(led_sw_hits[i-1]+led_sw_misses[i-1]+led_sw_incomplete[i-1])*100) + (led_sw_incomplete[i-1]/(led_sw_hits[i-1]+led_sw_misses[i-1]+led_sw_incomplete[i-1])*100)
    ax.scatter(4.5+np.random.rand(), data_st_narrow, color='darkgrey')
    ax.scatter(6.5+np.random.rand(), data_sw_narrow, color='darkgrey')
    data_st_narrow_all.append(data_st_narrow)
    data_sw_narrow_all.append(data_sw_narrow)
ax.bar(1, np.nanmean(data_st_wide_all), color='black', zorder=0)
ax.bar(3, np.nanmean(data_sw_wide_all), color='black', zorder=0)
ax.bar(5, np.nanmean(data_st_narrow_all), color='black', zorder=0)
ax.bar(7, np.nanmean(data_sw_narrow_all), color='black', zorder=0)
ax.set_xticks([1, 3, 5, 7])
ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
ax.set_ylabel('Percentage of \n correct hits', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\laser_accuracy_total.png')


#STIMULATION DURATION
fig, ax = plt.subplots(tight_layout=True, figsize=(10,7))
data_st_narrow_all = []
data_st_wide_all = []
data_sw_narrow_all = []
data_sw_wide_all = []
for i in trials[:12]:
    data_st_wide = np.nanmean(laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000
    data_sw_wide = np.nanmean(sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000
    ax.scatter(0.5+np.random.rand(), data_st_wide, color='darkgrey')
    ax.scatter(2.5+np.random.rand(), data_sw_wide, color='darkgrey')
    data_st_wide_all.append(data_st_wide)
    data_sw_wide_all.append(data_sw_wide)
for i in trials[12:]:
    data_st_narrow = np.nanmean(laser_on.loc[laser_on['trial']==i]['time_off']-laser_on.loc[laser_on['trial']==i]['time_on'])*1000
    data_sw_narrow = np.nanmean(sw_led_on.loc[sw_led_on['trial']==i]['time_off']-sw_led_on.loc[sw_led_on['trial']==i]['time_on'])*1000
    ax.scatter(4.5+np.random.rand(), data_st_narrow, color='darkgrey')
    ax.scatter(6.5+np.random.rand(), data_sw_narrow, color='darkgrey')
    data_st_narrow_all.append(data_st_narrow)
    data_sw_narrow_all.append(data_sw_narrow)
ax.bar(1, np.nanmean(data_st_wide_all), color='black', zorder=0)
ax.bar(3, np.nanmean(data_sw_wide_all), color='black', zorder=0)
ax.bar(5, np.nanmean(data_st_narrow_all), color='black', zorder=0)
ax.bar(7, np.nanmean(data_sw_narrow_all), color='black', zorder=0)
ax.set_xticks([1, 3, 5, 7])
ax.set_xticklabels(['stance wide', 'swing wide', 'stance narrow', 'swing narrow'])
ax.set_ylabel('Laser duration (ms)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\laser_stim_duration.png')

