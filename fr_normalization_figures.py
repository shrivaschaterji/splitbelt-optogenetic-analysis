import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\\Users\\Ana\\Carey Lab Dropbox\\Ana Gonçalves\\Tati&Hugo&AnaG&Alice\\Tests setup\\HR tests\\25percent\\'
animal = 'MC18090'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)

trials = otrack_class.get_trials(animal)
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
 laser_on] = otrack_class.load_processed_files(animal)

#HIND RIGHT
trial = 4 #animal MC18090
time_beg = 24.2
time_end = 25.4
time_beg_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_beg)[0][0]
time_end_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_end)[0][0]
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
plt.plot(otracks.loc[otracks['trial'] == trial, 'time'][time_beg_idx:time_end_idx],
         otracks.loc[otracks['trial'] == trial, 'x'][time_beg_idx:time_end_idx], color='darkgrey', label='')
# ax.axhline(y=200, color='black', label='25%')
# ax.axhline(y=150, linestyle='dashed', color='black', label='50%')
# ax.axhline(y=100, linestyle='dotted', color='black', label='75%')
# ax.axhline(y=100, color='black', label='25%')
# ax.axhline(y=150, linestyle='dashed', color='black', label='50%')
ax.axhline(y=200, linestyle='dotted', color='black', label='75%')
# ax.legend(bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=16)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel('FR - Tail base', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\for figures\\otrack_hr_sw_75', dpi=128)

# #CENTER OF MASS
# trial = 4 #animal MC18090
# time_beg = 14.4
# time_end = 17.2
# time_beg_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_beg)[0][0]
# time_end_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_end)[0][0]
# fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
# plt.plot(otracks.loc[otracks['trial'] == trial, 'time'][time_beg_idx:time_end_idx],
#          otracks.loc[otracks['trial'] == trial, 'x'][time_beg_idx:time_end_idx], color='darkgrey', label='')
# # ax.axhline(y=95, color='black', label='25%')
# # ax.axhline(y=70, linestyle='dashed', color='black', label='50%')
# ax.axhline(y=45, linestyle='dotted', color='black', label='75%')
# # ax.axhline(y=45, color='black', label='25%')
# # ax.axhline(y=70, linestyle='dashed', color='black', label='50%')
# # ax.axhline(y=95, linestyle='dotted', color='black', label='75%')
# # ax.legend(bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=16)
# ax.set_xlabel('Time (s)', fontsize=20)
# ax.set_ylabel('FR - Tail base', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('J:\\Thesis\\for figures\\otrack_cm_st_75', dpi=128)

# #TAILBASE
# trial = 2 #animal MC18090
# time_beg = 35
# time_end = 40
# time_beg_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_beg)[0][0]
# time_end_idx = np.where(otracks.loc[otracks['trial'] == trial, 'time']>=time_end)[0][0]
# fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
# plt.plot(otracks.loc[otracks['trial'] == trial, 'time'][time_beg_idx:time_end_idx],
#          otracks.loc[otracks['trial'] == trial, 'x'][time_beg_idx:time_end_idx], color='darkgrey', label='')
# # ax.axhline(y=215, color='black', label='25%')
# # ax.axhline(y=190, linestyle='dashed', color='black', label='50%')
# # ax.axhline(y=165, linestyle='dotted', color='black', label='75%')
# ax.axhline(y=165, color='black', label='25%')
# ax.axhline(y=190, linestyle='dashed', color='black', label='50%')
# ax.axhline(y=215, linestyle='dotted', color='black', label='75%')
# ax.legend(bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=16)
# ax.set_xlabel('Time (s)', fontsize=20)
# ax.set_ylabel('FR - Tail base', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('J:\\Thesis\\for figures\\otrack_tb_sw_all', dpi=128)