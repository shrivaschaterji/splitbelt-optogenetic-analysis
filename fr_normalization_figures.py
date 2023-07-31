import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\\Users\\Ana\\Carey Lab Dropbox\\Ana Gonçalves\\Tati&Hugo&AnaG&Alice\\Tests setup\\Tailbase tests\\25percent\\'
animal = 'MC18089'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)

trials = otrack_class.get_trials(animal)
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
 laser_on] = otrack_class.load_processed_files(animal)


plt.plot(otracks.loc[otracks['trial']==2, 'time'], otracks.loc[otracks['trial']==2, 'x'], color='black')