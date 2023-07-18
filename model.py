import numpy as np
import math
import matplotlib.pyplot as plt

# right fast
ff = 6 
sf = 6
time = np.linspace(0,500,500)
fr = 1.3*np.sin(2*math.pi*ff*time*0.001-2)-0.3
fl = np.sin(2*math.pi*sf*time*0.001)
db_corr = np.exp(-(time-150)/1000)*(np.power(np.sin(2*math.pi*(time-150)/1000),20))/ 1.2848
ub_corr = -0.3*np.ones(time.shape)

cosp_times = []

plt.plot(time, db_corr)
plt.show()
plt.plot(time, fr)
plt.plot(time, fl)


plt.plot(time, fr+db_corr-ub_corr)
plt.plot(time, fl)
plt.show()


