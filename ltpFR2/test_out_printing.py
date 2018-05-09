import matplotlib.pyplot as plt
import numpy as np


# BOTTOM - MIDDLE ONES IN LTPFR

early =  [-0.56773132, -0.64126984, -0.70332946, -0.57291412]
late = [-0.4717487,  -0.58796519, -0.521875,   -0.5526381]

early_mean = sum(early)/4
late_mean = sum(late)/4
print(early_mean, late_mean)

means = [early_mean, late_mean]
sessions = ['Early Sessions (1-4)', 'Late Sessions (17-20)']
x = np.arange(len(sessions))
width = .005


fig = plt.figure()
ax = fig.add_subplot(111)

plt.bar(x, means, width = 0.5,  align = 'center', alpha = .5 )
plt.xticks(x, sessions, fontsize= 18)

plt.ylabel(' Change in Recall Probability', fontsize= 22)
plt.title('Low Freq - Middle Freq', fontsize= 22)
plt.show()