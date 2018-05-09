

import matplotlib.pyplot as plt
import numpy as np


# BOTTOM - MIDDLE ONES IN LTPFR
#
# early =  [0.01660859, -0.0293254,  -0.07539683,  0.00385633]
# late = [0.02778111, -0.01033413, 0.03415179,  0.01919524]
#
# early_sem = [ 0.03408088,  0.03445868,  0.03750458,  0.03323388]
# late_sem = [0.03462189,  0.02854616, 0.03389487,  0.03253533]
#
# early_mean = sum(early)/4
# late_mean = sum(late)/4
#
# early_sem_mean = sum(early_sem)/4
# late_sem_mean = sum(late_sem)/4



# TOP - MIDDLE ONES IN LTPFR

# early =  [ 0.02133178,  0.00863345, -0.05256944,  0.01945462]
# late = [0.00078709, -0.00277441, 0.05728228, -0.02757619]
#
# early_sem = [  0.03323699,  0.03593055,  0.03531979,  0.03578077]
# late_sem = [0.03729969,  0.03463955, 0.03560656,  0.03269066]

# TOP + BOTTOM - MIDDLE

early = [ 0.02630435,  0.0039903,  -0.06961153, -0.00180736]
late = [0.01449323, -0.01589286, 0.02892157,  0.00112377]

early_sem = [ 0.02766832,  0.02905974,  0.030421,    0.02919361]
late_sem = [0.03032999,  0.0247751, 0.02842167,  0.02633657]

early_mean = sum(early)/4
late_mean = sum(late)/4

early_sem_mean = sum(early_sem)/4
late_sem_mean = sum(late_sem)/4

means = [early_mean, late_mean]
sems = [early_sem_mean, late_sem_mean]
sessions = ['Early Sessions (1-4)', 'Late Sessions (17-20)']
x = np.arange(len(sessions))
width = .005


fig = plt.figure()
ax = fig.add_subplot(111)

plt.bar(x, means, width = 0.5,  align = 'center', alpha = .5, yerr=sems )
plt.xticks(x, sessions, fontsize= 18)

plt.ylabel(' Change in Recall Probability', fontsize= 22)
plt.title('Top Freq + Bottom Freq - Middle Freq', fontsize= 22)
plt.show()