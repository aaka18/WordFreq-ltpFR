

import matplotlib.pyplot as plt
import numpy as np

# BAR CHARTS WITH

# TOP QUARTILE - MIDDLE ONES (AVERAGED) IN LTPFR

# early_top = [0.02133178, 0.00863345, - 0.05256944, 0.01945462]
# late_top = [0.00078709, - 0.00277441, 0.05728228, - 0.02757619]
#
# early_top_sem = [0.03323699,  0.03593055,  0.03531979,  0.03578077]
# late_top_sem = [0.03729969,  0.03463955, 0.03560656,  0.03269066]
#
# early_top_mean = sum(early_top)/4
# late_top_mean = sum(late_top)/4
#print(early_top_mean, late_top_mean)


# BOTTOM - MIDDLE ONES (AVERAGED) IN LTPFR

early_bottom = [ 0.01660859, -0.0293254,  -0.07539683,  0.00385633]
late_bottom = [0.02778111, -0.01033413, 0.03415179,  0.01919524]


early_bottom_sem = [0.03408088,  0.03445868,  0.03750458,  0.03323388]
late_bottom_sem = [0.03462189,  0.02854616, 0.03389487,  0.03253533]

early_bottom_mean = sum(early_bottom)/4
late_bottom_mean = sum(late_bottom)/4
print(early_bottom_mean, late_bottom_mean)

means = [early_bottom_mean, late_bottom_mean]
sessions = ['Early Sessions (1-4)', 'Late Sessions (17-20)']
x = np.arange(len(sessions))
width = .005


fig = plt.figure()
ax = fig.add_subplot(111)

plt.bar(x, means, width = 0.5,  align = 'center', alpha = .5 )
plt.xticks(x, sessions, fontsize= 18)

plt.ylabel(' Change in Recall Probability', fontsize= 22)
plt.title('Low Freq - Middle Freq (ltpFR)', fontsize= 22)
plt.show()



#######
#
# means = [early_bottom_mean, late_bottom_mean]
# sessions = ['Early Sessions (1-4)', 'Late Sessions (17-20)']
# x = np.arange(len(sessions))
# width = .005
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# plt.bar(x, means, width = 0.5,  align = 'center', alpha = .5 )
# plt.xticks(x, sessions, fontsize= 18)
#
# plt.ylabel(' Change in Recall Probability', fontsize= 22)
# plt.title('Low Freq - Middle Freq (ltpFR)', fontsize= 22)
# plt.show()

#######

#
# early_all =  [0.02630435,  0.0039903,  -0.06961153, -0.00180736]
# late_all = [0.01449323, -0.01589286, 0.02892157,  0.00112377]
#
# early_all_sem = [0.02766832,  0.02905974,  0.030421,    0.02919361]
# late_all_sem = [0.03032999,  0.0247751, 0.02842167,  0.02633657]
#
# early_all_mean = sum(early_all)/4
# late_all_mean = sum(late_all)/4
# print(early_all_mean, late_all_mean)
#
# means = [early_all_mean, late_all_mean]
# sessions = ['Early Sessions (1-4)', 'Late Sessions (17-20)']
# x = np.arange(len(sessions))
# width = .005
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# plt.bar(x, means, width = 0.5,  align = 'center', alpha = .5 )
# plt.xticks(x, sessions, fontsize= 18)
#
# plt.ylabel(' Change in Recall Probability', fontsize= 22)
# plt.title('High Freq + Low Freq - Middle Freqs (ltpFR)', fontsize= 22)
# plt.show()