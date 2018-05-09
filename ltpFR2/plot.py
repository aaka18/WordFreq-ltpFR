import matplotlib.pyplot as plt
import numpy as np

##LTPFR2 (Numbers taken from Quartiles Top + Last - Middles .py)
# TOP + LAST QUARTILE (averaged) - MIDDLE ONES (averaged) - ltpFR2 updated

y = [-0.03355303, -0.00196032, -0.02259615,  0.0077381,   0.09042328, -0.01273516,
   0.02225673,  0.07485714,  0.04183007,  0.0031746,   0.04306319,  0.01304182,
   0.02479762,  0.01661523, -0.0375,     -0.01585394,  0.05191799, -0.03166822,
  -0.04030952, -0.00404687,  0.0591342,  -0.02343604, -0.04044785]

sd_y = [0.04218571,  0.04578768,  0.03858699,  0.03636477 , 0.03611247,  0.03868961,
   0.03859554 , 0.04233527,  0.03224731,  0.04256772,  0.03825422,  0.03542119,
   0.04052632,  0.03739965,  0.04192267,  0.04013161,  0.03727183,  0.03371272,
   0.03240544,  0.02760524,  0.03726731,  0.03858166,  0.03542401]

# Bottom  - MIDDLE ONES (averaged) - Updated ltpFR2
t = [-0.06828898, -0.01515717, -0.06517857, -0.027587,    0.04908815, -0.02551227,
  -0.00900621,  0.0905929,   0.0231456,   0.01872082,  0.02612434,  0.02545024,
   0.06930439, -0.01262755, -0.06740476, -0.02628032,  0.07534632, -0.05369636,
  -0.06570513, -0.03317901,  0.07608225, -0.08731685, -0.05331633]

sd_t = [ 0.0548208,   0.05034604,  0.04802834,  0.04725461,  0.04132479,  0.05214679,
   0.04460864,  0.05296423,  0.04666032,  0.05301937,  0.04810688,  0.04783467,
   0.04827821,  0.04344642,  0.05030509,  0.04839579,  0.04028915,  0.04731578,
   0.03618787,  0.03592062,  0.04553195,  0.04202554,  0.04219545]

# TOP - MIDDLE ONES (averaged) - Updated ltpFR2

z = [ 0.01962142, -0.00862123,  0.01709344,  0.05202156,  0.12158055,  0.00634921,
   0.05024802,  0.06064286,  0.08176808, -0.0034142,   0.04216083,  0.00462228,
  -0.02172619,  0.03844877, -0.0332381,  -0.00540293,  0.02114719,  0.00193071,
  -0.00371148,  0.01443269,  0.04218615,  0.02892157, -0.0285575 ]

sd_z = [ 0.04696429,  0.05605251,  0.04216781,  0.04480449,  0.04171621,  0.04459344,
   0.04547806,  0.04601468,  0.04279247,  0.04145436,  0.04482691,  0.03673691,
   0.04681807,  0.04405741,  0.05081524,  0.0457854,   0.04560516,  0.03561286,
   0.04067768 , 0.03579326,  0.04093555,  0.05097821,  0.04360494]




#x = [1,2,3,4,5,6,7,8,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
N = 23
x = np.arange(23)
width = .27

# print(len(x))
#plt.plot(x, y)

fig = plt.figure()
ax = fig.add_subplot(111)

bar_y = ax.bar(x, y, width, color = 'r', yerr = sd_y)
bar_z = ax.bar(x + width, z, width, color = 'g', yerr = sd_z)
bar_t = ax.bar(x + width * 2, t, width, color = 'b', yerr = sd_t)

plt.suptitle('Word Frequency ltpFR2', fontsize=20)
ax.set_xticks(x+width)
ax.set_xticklabels( ('Top + Bottom - Middle Quartiles', 'Top - Middle Quartiles', 'Bottom - Middle Quartiles') )
ax.legend( (bar_y[0], bar_z[0], bar_t[0]), ('Top + Bottom - Middle Quartiles', 'Top - Middle Quartiles', 'Bottom - Middle Quartiles'), loc='upper right' )


# plt.bar(x, z, width, color="blue")
plt.xticks([])
plt.yticks(fontsize = 16)
plt.xlabel('Sessions 1 - 23', fontsize = 22)
# plt.ylabel('Recall Probability (Top + Bottom Quartiles - Middle Quartiles)', fontsize=14)
plt.ylabel('Change in Recall Probability', fontsize= 22)
plt.show()