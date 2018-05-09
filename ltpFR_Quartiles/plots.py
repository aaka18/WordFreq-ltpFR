import matplotlib.pyplot as plt
import numpy as np

##LTPFR2 (Numbers taken from Quartiles Top + Last - Middles .py)
# TOP + LAST QUARTILE (averaged) - MIDDLE ONES (averaged) - ltpFR updated

#y = [ 0.02630435,  0.0039903,  -0.06961153, -0.00180736]
y =   [0.01449323, -0.01589286, 0.02892157,  0.00112377]

# sd_y = [ 0.02766832,  0.02905974,  0.030421,    0.02919361]
sd_y = [0.03032999,  0.0247751, 0.02842167,  0.02633657]


# Bottom  - MIDDLE ONES (averaged) - Updated ltpFR
#t =  [0.01660859, -0.0293254,  -0.07539683,  0.00385633]
t =  [0.02778111, -0.01033413, 0.03415179,  0.01919524]

#sd_t = [ 0.03408088,  0.03445868,  0.03750458,  0.03323388]
sd_t = [0.03462189,  0.02854616, 0.03389487,  0.03253533]

# TOP - MIDDLE ONES (averaged) - Updated ltpFR

# z = [ 0.02133178,  0.00863345, -0.05256944,  0.01945462]
z = [0.00078709, -0.00277441, 0.05728228, -0.02757619]

# sd_z = [  0.03323699,  0.03593055,  0.03531979,  0.03578077]
sd_z =  [0.03729969,  0.03463955, 0.03560656,  0.03269066]




x = [1,2,3,4]
N = 4
x = np.arange(4)
width = .27

# print(len(x))
#plt.plot(x, y)

fig = plt.figure()
ax = fig.add_subplot(111)

bar_y = ax.bar(x, y, width, color = 'r', yerr = sd_y)
bar_z = ax.bar(x + width, z, width, color = 'g', yerr = sd_z)
bar_t = ax.bar(x + width * 2, t, width, color = 'b', yerr = sd_t)

plt.suptitle('Word Frequency ltpFR Late Sessions', fontsize=20)
ax.set_xticks(x+width)
ax.set_xticklabels( ('Top + Bottom - Middle Quartiles', 'Top - Middle Quartiles', 'Bottom - Middle Quartiles') )
ax.legend( (bar_y[0], bar_z[0], bar_t[0]), ('Top + Bottom - Middle Quartiles', 'Top - Middle Quartiles', 'Bottom - Middle Quartiles'), loc='upper right' )


# plt.bar(x, z, width, color="blue")
plt.xticks([])
plt.yticks(fontsize = 16)
plt.xlabel('Sessions 17 - 20', fontsize = 22)
# plt.ylabel('Recall Probability (Top + Bottom Quartiles - Middle Quartiles)', fontsize=14)
plt.ylabel('Change in Recall Probability', fontsize= 22)
plt.ylim((-.15, .15))
plt.show()