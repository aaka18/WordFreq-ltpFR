import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table,Column
import math

# Array below is the word frequencies of all items in the word pool in an ascending order


array = [2	,
2	,
3	,
6	,
6	,
7	,
11	,
11	,
11	,
12	,
13	,
19	,
20	,
20	,
22	,
24	,
25	,
25	,
26	,
28	,
28	,
29	,
29	,
33	,
34	,
35	,
35	,
37	,
37	,
37	,
38	,
39	,
41	,
42	,
43	,
44	,
44	,
44	,
45	,
46	,
46	,
47	,
47	,
47	,
47	,
47	,
47	,
48	,
48	,
54	,
56	,
56	,
56	,
57	,
58	,
61	,
64	,
65	,
66	,
67	,
67	,
67	,
68	,
70	,
71	,
73	,
74	,
74	,
75	,
76	,
76	,
77	,
79	,
79	,
80	,
81	,
82	,
83	,
83	,
84	,
84	,
87	,
87	,
88	,
96	,
99	,
99	,
107	,
109	,
113	,
115	,
115	,
115	,
116	,
117	,
117	,
117	,
118	,
120	,
120	,
122	,
122	,
123	,
129	,
129	,
132	,
133	,
139	,
140	,
142	,
144	,
145	,
146	,
146	,
147	,
148	,
148	,
148	,
150	,
152	,
153	,
153	,
156	,
161	,
165	,
166	,
166	,
168	,
170	,
170	,
172	,
177	,
178	,
182	,
182	,
183	,
184	,
184	,
185	,
185	,
186	,
187	,
189	,
198	,
201	,
204	,
205	,
206	,
207	,
211	,
217	,
222	,
224	,
229	,
231	,
231	,
237	,
237	,
237	,
240	,
242	,
246	,
251	,
253	,
253	,
254	,
258	,
266	,
270	,
272	,
273	,
273	,
280	,
280	,
281	,
281	,
285	,
286	,
289	,
291	,
294	,
298	,
305	,
305	,
315	,
318	,
321	,
327	,
330	,
333	,
335	,
342	,
344	,
350	,
351	,
367	,
371	,
372	,
372	,
381	,
381	,
400	,
400	,
412	,
416	,
416	,
438	,
440	,
442	,
444	,
459	,
463	,
465	,
465	,
466	,
476	,
480	,
485	,
487	,
495	,
497	,
498	,
503	,
513	,
521	,
522	,
531	,
536	,
538	,
545	,
555	,
556	,
560	,
568	,
573	,
592	,
617	,
619	,
629	,
635	,
637	,
655	,
656	,
658	,
704	,
705	,
707	,
718	,
732	,
737	,
746	,
761	,
762	,
774	,
775	,
778	,
812	,
882	,
885	,
889	,
892	,
911	,
930	,
952	,
961	,
1003	,
1061	,
1077	,
1137	,
1196	,
1209	,
1214	,
1219	,
1219	,
1227	,
1280	,
1332	,
1380	,
1381	,
1414	,
1464	,
1469	,
1490	,
1518	,
1525	,
1531	,
1546	,
1610	,
1644	,
1753	,
1792	,
1860	,
1860	,
1905	,
1933	,
1938	,
1954	,
2225	,
2246	,
2259	,
2333	,
2372	,
2374	,
2405	,
2597	,
2844	,
3020	,
3084	,
3087	,
3128	,
3281	,
3587	,
3645	,
3777	,
4460	,
4944	,
5113	,
5243	,
6036	,
6072	,
7226	,
7645	,
7889	,
11914	,
13345	]

# x = [9, 2, 8, 4, 8, 9, 6, 8, 9, 9, 9, 2, 7, 9, 5, 6, 2, 3, 9, 7, 0, 8, 6, 9, 8, 9, 9, 9, 2, 7, 8, 0, 8, 8, 1, 7, 0, 8, 6, 9, 0, 5, 3, 5, 4, 8, 2, 3, 1, 5, 0, 9, 0, 8, 8, 0, 2, 9, 0, 8, 1, 7, 0, 9, 9, 0, 0, 2, 6, 2, 3, 0, 0, 9, 0, 7, 8, 5, 1, 1, 0, 5, 3, 2, 7, 1, 2, 0, 0, 6, 7, 8, 7, 1, 9, 5, 1, 4, 5, 4, 3, 0, 2, 0, 0, 0, 1, 1, 4, 5, 5, 4, 5, 0, 2, 5, 1, 4, 9, 9, 0, 7, 3, 6, 0, 3, 2, 0, 7, 0, 2, 4, 3, 2, 1, 8, 5, 0, 7, 2, 1, 1, 8, 6, 7, 9, 0, 9, 2, 9, 4, 4, 4, 4, 5, 5, 4, 0, 0, 5, 1, 4, 0, 9, 0, 4, 2, 7, 7, 7, 4, 3, 1, 1, 1, 0, 4, 5, 5, 5, 3, 7, 4, 3, 0, 8, 4, 8, 6, 0, 6, 2, 7, 3, 0, 0, 4, 0, 0, 5, 1, 1, 5, 1, 3, 1, 0, 9, 5, 7, 6, 4, 0, 4, 7, 4, 4, 3, 2, 4, 0, 0, 0, 3, 6, 7, 0, 4, 0, 6, 1, 9, 4, 5, 1, 8, 6, 3, 4, 5, 7, 1, 0, 3, 5, 7, 3, 7, 1, 4, 6, 2, 1, 1, 2, 1, 3, 2, 0, 7, 0, 0, 9, 0, 8, 4, 1, 9, 4, 8, 2, 4, 1, 7, 5, 3, 9, 0, 3, 3, 0, 9, 5, 5, 2, 7, 2, 9, 0, 7, 0, 2, 0, 5, 1, 3, 9, 1, 0, 6, 0, 1, 2, 3, 7, 1, 1, 0, 1, 2, 6, 1, 4, 7, 0, 7, 0, 2, 9, 2, 9, 2, 3, 6, 1, 3, 2, 0, 5, 4, 5, 7, 5, 9, 1, 5, 9, 3, 0, 2, 4, 1, 4, 0, 1, 1, 0, 2, 1, 3, 0, 1, 0, 0, 1, 0, 5, 4, 9, 0, 3, 1, 0, 9, 0, 1, 7, 5, 5, 1, 7, 4, 3, 1, 7, 4, 0, 0, 0, 1, 4, 1, 3, 5, 3, 0, 1, 0, 1, 0, 1, 5, 4, 4, 3, 8, 0, 0, 2, 2, 1, 1, 5, 6, 0, 1, 3, 0, 8, 4, 3, 0, 5, 3, 0, 0, 9, 1, 3, 3, 1, 2, 0, 8, 0, 8, 0, 3, 9, 2, 7, 2, 4, 3, 1, 0, 6, 0, 1, 4, 0, 1, 1, 7, 2, 2, 4, 7, 6, 5, 8, 5, 5, 3, 5, 9, 4, 2, 8, 0, 4, 5, 1, 1, 1, 4, 6, 0, 0, 2, 2, 3, 2, 0, 0, 4, 2, 0, 6, 2, 2, 0, 6, 0, 1, 2, 1, 2, 8, 0, 8, 5, 1, 1, 6, 0, 6, 8, 8, 4, 2, 4, 0, 1, 1, 0, 6, 1, 7, 5, 1, 7, 0, 2, 7, 0, 1, 0, 1, 8, 6, 1, 3, 8, 0, 0, 4, 6, 6, 3, 0, 6, 1, 7, 0, 6, 0, 0, 9, 9, 0, 0, 1, 4, 2, 4, 0, 1, 9, 3, 3, 7, 0, 3, 0, 7, 7, 2, 4, 2, 6, 6, 2, 2, 1, 6, 1, 1, 3, 5, 4, 8, 7, 3, 7, 0]
#
# plt.hist(x, bins =10)
# plt.suptitle('ltpFR2 Word Pool Binned With ltpFR Bin Boundaries', fontsize=20)
# plt.xlabel('Bins 1 Through 10', fontsize=18)
# plt.ylabel('Total Number of Words In Each Word Bin', fontsize=16)
# plt.xticks([])
# plt.show()

print("Mean is = ", np.mean(array))
print(array)
print('Length of list of array_sorted:', len(array))

bins = []

# Finding bin edges by looking at the word pool in groups of 1/10 and identifying the max bin values
for i in range(1,11):
    edge = array[int(i*len(array)/10)-1]
    bins.append(edge)

print(bins)

bin_zero = []
bin_one = []
bin_two = []
bin_three = []
bin_four = []
bin_five = []
bin_six = []
bin_seven = []
bin_eight = []
bin_nine = []


# Add the word frequency values to their corresponding bins depending on the max bin values
for i, freq in enumerate(array):
    if freq >= 0 and freq <= bins[0]:
        bin_zero.append([i, freq])
    elif (bins[0] < freq) and (freq <= bins[1]):
        bin_one.append([i, freq])
    elif (bins[1] < freq) and (freq <= bins[2]):
        bin_two.append([i, freq])
    elif (bins[2] < freq) and (freq <= bins[3]):
        bin_three.append([i, freq])
    elif (bins[3] < freq) and (freq <= bins[4]):
        bin_four.append([i, freq])
    elif (bins[4] < freq) and (freq <= bins[5]):
        bin_five.append([i, freq])
    elif (bins[5] < freq) and (freq <= bins[6]):
        bin_six.append([i, freq])
    elif (bins[6] < freq) and (freq <= bins[7]):
        bin_seven.append([i, freq])
    elif (bins[7] < freq) and (freq <= bins[8]):
        bin_eight.append([i, freq])
    elif (bins[8] < freq) and (freq  <= bins[9]):
        bin_nine.append([i, freq])

# Print lengths of each bin

print(len(bin_zero))
print(len(bin_one))
print(len(bin_two))
print(len(bin_three))
print(len(bin_four))
print(len(bin_five))
print(len(bin_six))
print(len(bin_seven))
print(len(bin_eight))
print(len(bin_nine))

# Sum of the lengths of each bin should be equal to the total number of words in the word pool
print((len(bin_zero) + len(bin_one) + len(bin_two) + len(bin_three) + len(bin_four) + len(bin_five) + len(bin_six) + len(bin_seven) + len(bin_eight) + len(bin_nine)))

# Confirming that the max bin values match the initial max bin values
print("Bin Max Values")
print(array[32-1])
print(array[(32+33-1)])
print(array[(32+33+32-1)])
print(array[(32+33+32+33-1)])
print(array[(32+33+32+33+32-1)])
print(array[(32+33+32+33+32+33-1)])
print(array[(32+33+32+33+32+33+32-1)])
print(array[(32+33+32+33+32+33+32+33-1)])
print(array[(32+33+32+33+32+33+32+33+33-1)])
print(array[(32+33+32+33+32+33+32+33+33+32-1)])

# Finding the mean word frequency of each word bin

mean_zero = (np.mean(np.asarray(bin_zero), axis=0)[1])
mean_one = (np.mean(np.asarray(bin_one), axis=0)[1])
mean_two = (np.mean(np.asarray(bin_two), axis=0) [1])
mean_three = (np.mean(np.asarray(bin_three), axis=0)[1])
mean_four = (np.mean(np.asarray(bin_four), axis=0) [1])
mean_five = (np.mean(np.asarray(bin_five), axis=0)[1])
mean_six = (np.mean(np.asarray(bin_six), axis=0)[1])
mean_seven = (np.mean(np.asarray(bin_seven), axis=0)[1])
mean_eight = (np.mean(np.asarray(bin_eight), axis=0)[1])
mean_nine = (np.mean(np.asarray(bin_nine), axis=0) [1])

# Creating a table: Frequency Information for Each Word Bin

Bin = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Range = [" 0 - 18   ", " 19 - 37   ", "  38 - 52   ", " 53 - 85   ", " 86 - 143   ", " 144 - 204   ", " 205 - 299   ", " 300 - 536   ", "537 - 1,219  ", "  1,220 - 13,345 "]
M = [math.floor(mean_zero), round(mean_one), round(mean_two), round(mean_three), round(mean_four), round(mean_five), round(mean_six), round(mean_seven), round(mean_eight), round(mean_nine)]
print(M)

t = Table([Bin, Range, M], names = ("Bin", "Range", "M"), dtype=('int', 'str', 'int'))
print('\033[1m' + "Table 1")
print("Frequency Information for Each Word Bin")
print('\033[0m')
print(t)
