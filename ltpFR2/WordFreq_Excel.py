import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table,Column
import math

# Array below is the word frequencies of all items in the word pool in an ascending order


array = [0	,
1	,
1	,
1	,
2	,
2	,
2	,
2	,
3	,
3	,
5	,
6	,
6	,
6	,
6	,
6	,
7	,
7	,
8	,
8	,
8	,
8	,
8	,
8	,
8	,
8	,
9	,
9	,
10	,
10	,
10	,
10	,
11	,
11	,
11	,
11	,
11	,
11	,
12	,
12	,
12	,
12	,
13	,
13	,
13	,
13	,
14	,
14	,
15	,
15	,
15	,
15	,
16	,
16	,
17	,
17	,
18	,
18	,
19	,
19	,
19	,
19	,
20	,
20	,
20	,
20	,
20	,
21	,
21	,
22	,
22	,
24	,
24	,
24	,
24	,
25	,
25	,
26	,
26	,
26	,
26	,
26	,
28	,
28	,
28	,
29	,
29	,
29	,
29	,
30	,
30	,
30	,
31	,
31	,
31	,
31	,
31	,
32	,
32	,
33	,
33	,
33	,
33	,
34	,
34	,
35	,
35	,
35	,
35	,
35	,
36	,
36	,
37	,
37	,
37	,
37	,
37	,
37	,
38	,
38	,
38	,
38	,
39	,
39	,
40	,
41	,
41	,
41	,
41	,
42	,
42	,
42	,
42	,
43	,
43	,
44	,
44	,
44	,
44	,
45	,
45	,
46	,
46	,
46	,
46	,
46	,
47	,
47	,
47	,
47	,
47	,
47	,
47	,
48	,
48	,
48	,
48	,
49	,
49	,
49	,
49	,
49	,
49	,
50	,
50	,
50	,
50	,
50	,
50	,
50	,
51	,
52	,
53	,
54	,
55	,
56	,
56	,
56	,
56	,
57	,
57	,
58	,
58	,
58	,
58	,
61	,
61	,
62	,
63	,
64	,
65	,
66	,
67	,
67	,
67	,
67	,
67	,
68	,
68	,
69	,
70	,
71	,
72	,
73	,
73	,
73	,
74	,
74	,
75	,
76	,
76	,
76	,
77	,
77	,
78	,
79	,
79	,
80	,
80	,
81	,
82	,
82	,
83	,
83	,
83	,
84	,
84	,
84	,
84	,
85	,
85	,
86	,
87	,
87	,
87	,
87	,
88	,
94	,
94	,
94	,
96	,
96	,
99	,
99	,
99	,
102	,
103	,
104	,
106	,
106	,
107	,
108	,
108	,
109	,
113	,
115	,
115	,
115	,
116	,
116	,
117	,
117	,
117	,
118	,
118	,
120	,
120	,
122	,
122	,
122	,
123	,
125	,
129	,
129	,
129	,
131	,
132	,
133	,
135	,
137	,
137	,
138	,
139	,
139	,
140	,
142	,
142	,
143	,
144	,
144	,
145	,
146	,
146	,
147	,
148	,
148	,
148	,
149	,
149	,
150	,
152	,
153	,
153	,
156	,
160	,
161	,
161	,
162	,
164	,
165	,
166	,
166	,
166	,
167	,
168	,
168	,
169	,
170	,
170	,
172	,
173	,
174	,
175	,
177	,
178	,
182	,
182	,
183	,
183	,
183	,
184	,
184	,
184	,
184	,
185	,
185	,
186	,
187	,
189	,
195	,
198	,
198	,
201	,
201	,
204	,
205	,
205	,
206	,
207	,
209	,
211	,
211	,
213	,
217	,
220	,
222	,
223	,
224	,
229	,
229	,
231	,
231	,
235	,
237	,
237	,
237	,
237	,
238	,
238	,
240	,
241	,
242	,
246	,
246	,
251	,
253	,
253	,
254	,
254	,
254	,
254	,
258	,
262	,
266	,
270	,
270	,
272	,
273	,
273	,
278	,
280	,
280	,
281	,
281	,
285	,
285	,
286	,
288	,
289	,
291	,
294	,
298	,
299	,
305	,
305	,
315	,
315	,
318	,
321	,
327	,
330	,
331	,
333	,
335	,
342	,
344	,
346	,
350	,
351	,
358	,
358	,
367	,
371	,
372	,
372	,
373	,
381	,
381	,
400	,
400	,
412	,
416	,
416	,
430	,
438	,
440	,
442	,
444	,
459	,
463	,
465	,
465	,
466	,
475	,
476	,
480	,
485	,
485	,
487	,
495	,
497	,
498	,
503	,
509	,
513	,
520	,
521	,
522	,
531	,
536	,
538	,
541	,
543	,
545	,
555	,
556	,
560	,
568	,
573	,
584	,
592	,
617	,
619	,
629	,
635	,
637	,
655	,
656	,
658	,
661	,
704	,
705	,
707	,
718	,
732	,
737	,
746	,
761	,
762	,
766	,
774	,
775	,
778	,
785	,
785	,
786	,
812	,
867	,
882	,
885	,
889	,
892	,
911	,
930	,
932	,
952	,
961	,
1003	,
1017	,
1044	,
1061	,
1077	,
1137	,
1196	,
1209	,
1214	,
1219	,
1219	,
1227	,
1266	,
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
1779	,
1792	,
1797	,
1860	,
1860	,
1890	,
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
2784	,
2844	,
3020	,
3084	,
3087	,
3128	,
3281	,
3587	,
3645	,
3694	,
3777	,
4460	,
4944	,
5113	,
5243	,
5368	,
6036	,
6072	,
7226	,
7645	,
7889	,
11914	,
13345]

x_ltpFR = [2, 0, 8, 3, 0, 3, 2, 4, 9, 4, 1, 7, 2, 5, 9, 9, 3, 7, 7, 7, 1, 3, 8, 4, 8, 7, 7, 5, 9, 9, 3, 8, 9, 1, 3, 2, 7, 1, 0, 9, 3, 3, 2, 4, 5, 4, 5, 7, 7, 1, 4, 6, 5, 0, 9, 8, 4, 5, 8, 2, 2, 0, 6, 5, 6, 6, 1, 0, 5, 8, 7, 4, 4, 5, 1, 3, 8, 8, 9, 2, 7, 6, 9, 3, 3, 6, 8, 2, 2, 6, 7, 9, 1, 8, 1, 8, 5, 8, 7, 7, 8, 2, 2, 8, 4, 1, 4, 0, 3, 9, 4, 5, 7, 2, 6, 2, 5, 3, 6, 7, 4, 5, 4, 6, 4, 3, 6, 9, 4, 1, 7, 3, 2, 3, 1, 1, 5, 9, 8, 2, 9, 3, 7, 3, 6, 4, 2, 3, 8, 0, 7, 0, 5, 7, 0, 6, 6, 4, 9, 4, 7, 9, 4, 4, 6, 5, 6, 7, 2, 7, 7, 8, 9, 0, 0, 4, 9, 8, 8, 1, 5, 5, 9, 0, 0, 1, 6, 5, 7, 3, 9, 7, 0, 1, 7, 7, 8, 3, 9, 3, 3, 5, 8, 1, 9, 6, 6, 9, 9, 5, 6, 1, 5, 7, 3, 5, 1, 1, 3, 1, 6, 6, 6, 7, 6, 3, 9, 9, 9, 6, 6, 2, 2, 7, 7, 6, 0, 2, 6, 1, 6, 0, 4, 2, 1, 2, 8, 0, 1, 6, 7, 1, 7, 3, 0, 0, 0, 5, 2, 2, 9, 5, 8, 5, 1, 3, 4, 1, 9, 6, 3, 4, 9, 8, 5, 8, 1, 9, 4, 0, 1, 8, 2, 2, 8, 7, 6, 3, 2, 1, 0, 7, 3, 7, 9, 7, 5, 2, 0, 3, 0, 5, 6, 8, 7, 4, 5, 5, 0, 9, 9, 7, 4, 9, 3, 8, 9, 4, 7, 6, 0, 8, 3, 8, 4, 2, 2, 0, 5, 8, 9, 6, 1, 5, 5, 9, 7, 6, 8, 4, 6, 3, 4, 6, 0, 9, 2, 3, 7, 0, 5, 7, 2, 3, 6, 4, 9, 2, 8, 0, 0, 9, 6, 7, 6, 7, 9, 1, 9, 4, 2, 4, 3, 8, 0, 3, 3, 0, 8, 0, 0, 5, 4, 0, 9, 7, 6, 8, 2, 1, 9, 3, 3, 9, 5, 1, 1, 8, 0, 1, 5, 9, 9, 4, 4, 4, 2, 2, 3, 7, 5, 0, 9, 6, 2, 6, 8, 1, 9, 1, 0, 9, 9, 1, 9, 6, 8, 2, 8, 6, 4, 3, 2, 8, 8, 6, 7, 4, 4, 4, 3, 4, 4, 1, 5, 7, 0, 7, 2, 0, 4, 8, 7, 9, 1, 1, 7, 7, 3, 1, 0, 0, 4, 8, 7, 5, 6, 7, 4, 6, 6, 8, 5, 8, 4, 2, 3, 9, 2, 3, 8, 5, 4, 5, 3, 3, 3, 1, 5, 0, 3, 5, 8, 5, 1, 6, 0, 4, 4, 2, 0, 8, 4, 0, 8, 7, 1, 2, 4, 2, 9, 3, 7, 1, 6, 9, 2, 7, 4, 4, 8, 8, 6, 2, 7, 0, 8, 2, 9, 9, 2, 0, 7, 6, 1, 0, 4, 2, 2, 9, 3, 3, 4, 8, 0, 1, 2, 1, 9, 6, 8, 3, 9, 7, 2, 2, 2, 4, 2, 8, 1, 4, 3, 5, 4, 5, 8, 0, 1, 1, 0, 8, 8, 6, 2, 7, 6, 0, 0, 0, 6, 9, 8, 3, 4, 5, 5, 1, 5, 1, 4, 4, 5, 1, 7, 2, 1, 5, 8, 1, 6, 5, 7, 7, 6, 6, 5, 9, 3, 6, 9, 7, 6, 8, 7, 0, 1, 1, 2, 0, 0, 5, 5, 1, 5, 9, 3, 0, 1, 9, 5, 3, 6, 5, 0, 9, 4, 2, 5, 1, 4, 5, 0, 3, 6, 1, 4, 7, 6, 8, 7, 1, 0, 1, 8, 5, 4, 4, 7, 5, 3, 7, 4, 4, 3, 6, 5, 6, 4, 5, 3, 9, 7, 7, 4, 8, 5, 8, 8, 0, 0, 4, 0, 4, 7, 2, 3, 2, 0, 8, 8, 0, 1, 4, 3, 3, 8, 6, 3, 0, 2, 2, 3, 0, 3, 1, 2, 3, 8, 2, 3, 3, 9, 2, 5, 7, 1, 2, 6, 5, 7, 9, 9, 1, 4, 8, 3, 7, 8, 9, 6, 7, 5, 0, 5, 4, 2, 1, 4, 3, 1, 5, 4, 7, 8, 1, 0, 0, 2, 9, 0, 8, 9, 2, 3, 1, 1, 1, 9, 2, 8, 8, 7, 6, 8, 5, 4, 6, 7, 7, 2, 7, 7, 5, 8, 6, 8, 1, 5, 0, 1, 0, 8, 6, 5, 1, 8, 0, 3, 2, 0, 9, 6, 5, 0, 0, 4, 0, 3, 0, 5, 1, 5, 6, 9, 1, 3, 6, 6, 9, 1, 0, 1, 2, 2, 2, 1, 4, 1, 5, 1, 9, 4, 8, 5, 5, 4, 8, 3, 3, 7, 7, 3, 8, 5, 6, 6, 1, 9, 6, 9, 2, 3, 8, 7, 4, 9, 3, 6, 9, 3, 2, 4, 9, 1, 7, 0, 6, 7, 2, 9, 8, 8, 9, 2, 6, 7, 7, 2, 1, 3, 0, 2, 4, 6, 6, 6, 5, 2, 5, 1, 5, 0, 3, 6, 3, 5, 7, 5, 4, 0, 1, 5, 8, 9, 5, 3, 8, 2, 1, 6, 3, 8, 4, 6, 0, 5, 6, 2, 5, 0, 4, 0, 6, 0, 1, 3, 0, 4, 0, 8, 6, 9, 1, 5, 8, 8, 1, 0, 2, 6, 6, 2, 4, 3, 2, 2, 7, 9, 3, 3, 2, 7, 1, 6, 9, 3, 1, 0, 1, 2, 3, 9, 7, 3, 2, 3, 9, 2, 7, 6, 4, 3, 5, 5, 9, 4, 9, 8, 7, 5, 2, 9, 9, 9, 3, 5, 2, 7, 0, 1, 8, 0, 3]
x_ltpFR2 = [9, 3, 8, 6, 9, 9, 7, 9, 9, 9, 9, 3, 8, 9, 6, 7, 4, 4, 9, 8, 1, 8, 7, 9, 8, 9, 9, 9, 4, 8, 9, 1, 9, 9, 2, 8, 1, 8, 7, 9, 0, 6, 5, 6, 5, 8, 3, 4, 2, 6, 1, 9, 0, 9, 8, 1, 3, 9, 0, 9, 3, 8, 0, 9, 9, 1, 0, 4, 7, 3, 4, 0, 0, 9, 1, 8, 8, 7, 2, 3, 1, 6, 4, 4, 8, 2, 4, 0, 1, 7, 8, 8, 8, 3, 9, 6, 3, 5, 6, 5, 4, 1, 4, 0, 1, 1, 2, 2, 6, 6, 6, 6, 6, 0, 3, 6, 2, 5, 9, 9, 1, 8, 4, 7, 1, 5, 3, 1, 7, 0, 4, 5, 5, 4, 2, 8, 6, 1, 8, 4, 2, 2, 8, 7, 7, 9, 0, 9, 4, 9, 6, 5, 5, 6, 6, 7, 5, 0, 0, 6, 2, 6, 0, 9, 0, 5, 3, 8, 8, 8, 5, 4, 2, 3, 2, 0, 6, 6, 6, 6, 5, 7, 5, 4, 1, 8, 5, 9, 7, 0, 7, 4, 8, 5, 1, 0, 5, 1, 0, 6, 2, 2, 6, 2, 5, 3, 1, 9, 6, 7, 7, 5, 0, 5, 7, 6, 5, 5, 3, 6, 0, 1, 0, 4, 7, 8, 0, 6, 1, 7, 2, 9, 6, 6, 2, 8, 7, 5, 5, 6, 8, 2, 1, 5, 7, 8, 5, 8, 2, 5, 7, 3, 2, 2, 4, 2, 4, 4, 0, 8, 1, 0, 9, 0, 9, 5, 2, 9, 6, 8, 4, 5, 3, 8, 6, 5, 9, 0, 5, 4, 0, 9, 6, 6, 3, 8, 3, 9, 1, 8, 0, 4, 1, 6, 3, 4, 9, 2, 1, 7, 0, 2, 4, 5, 8, 2, 2, 0, 2, 4, 7, 3, 6, 8, 1, 7, 0, 4, 9, 3, 9, 4, 4, 7, 2, 4, 3, 1, 6, 5, 6, 7, 7, 9, 2, 6, 9, 5, 1, 3, 5, 2, 5, 1, 3, 2, 0, 3, 3, 4, 1, 3, 1, 0, 3, 1, 7, 5, 9, 0, 4, 2, 0, 9, 0, 3, 7, 6, 6, 2, 8, 5, 4, 3, 7, 5, 1, 1, 1, 2, 6, 2, 5, 7, 4, 0, 2, 1, 3, 1, 2, 6, 6, 5, 4, 9, 0, 0, 3, 3, 3, 2, 7, 7, 0, 3, 4, 1, 8, 6, 5, 0, 6, 5, 0, 0, 9, 2, 5, 4, 3, 3, 0, 9, 0, 8, 1, 5, 9, 4, 8, 4, 5, 4, 3, 0, 7, 1, 2, 6, 1, 2, 2, 8, 4, 3, 6, 8, 7, 7, 9, 7, 6, 4, 7, 9, 5, 3, 8, 0, 5, 7, 3, 2, 3, 5, 7, 1, 1, 3, 4, 5, 4, 0, 0, 6, 3, 1, 7, 3, 4, 1, 7, 0, 2, 4, 2, 3, 9, 1, 8, 6, 2, 2, 7, 0, 7, 8, 8, 6, 4, 5, 0, 3, 3, 1, 7, 3, 8, 6, 2, 7, 1, 3, 8, 1, 3, 1, 3, 9, 7, 2, 4, 8, 1, 1, 5, 7, 7, 4, 0, 7, 3, 8, 1, 7, 1, 0, 9, 9, 0, 1, 2, 6, 3, 5, 0, 2, 9, 4, 5, 8, 1, 5, 1, 8, 8, 4, 5, 4, 7, 7, 3, 3, 3, 7, 2, 3, 4, 7, 6, 8, 8, 4, 8, 1]


# print(len(x_ltpFR))
# plt.hist(x_ltpFR, bins =10)
# plt.suptitle('ltpFR Word Pool Binned With ltpFR Bin Boundaries', fontsize=20)
# plt.xlabel('Bins 1 Through 10', fontsize=18)
# plt.ylabel('Total Number of Words In Each Word Bin', fontsize=16)
# plt.xticks([])
# plt.show()

print("Mean is = ", np.mean(array))
print(array)
print('Length of list of array_sorted:', len(array))

bin_edges = []
for i in range(1,21):
    edge = array[int(i*len(array)/20)-1]
    bin_edges.append(edge)

print("Bins",bin_edges)
print("Len", len(bin_edges))

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
bin_ten = []
bin_eleven = []
bin_twelve = []
bin_thirteen = []
bin_fourteen = []
bin_fifteen = []
bin_sixteen = []
bin_seventeen = []
bin_eighteen = []
bin_nineteen = []


# Add the word frequency values to their corresponding bins depending on the max bin values
for i, freq in enumerate(array):
    if freq >= 0 and freq <= bin_edges[0]:
        bin_zero.append([i, freq])
    elif (bin_edges[0] < freq) and (freq <= bin_edges[1]):
        bin_one.append([i, freq])
    elif (bin_edges[1] < freq) and (freq <= bin_edges[2]):
        bin_two.append([i, freq])
    elif (bin_edges[2] < freq) and (freq <= bin_edges[3]):
        bin_three.append([i, freq])
    elif (bin_edges[3] < freq) and (freq <= bin_edges[4]):
        bin_four.append([i, freq])
    elif (bin_edges[4] < freq) and (freq <= bin_edges[5]):
        bin_five.append([i, freq])
    elif (bin_edges[5] < freq) and (freq <= bin_edges[6]):
        bin_six.append([i, freq])
    elif (bin_edges[6] < freq) and (freq <= bin_edges[7]):
        bin_seven.append([i, freq])
    elif (bin_edges[7] < freq) and (freq <= bin_edges[8]):
        bin_eight.append([i, freq])
    elif (bin_edges[8] < freq) and (freq  <= bin_edges[9]):
        bin_nine.append([i, freq])
    elif (bin_edges[9] < freq) and (freq <= bin_edges[10]):
        bin_ten.append([i, freq])
    elif (bin_edges[10] < freq) and (freq <= bin_edges[11]):
        bin_eleven.append([i, freq])
    elif (bin_edges[11] < freq) and (freq <= bin_edges[12]):
        bin_twelve.append([i, freq])
    elif (bin_edges[12] < freq) and (freq <= bin_edges[13]):
        bin_thirteen.append([i, freq])
    elif (bin_edges[13] < freq) and (freq <= bin_edges[14]):
        bin_fourteen.append([i, freq])
    elif (bin_edges[14] < freq) and (freq <= bin_edges[15]):
        bin_fifteen.append([i, freq])
    elif (bin_edges[15] < freq) and (freq <= bin_edges[16]):
        bin_sixteen.append([i, freq])
    elif (bin_edges[16] < freq) and (freq <= bin_edges[17]):
        bin_seventeen.append([i, freq])
    elif (bin_edges[17] < freq) and (freq  <= bin_edges[18]):
        bin_eighteen.append([i, freq])
    else:
        bin_nineteen.append([i, freq])
# Print lengths of each bin
#
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
print(len(bin_ten))
print(len(bin_eleven))
print(len(bin_twelve))
print(len(bin_thirteen))
print(len(bin_fourteen))
print(len(bin_fifteen))
print(len(bin_sixteen))
print(len(bin_seventeen))
print(len(bin_eighteen))
print(len(bin_nineteen))

# Sum of the lengths of each bin should be equal to the total number of words in the word pool
print((len(bin_zero) + len(bin_one) + len(bin_two) + len(bin_three) + len(bin_four) + len(bin_five) + len(bin_six) + len(bin_seven) + len(bin_eight) + len(bin_nine) + len(bin_ten)+
       len(bin_eleven) + len(bin_twelve) + len(bin_thirteen) + len(bin_fourteen) + len(bin_fifteen) + len(bin_sixteen) + len(bin_seventeen) + len(bin_eighteen) + len(bin_nineteen )))

# Confirming that the max bin values match the initial max bin values
# print("Bin Max Values")
# print(array[58-1])
# print(array[(58+60-1)])
# print(array[(58+60+54-1)])
# print(array[(58+60+54+59-1)])
# print(array[(58+60+54+59+57-1)])
# print(array[(58+60+54+59+57+57-1)])
# print(array[(58+60+54+59+57+57+58-1)])
# print(array[(58+60+54+59+57+57+58+57-1)])
# print(array[(58+60+54+59+57+57+58+57+58-1)])
# print(array[(58+60+54+59+57+57+58+57+58+58-1)])

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
mean_ten = (np.mean(np.asarray(bin_ten), axis=0) [1])
mean_eleven = (np.mean(np.asarray(bin_eleven), axis=0) [1])
mean_twelve = (np.mean(np.asarray(bin_twelve), axis=0) [1])
mean_thirteen = (np.mean(np.asarray(bin_thirteen), axis=0)[1])
mean_fourteen = (np.mean(np.asarray(bin_fourteen), axis=0) [1])
mean_fifteen = (np.mean(np.asarray(bin_fifteen), axis=0)[1])
mean_sixteen = (np.mean(np.asarray(bin_sixteen), axis=0)[1])
mean_seventeen = (np.mean(np.asarray(bin_seventeen), axis=0)[1])
mean_eightteen = (np.mean(np.asarray(bin_eighteen), axis=0)[1])
mean_nineteen = (np.mean(np.asarray(bin_nineteen), axis=0) [1])
# Creating a table: Frequency Information for Each Word Bin

M = [mean_zero, mean_one, mean_two, mean_three,mean_four, mean_five,mean_six, mean_seven, mean_eight, mean_nine, mean_ten,
     mean_eleven, mean_twelve, mean_thirteen, mean_fourteen, mean_fifteen, mean_sixteen, mean_seventeen, mean_eightteen, mean_nineteen]
print(M)

for i in M:
    print(i)
