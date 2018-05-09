import scipy.io
import numpy as np
import copy

def get_bins():
    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"
    word_freq = scipy.io.loadmat(word_freq_path, squeeze_me=True, struct_as_record=False)

# The array below includes the final indices of 576 ltpFR2 words in the ltpFR word pool, so that their word frequencies could be found

    final_indexes = [1608	,
1148	,
1124	,
1311	,
636	,
178	,
727	,
1619	,
993	,
63	,
1123	,
604	,
793	,
736	,
1535	,
1394	,
1143	,
1439	,
882	,
962	,
808	,
391	,
37	,
979	,
826	,
112	,
23	,
811	,
298	,
345	,
347	,
718	,
994	,
170	,
38	,
605	,
446	,
1628	,
820	,
661	,
795	,
32	,
1455	,
784	,
1030	,
1318	,
1005	,
306	,
1191	,
1485	,
1146	,
1001	,
952	,
93	,
1607	,
1510	,
1274	,
597	,
589	,
622	,
254	,
158	,
414	,
108	,
138	,
723	,
609	,
202	,
744	,
519	,
260	,
592	,
210	,
1202	,
1193	,
894	,
248	,
1610	,
976	,
791	,
371	,
118	,
314	,
1332	,
981	,
133	,
1316	,
682	,
1151	,
934	,
507	,
498	,
1286	,
878	,
83	,
833	,
247	,
706	,
657	,
569	,
754	,
1088	,
1293	,
494	,
264	,
899	,
813	,
760	,
209	,
1326	,
819	,
466	,
168	,
436	,
1521	,
725	,
765	,
1306	,
1095	,
658	,
434	,
193	,
1107	,
396	,
1554	,
303	,
712	,
1142	,
1507	,
653	,
583	,
638	,
1422	,
1223	,
1544	,
1016	,
385	,
54	,
383	,
1214	,
1383	,
1511	,
294	,
954	,
1064	,
1196	,
511	,
1228	,
1047	,
1344	,
848	,
266	,
1346	,
387	,
1518	,
1372	,
1261	,
1138	,
705	,
107	,
1407	,
696	,
626	,
222	,
1414	,
144	,
832	,
198	,
1415	,
1085	,
971	,
623	,
690	,
279	,
691	,
342	,
1425	,
1526	,
860	,
1070	,
666	,
591	,
590	,
489	,
1561	,
1621	,
575	,
340	,
1492	,
1447	,
1331	,
1201	,
156	,
1387	,
1635	,
900	,
171	,
1487	,
907	,
577	,
82	,
1534	,
495	,
465	,
1525	,
1218	,
939	,
762	,
1350	,
1087	,
215	,
1620	,
326	,
327	,
1046	,
1632	,
1384	,
1356	,
428	,
60	,
987	,
838	,
359	,
1250	,
1327	,
90	,
1198	,
698	,
1532	,
1355	,
1139	,
1249	,
412	,
142	,
337	,
1270	,
225	,
1220	,
1010	,
239	,
578	,
402	,
229	,
665	,
1075	,
806	,
484	,
1023	,
641	,
183	,
1056	,
1110	,
851	,
759	,
996	,
617	,
1116	,
870	,
640	,
176	,
1092	,
768	,
1325	,
884	,
701	,
1094	,
1367	,
713	,
74	,
847	,
1504	,
529	,
497	,
496	,
1147	,
1373	,
1298	,
584	,
714	,
182	,
1294	,
560	,
1169	,
599	,
307	,
167	,
1171	,
1018	,
1459	,
438	,
564	,
959	,
377	,
1172	,
887	,
621	,
382	,
500	,
67	,
865	,
634	,
1036	,
1410	,
389	,
1314	,
1500	,
367	,
1060	,
1183	,
1471	,
1340	,
688	,
877	,
936	,
572	,
1354	,
745	,
921	,
1050	,
797	,
165	,
1154	,
1175	,
324	,
1605	]

    # Given the final index, finding the word freq and the appropriate bin #
    final_freq = pool_freq(word_freq, final_indexes)
    bins = (bin(final_freq))
    #print(bins)


    # Creating a word dictionary with the final indices of words and their corresponding word frequency values
    word_dict = {}
    for i in range(len(final_indexes)):
        word_dict[final_indexes[i]] = bins[i]
    return word_dict



#    print((bin(final_freq)).count(0))
#    print((bin(final_freq)).count(1))
#    print((bin(final_freq)).count(2))
#    print((bin(final_freq)).count(3))
#    print((bin(final_freq)).count(4))
#     print((bin(final_freq)).count(5))
#     print((bin(final_freq)).count(6))
#     print((bin(final_freq)).count(7))
#     print((bin(final_freq)).count(8))
#     print((bin(final_freq)).count(9))
#     print(len(bin(final_freq)))




def pool_freq(word_freq, final_indexes):

    # Finding the corresponding word frequencies when the word index is given
    frequency = word_freq['F'].astype('int')
    final_freq = []
    for i in final_indexes:
        final_freq.append(frequency[i])
    #print(final_freq)
    #print("LENGTH IS:", len(final_freq))
    return(final_freq)

def bin(final_freq):
    bins = []

    # Finding the bin number that corresponds to word's frequency value. For example, a word with a freq value of 50 would be in bin 2

    for i in range(len(final_freq)):
        if final_freq[i] >= 0 and final_freq[i] <= 39:
            bins.append(0)
        elif final_freq[i] > 39 and final_freq[i] <= 71:
            bins.append(1)
        elif final_freq[i] > 71 and final_freq[i] <= 117:
            bins.append(2)
        elif final_freq[i] > 117 and final_freq[i] <= 170:
            bins.append(3)
        elif final_freq[i] > 170 and final_freq[i] <= 246:
            bins.append(4)
        elif final_freq[i] > 246 and final_freq[i] <= 351:
            bins.append(5)
        elif final_freq[i] > 351 and final_freq[i] <= 531:
            bins.append(6)
        elif final_freq[i] > 531 and final_freq[i] <= 889:
            bins.append(7)
        elif final_freq[i] > 889 and final_freq[i] <= 1860:
            bins.append(8)
        elif final_freq[i] > 1860 and final_freq[i] <= 13345:
            bins.append(9)
    print(bins)
    return(bins)






if __name__ == "__main__":
    get_bins()

