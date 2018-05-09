import numpy as np
import math
from scipy.stats.stats import pearsonr

concreteness = [562	,
0	,
537	,
568	,
575	,
607	,
585	,
0	,
549	,
580	,
633	,
0	,
0	,
0	,
0	,
0	,
0	,
589	,
542	,
543	,
564	,
579	,
0	,
580	,
0	,
0	,
549	,
623	,
0	,
569	,
0	,
0	,
587	,
0	,
588	,
0	,
626	,
502	,
592	,
0	,
601	,
627	,
637	,
608	,
0	,
0	,
595	,
0	,
0	,
0	,
581	,
0	,
554	,
0	,
0	,
0	,
523	,
0	,
613	,
573	,
580	,
0	,
543	,
450	,
0	,
0	,
607	,
0	,
606	,
609	,
443	,
0	,
0	,
0	,
585	,
0	,
620	,
0	,
0	,
0	,
0	,
519	,
0	,
612	,
617	,
0	,
0	,
0	,
614	,
579	,
0	,
0	,
0	,
587	,
606	,
602	,
0	,
0	,
0	,
0	,
0	,
0	,
600	,
0	,
0	,
538	,
644	,
585	,
547	,
617	,
0	,
0	,
0	,
611	,
0	,
605	,
558	,
602	,
613	,
597	,
0	,
586	,
589	,
640	,
0	,
583	,
612	,
0	,
556	,
0	,
0	,
0	,
0	,
596	,
607	,
0	,
600	,
0	,
0	,
0	,
609	,
596	,
591	,
585	,
0	,
670	,
622	,
609	,
0	,
632	,
545	,
0	,
502	,
630	,
0	,
0	,
564	,
621	,
599	,
584	,
579	,
604	,
0	,
635	,
595	,
550	,
553	,
616	,
640	,
547	,
588	,
0	,
0	,
0	,
576	,
0	,
575	,
535	,
592	,
0	,
560	,
0	,
645	,
589	,
0	,
595	,
606	,
0	,
597	,
579	,
506	,
611	,
545	,
0	,
0	,
0	,
0	,
525	,
515	,
0	,
595	,
0	,
558	,
0	,
644	,
583	,
0	,
540	,
538	,
527	,
0	,
587	,
613	,
595	,
0	,
585	,
626	,
580	,
611	,
0	,
0	,
0	,
611	,
570	,
606	,
0	,
611	,
584	,
0	,
635	,
577	,
595	,
0	,
575	,
500	,
578	,
481	,
0	,
593	,
0	,
0	,
0	,
611	,
617	,
0	,
580	,
512	,
0	,
595	,
0	,
609	,
0	,
0	,
545	,
616	,
0	,
0	,
594	,
576	,
0	,
381	,
0	,
552	,
572	,
629	,
465	,
555	,
574	,
587	,
593	,
0	,
428	,
564	,
525	,
605	,
558	,
597	,
379	,
568	,
617	,
547	,
570	,
595	,
0	,
569	,
0	,
0	,
0	,
636	,
0	,
0	,
623	,
581	,
604	,
0	,
607	,
587	,
0	,
559	,
549	,
0	,
517	,
0	,
0	,
579	,
0	,
0	,
590	,
0	,
0	,
584	,
606	,
467	,
0	,
0	,
532	,
595	,
613	,
0	,
590	,
565	,
579	,
645	,
566	,
0	,
591	,
0	,
0	,
610	,
533	,
574	,
0	,
520	,
573	,
0	,
581	,
560	,
582	,
0	,
0	,
590	,
553	,
633	,
547	,
607	,
479	,
0	,
599	,
349	,
0	,
0	,
0	,
0	,
568	,
0	,
600	,
0	,
591	,
0	,
0	,
532	,
0	,
0	,
554	,
0	,
0	,
593	,
614	,
614	,
0	,
549	,
502	,
515	,
0	,
0	,
0	,
0	,
538	,
0	,
583	,
513	,
0	,
0	,
0	,
558	,
565	,
604	,
0	,
590	,
637	,
615	,
0	,
565	,
0	,
596	,
0	,
0	,
0	,
514	,
0	,
0	,
0	,
0	,
515	,
598	,
603	,
0	,
252	,
0	,
0	,
0	,
599	,
0	,
576	,
0	,
0	,
516	,
0	,
487	,
0	,
597	,
0	,
0	,
558	,
0	,
541	,
602	,
0	,
0	,
586	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
409	,
487	,
0	,
607	,
594	,
0	,
615	,
570	,
578	,
563	,
580	,
488	,
513	,
551	,
535	,
611	,
578	,
0	,
602	,
552	,
0	,
0	,
623	,
0	,
504	,
0	,
560	,
0	,
459	,
504	,
571	,
0	,
0	,
540	,
547	,
0	,
0	,
0	,
0	,
0	,
429	,
0	,
0	,
516	,
0	,
0	,
549	,
0	,
509	,
600	,
0	,
587	,
0	,
0	,
562	,
580	,
583	,
0	,
0	,
607	,
546	,
0	,
588	,
0	,
525	,
563	,
0	,
0	,
549	,
0	,
0	,
439	,
558	,
0	,
581	,
582	,
0	,
604	,
529	,
572	,
474	,
459	,
0	,
0	,
572	,
544	,
555	,
561	,
0	,
0	,
534	,
533	,
602	,
563	,
0	,
0	,
579	,
417	,
579	,
0	,
591	,
0	,
585	,
606	,
0	,
0	,
389	,
535	,
399	,
467	,
0	,
600	,
0	,
0	,
535	,
0	,
0	,
597	,
436	,
565	,
0	,
0	,
399	,
615	,
0	,
526	,
562	,
0	,
0	,
558	,
591	,
579	,
0	,
0	]

print(concreteness.count(0))
print(576-246)
imageability = [575	,
0	,
612	,
547	,
613	,
634	,
589	,
0	,
537	,
626	,
548	,
0	,
0	,
0	,
0	,
0	,
0	,
608	,
606	,
497	,
571	,
600	,
0	,
574	,
0	,
0	,
603	,
635	,
0	,
575	,
541	,
0	,
618	,
0	,
617	,
0	,
601	,
478	,
593	,
0	,
626	,
626	,
625	,
613	,
0	,
0	,
635	,
0	,
0	,
0	,
619	,
0	,
590	,
0	,
0	,
0	,
552	,
0	,
624	,
521	,
597	,
0	,
578	,
587	,
0	,
0	,
622	,
0	,
624	,
541	,
523	,
0	,
0	,
0	,
616	,
0	,
637	,
0	,
0	,
0	,
0	,
529	,
0	,
633	,
613	,
0	,
0	,
0	,
573	,
612	,
0	,
0	,
0	,
616	,
581	,
539	,
0	,
546	,
0	,
0	,
0	,
0	,
505	,
0	,
585	,
508	,
564	,
571	,
505	,
617	,
0	,
0	,
0	,
602	,
0	,
607	,
597	,
635	,
588	,
591	,
0	,
602	,
612	,
595	,
0	,
596	,
587	,
0	,
596	,
0	,
0	,
0	,
0	,
643	,
602	,
0	,
619	,
0	,
0	,
0	,
633	,
582	,
549	,
633	,
0	,
638	,
577	,
602	,
0	,
617	,
506	,
0	,
504	,
576	,
0	,
0	,
541	,
627	,
597	,
568	,
586	,
598	,
0	,
611	,
599	,
549	,
567	,
612	,
601	,
565	,
632	,
0	,
0	,
0	,
529	,
0	,
581	,
619	,
639	,
0	,
590	,
565	,
588	,
577	,
0	,
595	,
557	,
0	,
602	,
608	,
558	,
573	,
558	,
0	,
0	,
0	,
0	,
599	,
497	,
0	,
590	,
0	,
551	,
0	,
564	,
543	,
0	,
580	,
510	,
513	,
0	,
569	,
573	,
611	,
0	,
522	,
589	,
569	,
609	,
0	,
0	,
0	,
608	,
572	,
579	,
0	,
591	,
618	,
0	,
585	,
597	,
623	,
0	,
600	,
513	,
536	,
499	,
0	,
607	,
0	,
0	,
0	,
591	,
551	,
0	,
556	,
516	,
0	,
621	,
0	,
610	,
0	,
0	,
549	,
601	,
0	,
0	,
570	,
462	,
0	,
538	,
0	,
602	,
572	,
583	,
539	,
578	,
623	,
563	,
588	,
0	,
483	,
558	,
633	,
617	,
539	,
561	,
377	,
614	,
571	,
497	,
600	,
527	,
546	,
518	,
0	,
0	,
0	,
565	,
0	,
555	,
591	,
603	,
582	,
0	,
527	,
560	,
0	,
555	,
541	,
0	,
530	,
0	,
0	,
617	,
0	,
0	,
596	,
0	,
0	,
597	,
599	,
521	,
0	,
0	,
560	,
614	,
618	,
0	,
585	,
547	,
577	,
612	,
599	,
0	,
592	,
0	,
0	,
574	,
577	,
604	,
0	,
510	,
560	,
0	,
572	,
556	,
590	,
0	,
0	,
619	,
547	,
606	,
553	,
597	,
351	,
0	,
632	,
365	,
0	,
0	,
0	,
0	,
625	,
0	,
629	,
0	,
587	,
0	,
0	,
486	,
0	,
0	,
595	,
0	,
0	,
530	,
585	,
584	,
0	,
515	,
460	,
487	,
0	,
521	,
0	,
0	,
464	,
0	,
590	,
578	,
0	,
0	,
0	,
516	,
552	,
562	,
0	,
602	,
615	,
583	,
0	,
592	,
0	,
609	,
0	,
0	,
0	,
501	,
0	,
0	,
0	,
0	,
508	,
588	,
578	,
0	,
578	,
0	,
0	,
0	,
590	,
0	,
604	,
0	,
0	,
518	,
0	,
502	,
0	,
567	,
0	,
0	,
513	,
0	,
492	,
601	,
0	,
0	,
526	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
406	,
526	,
0	,
596	,
586	,
0	,
630	,
565	,
575	,
509	,
529	,
525	,
524	,
583	,
583	,
597	,
558	,
0	,
620	,
518	,
0	,
0	,
583	,
0	,
515	,
0	,
530	,
0	,
421	,
490	,
529	,
0	,
0	,
506	,
558	,
0	,
0	,
0	,
0	,
0	,
470	,
0	,
0	,
571	,
0	,
0	,
553	,
0	,
531	,
562	,
0	,
581	,
0	,
0	,
508	,
619	,
548	,
0	,
0	,
568	,
554	,
0	,
556	,
0	,
541	,
604	,
0	,
334	,
554	,
0	,
0	,
497	,
525	,
0	,
538	,
555	,
0	,
562	,
547	,
561	,
432	,
438	,
0	,
0	,
596	,
538	,
564	,
519	,
0	,
488	,
511	,
553	,
556	,
590	,
0	,
0	,
581	,
411	,
580	,
0	,
621	,
0	,
582	,
607	,
0	,
0	,
391	,
510	,
360	,
457	,
0	,
544	,
0	,
0	,
490	,
0	,
0	,
577	,
369	,
521	,
0	,
0	,
388	,
575	,
0	,
511	,
530	,
0	,
0	,
591	,
614	,
574	,
0	,
0	]

print(imageability.count(0))

word_freq = [3777	,
80	,
889	,
217	,
1464	,
4944	,
1531	,
1797	,
2405	,
6072	,
79	,
541	,
5368	,
254	,
430	,
87	,
138	,
3281	,
592	,
19	,
372	,
2784	,
1219	,
1061	,
3694	,
1779	,
87	,
3587	,
584	,
1490	,
26	,
1266	,
1280	,
50	,
556	,
26	,
400	,
1137	,
1860	,
152	,
258	,
15	,
273	,
185	,
1044	,
83	,
118	,
49	,
285	,
30	,
7645	,
13	,
1380	,
932	,
24	,
84	,
1610	,
1518	,
3	,
64	,
774	,
12	,
1933	,
3087	,
20	,
18	,
115	,
358	,
3	,
83	,
120	,
19	,
1890	,
9	,
718	,
315	,
1017	,
49	,
33	,
69	,
288	,
120	,
103	,
635	,
47	,
106	,
30	,
8	,
351	,
761	,
867	,
785	,
58	,
2844	,
286	,
166	,
71	,
135	,
241	,
198	,
108	,
29	,
7	,
26	,
19	,
45	,
48	,
224	,
238	,
254	,
280	,
211	,
72	,

246	,
50	,
1753	,
177	,
1954	,
35	,
704	,
122	,
412	,
37	,
156	,
73	,
33	,
513	,
87	,
17	,
162	,
195	,
104	,
1209	,
280	,
49	,
28	,
568	,
1219	,
485	,
38	,
38	,
108	,
1938	,
503	,
1792	,
1	,
99	,
2372	,
170	,
229	,
305	,
186	,
235	,
266	,
175	,
13	,
9	,
251	,
205	,
44	,
7889	,
2	,
8	,
762	,
746	,
189	,
84	,
170	,
812	,
129	,
56	,
47	,
12	,
231	,
41	,
285	,
148	,
299	,
270	,
131	,
509	,
172	,
26	,
1332	,
463	,
1196	,
183	,
732	,
465	,
115	,
144	,
17	,
35	,
1	,
184	,
8	,
50	,
253	,
20	,
291	,
47	,
146	,
67	,
50	,
20	,
5113	,
521	,
254	,
10	,
440	,
185	,
204	,
229	,
531	,
184	,
150	,
80	,
2	,
206	,
35	,
371	,
118	,
12	,
617	,
11	,
213	,
37	,
476	,
2246	,
50	,
237	,
41	,
281	,
161	,
885	,
459	,
187	,
36	,
262	,
555	,
44	,
149	,
321	,
705	,
164	,
778	,
46	,
444	,
173	,
84	,
38	,
46	,
96	,
129	,
42	,
99	,
16	,
34	,
629	,
183	,
1381	,
2259	,
2	,
18	,
47	,
6036	,
237	,
545	,
952	,
198	,
63	,
88	,
298	,
146	,
2597	,
165	,
12	,
8	,

5243	,
242	,
289	,
656	,
4460	,
76	,
35	,
85	,
661	,
115	,
6	,
20	,
67	,
132	,
254	,
3645	,
372	,
38	,
28	,
15	,
44	,
144	,
116	,
658	,
49	,
11	,
49	,
46	,
438	,
106	,
57	,
573	,
205	,
20	,
94	,
497	,
15	,
13345	,
76	,
1644	,
31	,
416	,
94	,
142	,
123	,
46	,
75	,
238	,
498	,
294	,
201	,
333	,
7226	,
2333	,
42	,
240	,
83	,
184	,
160	,
25	,
169	,
51	,
67	,
41	,
29	,
6	,
61	,
125	,
73	,
24	,
34	,
10	,
68	,
58	,
37	,
342	,
2225	,
168	,
117	,
2	,
42	,
8	,
3020	,
11	,
536	,
68	,
246	,
270	,
522	,
737	,
201	,
47	,
54	,
129	,
182	,
37	,
21	,
26	,
39	,
222	,
48	,
153	,
330	,
13	,
139	,
37	,
40	,
41	,
29	,
56	,
253	,
178	,
207	,
137	,
1525	,


8	,
400	,
331	,
42	,
6	,
62	,
79	,
24	,
137	,
55	,
892	,
211	,
149	,
272	,
148	,
13	,
6	,
3128	,
10	,
147	,
43	,
143	,
61	,
81	,
1227	,
8	,
930	,
7	,
36	,
161	,
113	,
3084	,
174	,
94	,
619	,
142	,
346	,
33	,
1	,
56	,
237	,
47	,
31	,
49	,
48	,
707	,
82	,
86	,
766	,
237	,
466	,
335	,
318	,
1469	,
281	,
117	,
2374	,
327	,
184	,
74	,
1003	,
166	,
10	,
305	,
58	,
46	,
58	,
167	,
416	,
31	,
25	,
78	,
96	,
153	,
99	,
5	,
14	,
77	,
231	,
22	,
76	,
373	,
102	,
24	,
381	,
15	,
52	,
109	,
45	,
1414	,
77	,
911	,
32	,
273	,
50	,
44	,
485	,
882	,
442	,
11	,
961	,
220	,
116	,
182	,
6	,
53	,
67	,
66	,
31	,
487	,
786	,
278	,
47	,
520	,
21	,
655	,
74	,
29	,
57	,
28	,
67	,
1546	,
495	,
39	,
1214	,
122	,
33	,
32	,
166	,
367	,
344	,
117	,
11	,
560	,
358	,
56	,
22	,
480	,
37	,
14	,
1905	,
1860	,
11	,
19	,
43	,
209	,
84	,
168	,
16	,
11914	,
50	,
122	,
145	,
538	,
30	,
148	,
31	,
785	,
775	,
87	,
183	,
107	,
350	,
465	,
73	,
82	,
70	,
381	,
48	,
65	,
139	,
315	,
223	,
1077	,
637	,
140	,
543	,
35	]

probability = [0.6811501597	,
0.6773162939	,
0.6747603834	,
0.6651757188	,
0.6607028754	,
0.6600638978	,
0.6498402556	,
0.6485623003	,
0.647284345	,
0.6466453674	,
0.6428115016	,
0.6261980831	,
0.6236421725	,
0.6115015974	,
0.6076677316	,
0.6063897764	,
0.6044728435	,
0.6038338658	,
0.6	,
0.6	,
0.5993610224	,
0.5993610224	,
0.5993610224	,
0.5987220447	,
0.5968051118	,
0.5955271565	,
0.5948881789	,
0.5948881789	,
0.5942492013	,
0.5910543131	,
0.5904153355	,
0.5891373802	,
0.5891373802	,
0.5846645367	,
0.5833865815	,
0.5827476038	,
0.5776357827	,
0.5776357827	,
0.5769968051	,
0.5750798722	,
0.5750798722	,
0.5750798722	,
0.5744408946	,
0.5731629393	,
0.571884984	,
0.5699680511	,
0.5693290735	,
0.5686900958	,
0.5686900958	,
0.5680511182	,
0.5674121406	,
0.5667731629	,
0.5661341853	,
0.5661341853	,
0.5654952077	,
0.5654952077	,
0.5654952077	,
0.56485623	,
0.56485623	,
0.5642172524	,
0.5642172524	,
0.5623003195	,
0.5616613419	,
0.5610223642	,
0.5603833866	,
0.5584664537	,
0.557827476	,
0.5565495208	,
0.5559105431	,
0.5559105431	,
0.5559105431	,
0.5552715655	,
0.5552715655	,
0.5552715655	,
0.5546325879	,
0.5539936102	,
0.5539936102	,
0.5533546326	,
0.552715655	,
0.552715655	,
0.552715655	,
0.5520766773	,
0.5514376997	,
0.5514376997	,
0.5501597444	,
0.5495207668	,
0.5495207668	,
0.5495207668	,
0.5482428115	,
0.5476038339	,
0.5476038339	,
0.5476038339	,
0.5469648562	,
0.5469648562	,
0.5463258786	,
0.5450479233	,
0.5450479233	,
0.5444089457	,
0.5444089457	,
0.5444089457	,
0.5437699681	,
0.5437699681	,
0.5431309904	,
0.5424920128	,
0.5424920128	,
0.5418530351	,
0.5418530351	,
0.5412140575	,
0.5405750799	,
0.5405750799	,
0.5405750799	,
0.5405750799	,
0.5399361022	,
0.5380191693	,
0.5380191693	,
0.5373801917	,
0.5373801917	,
0.5354632588	,
0.5354632588	,
0.5354632588	,
0.5348242812	,
0.5341853035	,
0.5341853035	,
0.5335463259	,
0.5329073482	,
0.5322683706	,
0.5322683706	,
0.531629393	,
0.531629393	,
0.5309904153	,
0.5309904153	,
0.5303514377	,
0.5303514377	,
0.5303514377	,
0.5303514377	,
0.5290734824	,
0.5290734824	,
0.5284345048	,
0.5284345048	,
0.5284345048	,
0.5284345048	,
0.5284345048	,
0.5277955272	,
0.5277955272	,
0.5271565495	,
0.5271565495	,
0.5271565495	,
0.5265175719	,
0.5265175719	,
0.5265175719	,
0.5265175719	,
0.5265175719	,
0.5265175719	,
0.5265175719	,
0.5258785942	,
0.5258785942	,
0.5258785942	,
0.5252396166	,
0.5252396166	,
0.5252396166	,
0.524600639	,
0.524600639	,
0.524600639	,
0.5239616613	,
0.5239616613	,
0.5239616613	,
0.5239616613	,
0.5233226837	,
0.5233226837	,
0.5233226837	,
0.5226837061	,
0.5226837061	,
0.5220447284	,
0.5220447284	,
0.5220447284	,
0.5214057508	,
0.5214057508	,
0.5214057508	,
0.5214057508	,
0.5207667732	,
0.5207667732	,
0.5207667732	,
0.5201277955	,
0.5194888179	,
0.5194888179	,
0.5194888179	,
0.5194888179	,
0.5182108626	,
0.5182108626	,
0.5182108626	,
0.5182108626	,
0.5182108626	,
0.517571885	,
0.5169329073	,
0.5162939297	,
0.5156549521	,
0.5156549521	,
0.5156549521	,
0.5156549521	,
0.5156549521	,
0.5156549521	,
0.5150159744	,
0.5150159744	,
0.5150159744	,
0.5143769968	,
0.5143769968	,
0.5137380192	,
0.5137380192	,
0.5130990415	,
0.5130990415	,
0.5130990415	,
0.5130990415	,
0.5124600639	,
0.5124600639	,
0.5118210863	,
0.5118210863	,
0.5118210863	,
0.5111821086	,
0.5111821086	,
0.510543131	,
0.5099041534	,
0.5099041534	,
0.5099041534	,
0.5092651757	,
0.5086261981	,
0.5079872204	,
0.5079872204	,
0.5079872204	,
0.5073482428	,
0.5073482428	,
0.5054313099	,
0.5054313099	,
0.5054313099	,
0.5047923323	,
0.5047923323	,
0.5047923323	,
0.5047923323	,
0.5041533546	,
0.5041533546	,
0.5041533546	,
0.5041533546	,
0.503514377	,
0.5028753994	,
0.5022364217	,
0.5022364217	,
0.5015974441	,
0.5009584665	,
0.5003194888	,
0.5003194888	,
0.5003194888	,
0.4996805112	,
0.4996805112	,
0.4990415335	,
0.4984025559	,
0.4984025559	,
0.4977635783	,
0.4977635783	,
0.4971246006	,
0.4971246006	,
0.496485623	,
0.496485623	,
0.496485623	,
0.496485623	,
0.496485623	,
0.4958466454	,
0.4958466454	,
0.4952076677	,
0.4952076677	,
0.4952076677	,
0.4952076677	,
0.4952076677	,
0.4952076677	,
0.4952076677	,
0.4945686901	,
0.4945686901	,
0.4945686901	,
0.4945686901	,
0.4939297125	,
0.4939297125	,
0.4932907348	,
0.4932907348	,
0.4926517572	,
0.4926517572	,
0.4926517572	,
0.4926517572	,
0.4926517572	,
0.4920127796	,
0.4913738019	,
0.4913738019	,
0.4907348243	,
0.4907348243	,
0.4907348243	,
0.4907348243	,
0.4900958466	,
0.4900958466	,
0.4900958466	,
0.4900958466	,
0.489456869	,
0.4888178914	,
0.4888178914	,
0.4888178914	,
0.4881789137	,
0.4875399361	,
0.4869009585	,
0.4869009585	,
0.4869009585	,
0.4869009585	,
0.4869009585	,
0.4856230032	,
0.4856230032	,
0.4856230032	,
0.4849840256	,
0.4843450479	,
0.4843450479	,
0.4843450479	,
0.4837060703	,
0.4837060703	,
0.4837060703	,
0.4830670927	,
0.4830670927	,
0.4830670927	,
0.4830670927	,
0.4830670927	,
0.4830670927	,
0.4830670927	,
0.482428115	,
0.482428115	,
0.482428115	,
0.482428115	,
0.4817891374	,
0.4817891374	,
0.4811501597	,
0.4811501597	,
0.4811501597	,
0.4798722045	,
0.4798722045	,
0.4798722045	,
0.4798722045	,
0.4792332268	,
0.4792332268	,
0.4779552716	,
0.4779552716	,
0.4779552716	,
0.4773162939	,
0.4766773163	,
0.4766773163	,
0.4766773163	,
0.4760383387	,
0.4760383387	,
0.4760383387	,
0.4760383387	,
0.475399361	,
0.4747603834	,
0.4747603834	,
0.4741214058	,
0.4741214058	,
0.4734824281	,
0.4734824281	,
0.4728434505	,
0.4728434505	,
0.4709265176	,
0.4709265176	,
0.4709265176	,
0.4709265176	,
0.4702875399	,
0.4702875399	,
0.4690095847	,
0.4690095847	,
0.4690095847	,
0.4690095847	,
0.4690095847	,
0.4690095847	,
0.468370607	,
0.468370607	,
0.468370607	,
0.468370607	,
0.4677316294	,
0.4677316294	,
0.4670926518	,
0.4664536741	,
0.4658146965	,
0.4658146965	,
0.4658146965	,
0.4651757188	,
0.4651757188	,
0.4651757188	,
0.4651757188	,
0.4651757188	,
0.4645367412	,
0.4638977636	,
0.4638977636	,
0.4632587859	,
0.4632587859	,
0.4626198083	,
0.4619808307	,
0.4619808307	,
0.4619808307	,
0.4619808307	,
0.4619808307	,
0.4619808307	,
0.461341853	,
0.461341853	,
0.461341853	,
0.4607028754	,
0.4607028754	,
0.4600638978	,
0.4600638978	,
0.4600638978	,
0.4600638978	,
0.4587859425	,
0.4581469649	,
0.4581469649	,
0.4575079872	,
0.4575079872	,
0.4568690096	,
0.4568690096	,
0.4568690096	,
0.4562300319	,
0.4562300319	,
0.4555910543	,
0.4555910543	,
0.4549520767	,
0.454313099	,
0.454313099	,
0.454313099	,
0.454313099	,
0.454313099	,
0.454313099	,
0.4536741214	,
0.4536741214	,
0.4536741214	,
0.4536741214	,
0.4536741214	,
0.4523961661	,
0.4523961661	,
0.4523961661	,
0.4511182109	,
0.4511182109	,
0.4504792332	,
0.4504792332	,
0.4504792332	,
0.4498402556	,
0.4498402556	,
0.449201278	,
0.4485623003	,
0.4479233227	,
0.4479233227	,
0.447284345	,
0.447284345	,
0.4466453674	,
0.4466453674	,
0.4466453674	,
0.4460063898	,
0.4453674121	,
0.4453674121	,
0.4453674121	,
0.4447284345	,
0.4434504792	,
0.4434504792	,
0.4434504792	,
0.4428115016	,
0.4428115016	,
0.4428115016	,
0.4428115016	,
0.4428115016	,
0.442172524	,
0.4415335463	,
0.4415335463	,
0.4408945687	,
0.4408945687	,
0.4402555911	,
0.4402555911	,
0.4389776358	,
0.4389776358	,
0.4389776358	,
0.4389776358	,
0.4389776358	,
0.4376996805	,
0.4364217252	,
0.4357827476	,
0.4345047923	,
0.4338658147	,
0.4338658147	,
0.4338658147	,
0.4332268371	,
0.4332268371	,
0.4325878594	,
0.4325878594	,
0.4319488818	,
0.4319488818	,
0.4313099042	,
0.4313099042	,
0.4313099042	,
0.4306709265	,
0.4300319489	,
0.4293929712	,
0.4287539936	,
0.4287539936	,
0.428115016	,
0.4268370607	,
0.4268370607	,
0.4268370607	,
0.4268370607	,
0.4261980831	,
0.4242811502	,
0.4242811502	,
0.4236421725	,
0.4236421725	,
0.4230031949	,
0.4230031949	,
0.4223642173	,
0.421086262	,
0.421086262	,
0.4198083067	,
0.4191693291	,
0.4185303514	,
0.4185303514	,
0.4178913738	,
0.4178913738	,
0.4178913738	,
0.4172523962	,
0.4172523962	,
0.4166134185	,
0.4159744409	,
0.4159744409	,
0.4153354633	,
0.4146964856	,
0.4146964856	,
0.4146964856	,
0.414057508	,
0.4134185304	,
0.4121405751	,
0.4108626198	,
0.4102236422	,
0.4089456869	,
0.4083067093	,
0.4083067093	,
0.4076677316	,
0.407028754	,
0.4063897764	,
0.4057507987	,
0.4057507987	,
0.4051118211	,
0.4051118211	,
0.4044728435	,
0.4038338658	,
0.4012779553	,
0.4006389776	,
0.4006389776	,
0.4006389776	,
0.3993610224	,
0.3961661342	,
0.3942492013	,
0.3936102236	,
0.3910543131	,
0.3910543131	,
0.3891373802	,
0.3846645367	,
0.3808306709	,
0.3750798722	,
0.3750798722	,
0.371884984	,
0.3686900958	,
0.3674121406	,
0.3603833866	,
0.3584664537	,
0.3546325879	,
0.3514376997	,
0.3501597444	,
0.3418530351	,
0.3412140575	]


letters = [4	,
9	,
5	,
8	,
6	,
4	,
7	,
8	,
7	,
5	,
2	,
5	,
6	,
7	,
5	,
3	,
8	,
4	,
6	,
5	,
6	,
6	,
4	,
5	,
6	,
6	,
5	,
7	,
5	,
7	,
5	,
6	,
4	,
6	,
5	,
7	,
6	,
3	,
3	,
4	,
6	,
6	,
4	,
5	,
6	,
9	,
7	,
7	,
9	,
7	,
5	,
7	,
7	,
7	,
8	,
6	,
7	,
5	,
7	,
6	,
6	,
5	,
4	,
6	,
7	,
7	,
7	,
5	,
7	,
5	,
4	,
4	,
7	,
7	,
4	,
5	,
6	,
10	,
7	,
7	,
5	,
5	,
7	,
5	,
5	,
6	,
7	,
5	,
5	,
6	,
6	,
5	,
8	,
6	,
5	,
4	,
8	,
6	,
6	,
5	,
6	,
9	,
5	,
7	,
5	,
9	,
4	,
8	,
7	,
5	,
5	,
10	,
4	,
5	,
8	,
4	,
3	,
6	,
6	,
3	,
8	,
5	,
6	,
6	,
8	,
5	,
5	,
7	,
7	,
6	,
6	,
8	,
6	,
5	,
5	,
5	,
6	,
6	,
5	,
4	,
6	,
5	,
5	,
3	,
4	,
6	,
6	,
6	,
5	,
5	,
5	,
4	,
5	,
4	,
6	,
4	,
7	,
5	,
5	,
6	,
4	,
7	,
8	,
4	,
6	,
6	,
7	,
8	,
5	,
6	,
6	,
6	,
5	,
8	,
7	,
7	,
9	,
6	,
5	,
5	,
4	,
5	,
6	,
5	,
7	,
8	,
6	,
5	,
7	,
7	,
7	,
6	,
5	,
6	,
6	,
6	,
5	,
9	,
5	,
6	,
5	,
6	,
7	,
5	,
6	,
5	,
4	,
6	,
7	,
4	,
6	,
6	,
6	,
5	,
8	,
4	,
4	,
8	,
6	,
5	,
5	,
7	,
6	,
5	,
8	,
5	,
5	,
6	,
5	,
7	,
5	,
7	,
5	,
5	,
6	,
7	,
4	,
3	,
7	,
7	,
7	,
6	,
3	,
7	,
7	,
5	,
5	,
5	,
6	,
6	,
6	,
9	,
10	,
5	,
6	,
6	,
5	,
5	,
4	,
6	,
5	,
5	,
5	,
7	,
7	,
7	,
6	,
4	,
4	,
7	,
4	,
4	,
5	,
5	,
5	,
6	,
5	,
5	,

4	,
5	,
6	,
5	,
6	,
5	,
6	,
6	,
6	,
6	,
6	,
6	,
4	,
4	,
6	,
5	,
6	,
3	,
7	,
5	,
4	,
9	,
6	,
5	,
6	,
6	,
6	,
8	,
8	,
6	,
4	,
8	,
6	,
8	,
7	,
6	,
8	,
5	,
5	,
6	,
7	,
6	,
6	,
6	,
7	,
5	,
7	,
6	,
5	,
5	,
7	,
7	,
5	,
4	,
5	,
5	,
7	,
4	,
8	,
6	,
5	,
7	,
4	,
8	,
8	,
7	,
5	,
5	,
6	,
6	,
6	,
7	,
6	,
4	,
6	,
4	,
7	,
7	,
6	,
9	,
7	,
6	,
6	,
6	,
5	,
8	,
5	,
8	,
6	,
5	,
5	,
4	,
7	,
6	,
6	,
9	,
6	,
5	,
4	,
5	,
8	,
6	,
5	,
8	,
5	,
6	,
5	,
8	,
5	,
5	,
6	,
5	,
5	,
7	,
5	,
7	,
5	,
6	,
7	,
5	,
5	,
8	,
7	,
8	,
6	,
7	,
5	,
4	,
7	,
6	,
6	,
6	,
5	,
7	,
8	,
4	,
5	,
7	,
7	,
6	,
7	,
5	,
5	,
5	,
6	,
5	,
6	,
6	,
6	,
8	,
5	,
7	,
7	,
7	,
6	,
4	,
6	,
6	,
5	,
5	,
7	,
5	,
4	,
5	,
6	,
5	,
8	,
7	,
7	,
7	,
6	,
6	,
6	,
5	,
5	,
6	,
6	,
6	,
4	,
7	,
4	,
5	,
5	,
5	,
7	,
6	,
8	,
6	,
5	,
5	,
4	,
9	,
4	,
5	,
7	,
7	,
9	,
6	,
4	,
6	,
8	,
7	,
5	,
6	,
5	,
10	,
7	,
4	,
6	,
6	,
5	,
7	,
5	,
5	,
6	,
6	,
8	,
7	,
6	,
8	,
9	,
6	,
5	,
5	,
4	,
6	,
7	,
6	,
7	,
7	,
7	,
5	,
6	,
6	,
6	,
5	,
4	,
5	,
5	,
7	,
7	,
6	,
7	,
7	,
7	,
4	,
5	,
7	,
4	,
7	,
5	,
5	,
6	,
6	,
7	,
7	,
6	,
7	,
5	,
4	,
7	,
6	,
4	,
6	,
6	,
6	,
6	,
5	,
8	,
6	,
4	,
7	,
6	,
5	,
5	,
5	,
6	,
4	,
5	,
6	,
7	,
6	,
4	,
6	,
5	,
6	,
6	,
5	,
6	,
5	,
7	,
6	,
7	]

# print(len(concreteness), len(word_freq), len(imageability), len(probability))
#print(imageability)
#print(word_freq)
#print(p_rec)
print("Pearson r Imageability & Conc", pearsonr(imageability, concreteness))
print("Pearson r Imageability", pearsonr(imageability, probability))
print("Pearson r Concereteness", pearsonr(concreteness, probability))
print("Pearson r Word Freq", pearsonr(word_freq, probability))
print("Pearson r Word Length", pearsonr(letters, probability))
#print("Pearson r Number of words", pearsonr(letters, p_rec))
c1 = np.corrcoef([imageability, concreteness, word_freq, letters, probability])
print(c1)