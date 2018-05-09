
"""Slopes for each participant between p_rec list and every lists' semantic meaningfulness by looking at
how each word is meaningful compared to the other words in the whole wordlist"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec.Final import MPool_byPoolSim as m_pool, P_rec_ltpFR2_by_list as prbl

nan = np.nan

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_probs_imported = prbl.get_ltpFR2_precall(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
all_mpool = all_means_mpool
all_probs = np.array(all_probs_imported)
all_mpool = np.array(all_mpool)

print(np.shape(all_mpool))


all_probs_final = []
all_mpool_final = []
slopes_participants = []

# For every participant, get their lists' values (552), mask NaNs from P_recs, find the slope through the 552 lists, add slopes into a new list
# Conduct a t-test to the slopes to see whether they're different from a zero distribution

for i in range(len(all_probs)):
    all_probs_part = all_probs[i]
    all_mpool_part = all_mpool[i]

    mask = np.isfinite(all_probs_part)
    all_probs_part = np.array(all_probs_part)[mask]
    all_mpool_part = np.array(all_mpool_part)[mask]

    #print(len(all_probs_part))
    #print(len(all_num_top_ten_part))

    slope, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_mpool_part, all_probs_part)
    print(slope)

    slopes_participants.append(slope)
    all_probs_final.append(all_probs_part)
    all_mpool_final.append(all_mpool_part)


#print((np.array(all_probs_final)))
#print((np.array(all_num_top_ten_final)))
print(len(slopes_participants))
print(slopes_participants)

t_test = scipy.stats.ttest_1samp(slopes_participants, 0, axis=0)
print(t_test)


# Create data
N = len(all_probs)
x = np.zeros(N)
y = slopes_participants

# Plot
plt.scatter(x, y, alpha=0.5)
#plt.title('Slopes')
plt.title("Slopes: Word Similarities to All Other Word Pool In Each List & P-Rec")
plt.ylim(-2.5,3.5)
plt.xticks([])
#plt.title("Slopes: Semantic Meaningfulness of List & P-Rec")
plt.show()
"""Output
N = 75
[-1.9638403099859449, 1.4775239869478993, 0.059391785587335058, -0.68918710831975338, 1.0954769134978113, 0.87346026845561087, 0.39831899045877672, 0.15629387785136201, -1.4379076344205859, -0.35865943964186003, -0.53981503480698845, 2.4497556436487562, -0.4062691887248544, 1.6288970645377792, 1.5261015081619824, 1.0683660094869727, 2.3086069492297026, 0.47248253854940525, 0.77186400383213494, -1.9024166608814048, -1.156070309072198, 0.23831411548302867, 0.80544706685077816, 1.6712015544155285, -0.12685538395732601, -0.76393246290789307, 1.851862080895184, 2.3728412844774263, 1.869126124953399, 0.98743524639125979, 1.6049719568529697, 0.64953051474089962, -1.1988897143012993, 0.56389313504858252, 0.79292917431683541, 0.093235505427831863, 0.44209404023790816, -0.4544281528904231, 0.32531193675176817, -0.41157027864930612, 1.9458204714827128, 1.1156329312262148, -0.28812791406526672, 0.85538482265969362, 0.41458514210120379, 3.1572942543186766, 0.70824818592816619, 1.677455460064827, -1.6808972113912302, -0.44039596183925617, -1.1930037009945862, 0.40478466833713062, 0.83647050448281957, 2.2213317120564753, 1.1828654882573997, 0.47477529358147508, 1.5271905421906358, 1.3825987953763437, 2.2618717927722645, 1.1742480542798324, -1.25207641755183, 0.21531158939092815, 0.90388998661368902, 0.38361009560485598, 0.033091191298862448, 0.85582592397204915, 0.12898748651395697, -0.1400537665578897, 2.2235498809851117, 0.2459644883373544, -0.86116158264876508, 0.98601413540599336, 0.037034226338201366, 0.10459886671739077, 0.73526525088154737]
Ttest_1sampResult(statistic=4.1344788582386141, pvalue=9.2822542287269262e-05)

"""