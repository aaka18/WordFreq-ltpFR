import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec import Mlist10th_percentile as m_list_10
from w2vec import Mpool10th_percentile as m_pool_10
from w2vec.Final import Concreteness_Freq_WordLength as conc, MPool_byPoolSim as m_pool, P_rec_ltpFR2_by_list as prbl, \
    semantic_similarity_bylist as sem_list

# This is the code to find a slope for each participant with their probability of list recall and average all values per list

nan = np.nan
larger_90 = [33, 35, 100, 109, 134, 198, 210, 336, 397, 403, 407, 429, 579, 584, 592, 602, 624, 684, 713, 754, 755, 778, 785, 834, 876, 929, 953, 977, 1016, 1022, 1031, 1071, 1079, 1112, 1139, 1140, 1144, 1147, 1152, 1158, 1209, 1224, 1227, 1238, 1246, 1292, 1332, 1333, 1357, 1385, 1388, 1423, 1472, 1520, 1522, 1524, 1581, 1620]

# List of lists with all participants; 552 p_recs of lists per participant

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_probs_imported = prbl.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc.measures_by_list(files_ltpFR2)
all_concreteness_imported = all_mean_conc_participants
all_wordfreq_imported = all_mean_wordfreq_participants
all_wordlen_imported = all_mean_wordlen_participants
all_means, all_sems, all_parts_each_list_ave_sim = sem_list.all_parts_list_correlations(files_ltpFR2)
all_mlist_imported = all_parts_each_list_ave_sim
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
all_mpool_imported = all_means_mpool
m_list_10 = m_list_10.get_m_list(larger_90, files_ltpFR2)
m_pool_10 = m_pool_10.get_m_pool(files_ltpFR2)


all_probs = np.array(all_probs_imported)
all_concreteness = np.array(all_concreteness_imported)
all_wordfreq = np.array(all_wordfreq_imported)
all_wordlen = np.array(all_wordlen_imported)
all_mlist = np.array(all_mlist_imported)
all_mpool = np.array(all_mpool_imported)
all_mlist_10 = np.array(m_list_10)
all_mpool_10 = np.array(m_pool_10)


slopes_participants_concreteness = []
slopes_participants_wordfreq = []
slopes_participants_wordlen = []
slopes_participants_mlist = []
slopes_participants_mpool = []
slopes_participants_mlist_ten = []
slopes_participants_mpool_ten = []

#print(all_concreteness)

# For every participant, get their lists' values (552), mask NaNs from P_recs, find the slope through the 552 lists, add slopes into a new list
# Conduct a t-test to the slopes to see whether they're different from a zero distribution

for i in range(len(all_probs)):
    all_probs_part = all_probs[i]
    all_concreteness_part = all_concreteness[i]
    all_wordfreq_part = all_wordfreq[i]
    all_wordlen_part = all_wordlen[i]
    all_mlist_part = all_mlist[i]
    all_mpool_part = all_mpool[i]
    all_mlist_10_part = all_mlist_10[i]
    all_mpool_10_part = all_mpool_10[i]


    mask = np.isfinite(all_probs_part)
    all_probs_part = np.array(all_probs_part)[mask]
    all_concreteness_part = np.array(all_concreteness_part)[mask]
    all_wordfreq_part = np.array(all_wordfreq_part)[mask]
    all_wordlen_part = np.array(all_wordlen_part)[mask]
    all_mlist_part = np.array(all_mlist_part)[mask]
    all_mpool_part = np.array(all_mpool_part)[mask]
    all_mlist_10_part = np.array(all_mlist_10_part)[mask]
    all_mpool_10_part = np.array(all_mpool_10_part)[mask]

    #print(len(all_probs_part))
    #print(len(all_num_top_ten_part))

    slope_concreteness, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_concreteness_part, all_probs_part)
    slope_wordfreq, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_wordfreq_part, all_probs_part)
    slope_wordlen, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_wordlen_part, all_probs_part)
    slope_mlist, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_mlist_part, all_probs_part)
    slope_mpool, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_mpool_part, all_probs_part)
    slope_mlist_ten, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_mlist_10_part, all_probs_part)
    slope_mpool_ten, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_mpool_10_part, all_probs_part)

    slopes_participants_concreteness.append(slope_concreteness)
    slopes_participants_wordfreq.append(slope_wordfreq)
    slopes_participants_wordlen.append(slope_wordlen)
    slopes_participants_mlist.append(slope_mlist)
    slopes_participants_mpool.append(slope_mpool)
    slopes_participants_mlist_ten.append(slope_mlist_ten)
    slopes_participants_mpool_ten.append(slope_mpool_ten)

#print((np.array(all_probs_final)))
#print((np.array(all_num_top_ten_final)))

print("Slopes Concreteness: ", (slopes_participants_concreteness))
print("Slopes WordFreq", (slopes_participants_wordfreq))
print("Slopes WordLen ",(slopes_participants_wordlen))
print("Slopes MList",(slopes_participants_mlist))
print("Slopes MPool",(slopes_participants_mpool))
print("Slopes MList 10",(slopes_participants_mlist_ten))
print("Slopes MPool 10",(slopes_participants_mpool_ten))

print(len(slopes_participants_concreteness))
print(len(slopes_participants_wordfreq))
print(len(slopes_participants_wordlen))
print(len(slopes_participants_mlist))
print(len(slopes_participants_mpool))
print(len(slopes_participants_mlist_ten))
print(len(slopes_participants_mpool_ten))

t_test_concreteness = scipy.stats.ttest_1samp(slopes_participants_concreteness, 0, axis=0)
t_test_wordfreq = scipy.stats.ttest_1samp(slopes_participants_wordfreq, 0, axis=0)
t_test_wordlen = scipy.stats.ttest_1samp(slopes_participants_wordlen, 0, axis=0)
t_test_mlist = scipy.stats.ttest_1samp(slopes_participants_mlist, 0, axis=0)
t_test_mpool = scipy.stats.ttest_1samp(slopes_participants_mpool, 0, axis=0)
t_test_mlist_10 = scipy.stats.ttest_1samp(slopes_participants_mlist_ten, 0, axis=0)
t_test_mpool_10 = scipy.stats.ttest_1samp(slopes_participants_mpool_ten, 0, axis=0)

print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq ",t_test_wordfreq)
print("T-test WordLen",t_test_wordlen)
print("T-test Mlist ",t_test_mlist)
print("T-test Mpool",t_test_mpool)
print("T-test Mlist 10",t_test_mlist_10)
print("T-test MPool 10",t_test_mpool_10)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = slopes_participants_concreteness
y_wordlen = slopes_participants_wordlen
y_wordfreq = slopes_participants_wordfreq
y_mlist = slopes_participants_mlist
y_mlist_ten = slopes_participants_mlist_ten
y_mpool = slopes_participants_mpool
y_mpool_ten = slopes_participants_mpool_ten

print("Y concreteness", y_concreteness)
print("Y wordlen", y_wordlen)
print("Y wordfreq", y_wordfreq)
print("Y mlist", y_mlist)
print("Y mpool", y_mpool)
print("Y mlist ten", y_mlist_ten)
print("Y mpool ten", y_mpool_ten)


# Plot
plt.scatter(x, y_concreteness, alpha=0.5)
plt.title("Slopes: Concreteness of List & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordlen, alpha=0.5)
plt.title("Slopes: Word Length of List & P-Rec")
plt.xticks([])
#plt.ylim(-.002, .002)
plt.show()

plt.scatter(x, y_wordfreq, alpha=0.5)
plt.title("Slopes: Word Freq of List & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist, alpha=0.5)
plt.title("Slopes: List Meaningfulness of List & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool, alpha=0.5)
plt.title("Slopes: Pool Meaningfulness of List & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist_ten, alpha=0.5)
plt.title("Slopes: Number of Highly Meaningful (List) Words & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool_ten, alpha=0.5)
plt.title("Slopes: Number of Highly Meaningful (Pool) Words & P-Rec")
plt.xticks([])
plt.show()

# # Create data
# N = len(all_probs)
# x = np.zeros(N)
# y = slopes_participants
#
# # Plot
# plt.scatter(x, y, alpha=0.5)
# #plt.title('Slopes')
# plt.title("Slopes: Average Concreteness In Each List & P-Rec")
# plt.xticks([])
# #plt.title("Slopes: Semantic Meaningfulness of List & P-Rec")
# plt.show()
#

"""Output
N = 75
[-0.16717436031241831, 0.046531647886353271, -0.041588617379909154, -0.042535294883974312, 0.040281815704347947, 0.0056982606584674448, -0.0044537560184716562, 0.063346754305344741, -0.11685239646488731, 0.026730159171752442, -0.071521184860880146, 0.020575393567364573, -0.015335489952009791, 0.0032678762941053476, -0.0055726012125077865, 0.10305138205647045, 0.090760868296952499, 0.034296423092815984, 0.036140351609143626, -0.052444574773681002, -0.0087810178609711628, 0.0086019914201639035, -0.056458592750775652, 0.16049597757675158, -0.001557106092250079, -0.0085299337432764889, 0.053741606307816839, -0.044475223965202662, 0.062380521333357669, 0.10220857040273856, -0.066318251601744135, -0.033117694672279721, -0.11870841920251875, 0.056448182602195862, 0.055476042461948732, 0.011763325696787003, -0.079499336868893156, 0.016686782843490032, -0.05180623345047361, 0.11799403442889589, -0.038165668498515132, 0.027157577033590403, -0.0049623366322486843, 0.042670390052449103, 0.029900785434439189, 0.14068850713556799, -0.033981938328132996, -0.067340429250012851, 0.0011295107364754152, 0.034833134211284034, -0.17099547883570296, -0.037659692477872909, 0.11531240098711001, 0.033648680052982889, 0.024517946065101087, 0.094226085446684926, -0.13938588215246397, 0.0075927455517090653, 0.097552505863289798, -0.025250540179174656, -0.030601631751878491, 0.071867412169118142, 0.057240841496663752, -0.028654656510798366, 0.03754394720557213, 0.031562314191258163, 0.04700353420231982, -0.055409330063571648, 0.028238155475785554, -0.054305949986790947, -0.09200932017573786, -0.0048887970819433708, -0.010973078821612942, 0.12054353568380881, 0.037349380331664926]
Ttest_1sampResult(statistic=0.70730429360181324, pvalue=0.48159898010802782)

"""
