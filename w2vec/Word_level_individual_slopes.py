import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec import Mlist_and_Mpool_byword_10thpercentile as percentile
from w2vec.Final import Concreteness_Freq_WordLength as conc, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as prbl, semanticsim_each_word as m_list

# This is the code to find a slope for each participant with their probability of list recall and average concreteness value per word

nan = np.nan

# List of lists with all participants; 576 precs per participant

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_participants_all_recalled = prbl.get_ltpFR2_precall(files_ltpFR2)
all_probs = np.array(all_participants_all_recalled)


all_concreteness = conc.concreteness
all_concreteness = np.array(all_concreteness)

all_wordlen = conc.word_length
all_wordlen = np.array(all_wordlen)

all_wordfreq = conc.word_freq
all_wordfreq = np.array(all_wordfreq)

m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_list = np.array(m_list)

m_pool = m_pool.w2v_ltpFR2
m_pool = np.array(m_pool)

m_list_ten = percentile.if_m_list_is_larger
m_list_ten = np.array(m_list_ten)

m_pool_ten = percentile.is_it_larger_90_all_wordpool
m_pool_ten = np.array(m_pool_ten)

print(np.shape(all_probs))
print(np.shape(all_concreteness))
print(np.shape(all_wordfreq))
print(np.shape(all_wordlen))
print(np.shape(m_list))
print(np.shape(m_pool))
print(np.shape(m_list_ten))
print(np.shape(m_pool_ten))

all_probs_part = []
# all_concreteness_part = []
# all_wordlen_part = []
# all_wordfreq_part = []
all_m_list_part = []
# all_means_mpool_part = []
# all_m_list_ten_part = []
# all_m_pool_ten_part = []

slopes_participants_concreteness = []
slopes_participants_wordfreq = []
slopes_participants_wordlen = []
slopes_participants_mlist = []
slopes_participants_mpool = []
slopes_participants_mlist_ten = []
slopes_participants_mpool_ten = []

# For every participant, get their lists' values (576), find the slope, add slopes into a new list
# Conduct a t-test to the slopes to see whether they're different from zero

for i in range(len(all_probs)):
    all_probs_part = all_probs[i]
    all_m_list_part = m_list[i]

    slope_concreteness, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_concreteness, all_probs_part)
    slopes_participants_concreteness.append(slope_concreteness)

    slope_wordfreq, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_wordfreq, all_probs_part)
    slopes_participants_wordfreq.append(slope_wordfreq)

    slope_wordlen, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_wordlen, all_probs_part)
    slopes_participants_wordlen.append(slope_wordlen)

    slope_mlist, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_m_list_part, all_probs_part)
    slopes_participants_mlist.append(slope_mlist)

    slope_mpool, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(m_pool, all_probs_part)
    slopes_participants_mpool.append(slope_mpool)

    slope_mlist_ten, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(m_list_ten, all_probs_part)
    slopes_participants_mlist_ten.append(slope_mlist_ten)

    slope_mpool_ten, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(m_pool_ten, all_probs_part)
    slopes_participants_mpool_ten.append(slope_mpool_ten)

print(len(slopes_participants_concreteness))
print(len(slopes_participants_wordfreq))
print(len(slopes_participants_wordlen))
print(len(slopes_participants_mlist))
print(len(slopes_participants_mpool))
print(len(slopes_participants_mlist_ten))
print(len(slopes_participants_mpool_ten))


t_test_concreteness = scipy.stats.ttest_1samp(slopes_participants_concreteness, 0, axis=0)
print("T-test concreteness", t_test_concreteness)

t_test_wordfreq = scipy.stats.ttest_1samp(slopes_participants_wordfreq, 0, axis=0)
print("T-test wordfreq", t_test_wordfreq)

t_test_wordlen = scipy.stats.ttest_1samp(slopes_participants_wordlen, 0, axis=0)
print("T-test wordlen", t_test_wordlen)

t_test_mlist = scipy.stats.ttest_1samp(slopes_participants_mlist, 0, axis=0)
print("T-test mlist", t_test_mlist)

t_test_mpool = scipy.stats.ttest_1samp(slopes_participants_mpool, 0, axis=0)
print("T-test mpool", t_test_mpool)

t_test_mlist_ten = scipy.stats.ttest_1samp(slopes_participants_mlist_ten, 0, axis=0)
print("T-test mlist ten", t_test_mlist_ten)

t_test_mpool_ten = scipy.stats.ttest_1samp(slopes_participants_mpool_ten, 0, axis=0)
print("T-test mpool ten", t_test_mpool_ten)


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
plt.title("Slopes: Concreteness of Each Word & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordlen, alpha=0.5)
plt.title("Slopes: Word Length of Each Word & P-Rec")
plt.xticks([])
#plt.ylim(-.002, .002)
plt.show()

plt.scatter(x, y_wordfreq, alpha=0.5)
plt.title("Slopes: Word Freq of Each Word & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist, alpha=0.5)
plt.title("Slopes: List Meaningfulness Each Word & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool, alpha=0.5)
plt.title("Slopes: Pool Meaningfulness Each Word & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist_ten, alpha=0.5)
plt.title("Slopes: Highly Meaningful (List) Words & P-Rec")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool_ten, alpha=0.5)
plt.title("Slopes: Highly Meaningful (Pool) Words & P-Rec")
plt.xticks([])
plt.show()
"""Output
T-test concreteness Ttest_1sampResult(statistic=2.5631151225633744, pvalue=0.012404981362333027)
T-test wordfreq Ttest_1sampResult(statistic=8.448512635264823, pvalue=1.8355972561840172e-12)
T-test wordlen Ttest_1sampResult(statistic=-2.3121795551269315, pvalue=0.023552010929930794)
T-test mlist Ttest_1sampResult(statistic=6.1691969813654772, pvalue=3.3258639359679214e-08)
T-test mpool Ttest_1sampResult(statistic=9.2688592628156616, pvalue=5.1757927279318354e-14)
T-test mlist ten Ttest_1sampResult(statistic=5.0120770050514656, pvalue=3.5636677059504573e-06)
T-test mpool ten Ttest_1sampResult(statistic=4.8458287392161763, pvalue=6.7582667958962626e-06)
"""
