"""1. WORD RECALL MODEL I
{Each of these vectors will have a dimension of 576 words}
1)	Word Frequency -
2)	Word Concreteness -
3)	Word Length -
4)	Mlist (Each word’s aver. similarity to all other words in the list) - m_list
5)	Mpool (Each word’s aver. similarity to all other words in the pool) - m_pool
6) P_rec each_word - p_rec_word

"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import stats
from statsmodels.stats.multitest import multipletests

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, MPool_byPoolSim as m_pool, \
    P_rec_ltpFR2_by_list as p_rec, semantic_similarity_bylist as m_list

# Importing all of the data
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')

p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants,\
    all_mean_valence_participants, all_mean_arousal_participants = conc_freq_len.measures_by_list(files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)

all_probs = np.array(p_rec_list)
print(all_probs)
all_concreteness= np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_valence = np.array(all_mean_valence_participants)
all_arousal = np.array(all_mean_arousal_participants)
all_m_list = np.array(m_list)
all_means_mpool = np.array(all_means_mpool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_arousal_part = []
all_m_list_part = []
all_means_mpool_part = []

correlations = []
correlations_fisher = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordfreq_part = all_wordfreq[i]
        all_wordlen_part = all_wordlen[i]
        all_valence_part = all_valence[i]
        all_arousal_part = all_arousal[i]
        all_m_list_part = all_m_list[i]
        all_means_mpool_part = all_means_mpool[i]


        # mask them since some p_recs have nan's

        mask = np.isfinite(all_probs_part)
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part = np.array(all_probs_part)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part = np.array(all_valence_part)[mask]
        all_arousal_part = np.array(all_arousal_part)[mask]
        all_m_list_part = np.array(all_m_list_part)[mask]
        all_means_mpool_part = np.array(all_means_mpool_part)[mask]

        all_probs_part_norm = stats.mstats.zscore(all_probs_part, ddof=1)
        # print(all_probs_part_norm)
        all_concreteness_part_norm = stats.mstats.zscore(all_concreteness_part, ddof=1)
        # print("Conc", all_concreteness_part_norm)
        all_wordfreq_part_norm = stats.mstats.zscore(all_wordfreq_part, ddof=1)
        all_wordlen_part_norm = stats.mstats.zscore(all_wordlen_part, ddof=1)
        all_valence_part_norm = stats.mstats.zscore(all_valence_part, ddof=1)
        all_arousal_part_norm = stats.mstats.zscore(all_arousal_part, ddof=1)
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)

        correlations_data = np.stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                      all_valence_part_norm, all_arousal_part_norm,
                                      all_m_list_part_norm, all_means_mpool_part_norm))

        correlations.append(np.corrcoef(correlations_data))
    print(np.shape(np.array(correlations)))
    return correlations

correlations = calculate_params()

#print("Raw Correlations", correlations)
print(np.array(correlations).shape)

corr_all = np.mean(correlations, axis=0)
print("Raw Meaned correlations", corr_all)
print(np.array(correlations).shape)

corr_all = np.mean(correlations, axis=0)
print("Raw Meaned correlations", corr_all)
print("Rounded", np.round(corr_all, 3))

# all_flattened_raw = []
# for i in range(len(correlations)):
#     all_flattened_raw.append(correlations[i].flatten())
# all_flattened_raw = np.array(all_flattened_raw)
# print("Shape of all flattened raw", (all_flattened_raw).shape)
#
# all_flattened_fisher = []
#
# for i in range(all_flattened_raw.shape[0]):
#     for j in range(all_flattened_raw.shape[1]):
#         if all_flattened_raw[i][j] == 1:
#             all_flattened_fisher.append(np.nan)
#         else:
#             all_flattened_fisher.append((np.arctanh(all_flattened_raw[i][j])))
#
# all_flattened_fisher = np.array(all_flattened_fisher)
# print("Shpae of all flattened fisher", (all_flattened_fisher).shape)
# fisher_reshaped = np.reshape(all_flattened_fisher, (len(all_probs), 25))
# fisher_reshaped = np.array(fisher_reshaped)
#
# #print("Fisher reshaped", fisher_reshaped)
# print("Shape of fisher reshaped", fisher_reshaped.shape)
#
# freq_conc = fisher_reshaped[:, 5]
# len_conc = fisher_reshaped[:, 10]
# len_freq = fisher_reshaped[:, 11]
# mlist_conc = fisher_reshaped[:, 15]
# mlist_freq = fisher_reshaped[:, 16]
# mlist_len = fisher_reshaped[:, 17]
# mpool_conc = fisher_reshaped[:, 20]
# mpool_freq = fisher_reshaped[:, 21]
# mpool_len = fisher_reshaped[:, 22]
# mpool_mlist = fisher_reshaped[:, 23]
#
# # print(freq_conc)
# # print(len_conc)
# # print(len_freq)
# # print(mlist_conc)
# # print(mlist_freq)
# # print(mlist_len)
# # print(mpool_conc)
# # print(mpool_freq)
# # print(mpool_len)
# # print(mpool_mlist)
#
# t_test_freq_conc = scipy.stats.ttest_1samp(freq_conc, 0)
# t_test_len_conc = scipy.stats.ttest_1samp(len_conc, 0)
# t_test_len_freq = scipy.stats.ttest_1samp(len_freq, 0)
# t_test_mlist_conc = scipy.stats.ttest_1samp(mlist_conc, 0)
# t_test_mlist_freq = scipy.stats.ttest_1samp(mlist_freq, 0)
# t_test_mlist_len = scipy.stats.ttest_1samp(mlist_len, 0)
# t_test_mpool_conc = scipy.stats.ttest_1samp(mpool_conc, 0)
# t_test_mpool_freq = scipy.stats.ttest_1samp(mpool_freq, 0)
# t_test_mpool_len = scipy.stats.ttest_1samp(mpool_len, 0)
# t_test_mpool_mlist = scipy.stats.ttest_1samp(mpool_mlist, 0)
#
# print("t_test_freq_conc", t_test_freq_conc)
# print("t_test_len_conc", t_test_len_conc)
# print("t_test_len_freq", t_test_len_freq)
# print("t_test_mlist_conc", t_test_mlist_conc)
# print("t_test_mlist_freq",t_test_mlist_freq)
# print("t_test_mlist_len",t_test_mlist_len)
# print("t_test_mpool_conc",t_test_mpool_conc)
# print("t_test_mpool_freq",t_test_mpool_freq)
# print("t_test_mpool_len",t_test_mpool_len)
# print("t_test_mpool_mlist",t_test_mpool_mlist)
#
#
# print(multipletests([(t_test_freq_conc[1]), (t_test_len_conc[1]), (t_test_len_freq[1]), (t_test_mlist_conc[1]),
#                      (t_test_mlist_freq[1]), (t_test_mlist_len[1]), (t_test_mpool_conc[1]),
#                      (t_test_mpool_freq[1]), (t_test_mpool_len[1]), (t_test_mpool_mlist[1])], alpha=0.05,
#                     method='fdr_bh', is_sorted=False, returnsorted=False))
#
# mean_freq_conc = np.mean(fisher_reshaped[:, 5])
# mean_len_conc = np.mean(fisher_reshaped[:, 10])
# mean_len_freq = np.mean(fisher_reshaped[:, 11])
# mean_mlist_conc = np.mean(fisher_reshaped[:, 15])
# mean_mlist_freq = np.mean(fisher_reshaped[:, 16])
# mean_mlist_len = np.mean(fisher_reshaped[:, 17])
# mean_mpool_conc = np.mean(fisher_reshaped[:, 20])
# mean_mpool_freq = np.mean(fisher_reshaped[:, 21])
# mean_mpool_len = np.mean(fisher_reshaped[:, 22])
# mean_mpool_mlist = np.mean(fisher_reshaped[:, 23])
#
# # print(mean_freq_conc)
# # print(mean_len_conc)
# # print(mean_len_freq)
# # print(mean_mlist_conc)
# # print(mean_mlist_freq)
# # print(mean_mlist_len)
# # print(mean_mpool_conc)
# # print(mean_mpool_freq)
# # print(mean_mpool_len)
# # print(mean_mpool_mlist)
#
# correlations_word_mean = [[mean_freq_conc, np.nan, np.nan, np.nan],
#                           [mean_len_conc, mean_len_freq, np.nan, np.nan],
#                           [mean_mlist_conc, mean_mlist_freq, mean_mlist_len, np.nan],
#                           [mean_mpool_conc, mean_mpool_freq, mean_mpool_len, mean_mpool_mlist]]
#
# fig, ax = plt.subplots()
# # Using matshow here just because it sets the ticks up nicely. imshow is faster.
# ax.matshow(correlations_word_mean, cmap='gray')
#
# # for (i, j), z in np.ndenumerate(correlations_word_mean):
# #     if np.isnan(z):
# #         continue
# #     if z in [mean_mlist_len]:
# #         ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
# #                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.0'))
# #     else:
# #         ax.text(j, i, '{:0.2f}'.format(z) + '*', ha='center', va='center',
# #                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.0'))
#
# for (i, j), z in np.ndenumerate(correlations_word_mean):
#     if np.isnan(z):
#         continue
#     else:
#         ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
#                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.0'))
#
# plt.imshow(correlations_word_mean, interpolation='none', cmap='bwr', vmax=1, vmin=-1)
# plt.colorbar()
# ax.set_xticklabels([[], 'Conc', 'Freq', 'Length', 'M List', 'M Pool'])
# ax.set_yticklabels([[], 'Freq', 'Length', 'M List', 'M Pool'])
# plt.savefig("List_Corr_N86.pdf")
# plt.show()
#
#
# # plt.scatter((np.full(len(freq_conc), 1)), freq_conc)
# # plt.scatter((np.full(len(len_conc), 2)), len_conc)
# # plt.scatter((np.full(len(len_freq), 3)), len_freq)
# # plt.scatter((np.full(len(mlist_conc), 4)), mlist_conc)
# # plt.scatter((np.full(len(mlist_freq), 5)), mlist_freq)
# # plt.scatter((np.full(len(mlist_len), 6)), mlist_len)
# # plt.scatter((np.full(len(mpool_conc), 7)), mpool_conc)
# # plt.scatter((np.full(len(mpool_freq), 8)), mpool_freq)
# # plt.scatter((np.full(len(mpool_len), 9)),mpool_len)
# # plt.scatter((np.full(len(mpool_mlist), 10)), mpool_mlist)
# # plt.axhline(y=0, color='gray', linestyle='--')
# #
# # plt.show()
"""
(88, 7, 7)
(88, 7, 7)
Raw Meaned correlations [[  1.00000000e+00  -7.41772848e-02  -3.81289525e-02   1.27502357e-01
   -2.70143570e-01   2.29162580e-01   1.98515252e-01]
 [ -7.41772848e-02   1.00000000e+00  -1.16728974e-01   1.85836215e-01
    5.84288014e-03  -1.10565441e-01  -1.16489614e-01]
 [ -3.81289525e-02  -1.16728974e-01   1.00000000e+00  -2.38908010e-02
    3.40887433e-02   7.68148809e-04   2.01571914e-02]
 [  1.27502357e-01   1.85836215e-01  -2.38908010e-02   1.00000000e+00
   -5.61683286e-01   3.59208731e-02   5.18823566e-02]
 [ -2.70143570e-01   5.84288014e-03   3.40887433e-02  -5.61683286e-01
    1.00000000e+00  -9.82025629e-02  -1.00391481e-01]
 [  2.29162580e-01  -1.10565441e-01   7.68148809e-04   3.59208731e-02
   -9.82025629e-02   1.00000000e+00   6.11958951e-01]
 [  1.98515252e-01  -1.16489614e-01   2.01571914e-02   5.18823566e-02
   -1.00391481e-01   6.11958951e-01   1.00000000e+00]]
(88, 7, 7)
Raw Meaned correlations [[  1.00000000e+00  -7.41772848e-02  -3.81289525e-02   1.27502357e-01
   -2.70143570e-01   2.29162580e-01   1.98515252e-01]
 [ -7.41772848e-02   1.00000000e+00  -1.16728974e-01   1.85836215e-01
    5.84288014e-03  -1.10565441e-01  -1.16489614e-01]
 [ -3.81289525e-02  -1.16728974e-01   1.00000000e+00  -2.38908010e-02
    3.40887433e-02   7.68148809e-04   2.01571914e-02]
 [  1.27502357e-01   1.85836215e-01  -2.38908010e-02   1.00000000e+00
   -5.61683286e-01   3.59208731e-02   5.18823566e-02]
 [ -2.70143570e-01   5.84288014e-03   3.40887433e-02  -5.61683286e-01
    1.00000000e+00  -9.82025629e-02  -1.00391481e-01]
 [  2.29162580e-01  -1.10565441e-01   7.68148809e-04   3.59208731e-02
   -9.82025629e-02   1.00000000e+00   6.11958951e-01]
 [  1.98515252e-01  -1.16489614e-01   2.01571914e-02   5.18823566e-02
   -1.00391481e-01   6.11958951e-01   1.00000000e+00]]
Rounded [[ 1.    -0.074 -0.038  0.128 -0.27   0.229  0.199]
 [-0.074  1.    -0.117  0.186  0.006 -0.111 -0.116]
 [-0.038 -0.117  1.    -0.024  0.034  0.001  0.02 ]
 [ 0.128  0.186 -0.024  1.    -0.562  0.036  0.052]
 [-0.27   0.006  0.034 -0.562  1.    -0.098 -0.1  ]
 [ 0.229 -0.111  0.001  0.036 -0.098  1.     0.612]
 [ 0.199 -0.116  0.02   0.052 -0.1    0.612  1.   ]]


"""