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

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')

#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP36[0-9].json')

from scipy.stats.stats import pearsonr



concreteness_norm = conc_freq_len.concreteness_norm

word_freq_norm = conc_freq_len.word_freq_norm

word_length_norm = conc_freq_len.word_length_norm

m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_ltpFR2

valence = conc_freq_len.valences_norm
arousal = conc_freq_len.arousals_norm

all_probs = np.array(p_rec_word)
all_concreteness= np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence)
all_arousal = np.array(arousal)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_arousal_part = []
all_imag_part = []
all_m_list_part = []
all_means_mpool_part = []

correlations = []

def calculate_params():
    for i in range(len(m_list)):

        # for each participant (= each list in list of lists)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_valence_part = all_valence
        all_arousal_part = all_arousal
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool, axis=0, ddof=1)

        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part = np.array(all_valence_part)[mask]
        all_arousal_part = np.array(all_arousal_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        correlations_data = np.stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,
                                      all_valence_part, all_arousal_part,
                                      all_m_list_part_norm, all_means_mpool_part_norm))
        # print("Correlations Conc Freq", i,pearsonr(all_concreteness_part, all_wordfreq_part))
        # print("Correlations Conc Len", i, pearsonr(all_concreteness_part, all_wordlen_part))
        # print("Correlations Freq Len", i, pearsonr(all_wordfreq_part, all_wordlen_part))
        # print("Correlations Freq MPool", i, pearsonr(all_wordfreq_part, all_means_mpool_part_norm))
        # print("Correlations Len Mpool", i, pearsonr(all_wordlen_part, all_means_mpool_part_norm))
        # print(np.shape(np.array(correlations)))

        correlations.append(np.corrcoef(correlations_data))
    print(np.shape(np.array(correlations)))
    return correlations


correlations = calculate_params()

# print("Raw Correlations", correlations)
print(np.array(correlations).shape)

corr_all = np.mean(correlations, axis=0)
print("Raw Meaned correlations", corr_all)
print("Rounded", np.round(corr_all, 3))

"""
(88, 7, 7)
(88, 7, 7)
Raw Meaned correlations [[ 1.         -0.04922568 -0.03093967  0.13821764 -0.2427625   0.14126276
   0.47027421]
 [-0.04922568  1.         -0.12375809  0.16649041  0.00278592 -0.05402934
  -0.14924575]
 [-0.03093967 -0.12375809  1.         -0.02194163  0.03528778 -0.00541274
   0.00829743]
 [ 0.13821764  0.16649041 -0.02194163  1.         -0.54774874  0.02271095
   0.07619538]
 [-0.2427625   0.00278592  0.03528778 -0.54774874  1.         -0.05347979
  -0.18511188]
 [ 0.14126276 -0.05402934 -0.00541274  0.02271095 -0.05347979  1.
   0.2992176 ]
 [ 0.47027421 -0.14924575  0.00829743  0.07619538 -0.18511188  0.2992176
   1.        ]]
Rounded [[ 1.    -0.049 -0.031  0.138 -0.243  0.141  0.47 ]
 [-0.049  1.    -0.124  0.166  0.003 -0.054 -0.149]
 [-0.031 -0.124  1.    -0.022  0.035 -0.005  0.008]
 [ 0.138  0.166 -0.022  1.    -0.548  0.023  0.076]
 [-0.243  0.003  0.035 -0.548  1.    -0.053 -0.185]
 [ 0.141 -0.054 -0.005  0.023 -0.053  1.     0.299]
 [ 0.47  -0.149  0.008  0.076 -0.185  0.299  1.   ]]

"""