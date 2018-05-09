"""2.	LIST RECALL MODEL II
{Each of these vectors will have a dimension of N participants * 552 lists}
1)	Average Word Frequency Of The List - all_mean_wordfreq_participants
2)	Average Word Concreteness Of The List - all_mean_conc_participants
3)	Average Word Length Of The List - all_mean_wordlen_participants
4)	Mlist (Each word’s aver. similarity to all other words in the list) 10th percentile - m_list_ten
5)	Mpool (Each word’s aver. similarity to all other words in the pool) 10th percentile - m_pool_ten
6) P_rec per list

"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm

from w2vec import Mlist10th_percentile as m_list
from w2vec import Mpool10th_percentile as m_pool
from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, P_rec_ltpFR2_by_list as p_rec

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(files_ltpFR2)
larger_90 = [33, 35, 100, 109, 134, 198, 210, 336, 397, 403, 407, 429, 579, 584, 592, 602, 624, 684, 713, 754, 755, 778,
             785, 834, 876, 929, 953, 977, 1016, 1022, 1031, 1071, 1079, 1112, 1139, 1140, 1144, 1147, 1152, 1158, 1209,
             1224, 1227, 1238, 1246, 1292, 1332, 1333, 1357, 1385, 1388, 1423, 1472, 1520, 1522, 1524, 1581, 1620]
m_list = m_list.get_m_list(larger_90, files_ltpFR2)
all_means_mpool = m_pool.get_m_pool(files_ltpFR2)


all_probs = np.array(p_rec_list)
all_concreteness= np.array(all_mean_conc_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_m_list = np.array(m_list)
all_means_mpool = np.array(all_means_mpool)

all_probs_part = []
all_concreteness_part = []
all_wordlen_part = []
all_wordfreq_part = []
all_m_list_part = []
all_means_mpool_part = []

# print("P_rec_list", p_rec_list)
# print("Shape of P_rec_list", np.shape(p_rec_list))
# print(" ")
# print("Aver conc per list", all_mean_conc_participants)
# print("Shape of aver conc per list", np.shape(all_mean_conc_participants))
# print("  ")
# print("Aver freq per list", all_mean_wordfreq_participants)
# print("Shape of aver conc per list", np.shape(all_mean_wordfreq_participants))
# print("  ")
# print("Aver word len per list", all_mean_wordlen_participants)
# print("Shape of aver word len per list", np.shape(all_mean_wordlen_participants))
# print("  ")
# print("M list per list", m_list)
# print("Shape of m list", np.shape(m_list))
# print("  ")
# print("M pool per list", m_pool)
# print("Shape of m pool", np.shape(m_pool))

params = []
rsquareds = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordlen_part = all_wordlen[i]
        all_wordfreq_part = all_wordfreq[i]
        all_m_list_part = all_m_list[i]
        all_means_mpool_part = all_means_mpool[i]

        # mask them all

        mask = np.isfinite(all_probs_part)
        all_probs_part = np.array(all_probs_part)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_m_list_part = np.array(all_m_list_part)[mask]
        all_means_mpool_part = np.array(all_means_mpool_part)[mask]

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_m_list_part, all_means_mpool_part))
        # x-values without the m_list
        # x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        # print(x_values)
        # print(np.shape(x_values))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        y_value = all_probs_part

        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(np.array(results))
        print(results.summary())
        print(results.rsquared)

        params.append(results.params)

        rsquareds.append(results.rsquared)
    return params, rsquareds

params, rsquareds = calculate_params()
print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff, MPool 10,  MList 10%)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)

params = np.array(params)

t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mlist_10 = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool_10 = scipy.stats.ttest_1samp(params[:,5], 0)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordlen = params[:,2]
y_wordfreq = params[:,3]
y_mlist_ten = params[:,4]
y_mpool_ten = params[:,5]

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordLen", t_test_wordlen)
print("T-test WordFreq", t_test_wordfreq)
print("T-test MList", t_test_mlist_10)
print("T-test MPool", t_test_mpool_10)


print("Y concreteness", y_concreteness)
print("Y wordlen", y_wordlen)
print("Y wordfreq", y_wordfreq)
print("Y mlist ten", y_mlist_ten)
print("Y mpool ten", y_mpool_ten)



# Plot
plt.scatter(x, y_concreteness, alpha=0.5)
plt.title("Slopes/Beta Values Concreteness (List II)")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordlen, alpha=0.5)
plt.title("Slopes/Beta Values Word Length (List II)")
plt.xticks([])
#plt.ylim(-.002, .002)
plt.show()

plt.scatter(x, y_wordfreq, alpha=0.5)
plt.title("Slopes/Beta Values Word Freq (List II)")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist_ten, alpha=0.5)
plt.title("Slopes/Beta Values List Meaningfulness - 10th percentile (List II)")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool_ten, alpha=0.5)
plt.title("Slopes/Beta Values Pool Meaningfulness - 10th percentile (List II)")
plt.xticks([])
plt.show()

plt.scatter(x, rsquareds, alpha=0.5)
plt.title("R-Squares - List Recall Model II")
plt.xticks([])
plt.show()
"""T-test Intercepts Ttest_1sampResult(statistic=10.76203927039867, pvalue=8.5575698404773119e-17)
T-test Concreteness Ttest_1sampResult(statistic=0.22693314581682189, pvalue=0.82110147344330509)
T-test WordLen Ttest_1sampResult(statistic=-0.36546285471726608, pvalue=0.71580850138152241)
T-test WordFreq Ttest_1sampResult(statistic=6.8841704983391754, pvalue=1.6149204241897189e-09)
T-test MList Ttest_1sampResult(statistic=1.0341084260658258, pvalue=0.30445320965609585)
T-test MPool Ttest_1sampResult(statistic=3.102258891461505, pvalue=0.0027185474134368338)"""