"""1. WORD RECALL MODEL II
{Each of these vectors will have a dimension of 576 words}
1)	Word Frequency -
2)	Word Concreteness -
3)	Word Length -
4)	Mlist 10th percentile (Each word’s aver. similarity to all other words in the list) - m_list
5)	Mpool 10th percentile (Each word’s aver. similarity to all other words in the pool) - m_pool
6) P_rec each_word - p_rec_word

"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm

from w2vec import Mlist_and_Mpool_byword_10thpercentile as m_pool
from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, P_rec_ltpFR2_words as p_rec_word

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness = conc_freq_len.concreteness
word_freq = conc_freq_len.word_freq
word_length = conc_freq_len.word_length
m_list = m_pool.if_m_list_is_larger
m_pool = m_pool.is_it_larger_90_all_wordpool

# print("P_rec_word", p_rec_word)
# print("Length of P_rec_word", np.shape(p_rec_word))
# print(" ")
# print("Concreteness per word", concreteness)
# print("Length of conc per word", len(concreteness))
# print("  ")
# print("Word freq per word", word_freq)
# print("Length of word freq per word",len(word_freq))
# print("  ")
# print("Word length", word_length)
# print("Length of word len", len(word_length))
# print("  ")
# print("M list per word", m_list)
# print("Length of m list", len(m_list))
# print("  ")
# print("M pool per word", m_pool)
# print("Length of m pool", np.shape(m_pool))

all_probs = np.array(p_rec_word)
all_concreteness= np.array(concreteness)
all_wordlen = np.array(word_length)
all_wordfreq = np.array(word_freq)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordlen_part = []
all_wordfreq_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness
        all_wordlen_part = all_wordlen
        all_wordfreq_part = all_wordfreq
        all_m_list_part = all_m_list
        all_means_mpool_part = all_means_mpool


        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_m_list_part, all_means_mpool_part))

        # x-values without the m_list
        #x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        #print(x_values)
        #print(np.shape(x_values))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(np.array(results))
        print(results.summary())
        print(results.rsquared)
        params.append(results.params)
        rsquareds.append(results.rsquared)
    return params, rsquareds


params, rsquareds = calculate_params()
print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff, MList coeff, MPool coeff)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)



t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,3], 0)
#t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mlist_ten = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool_ten = scipy.stats.ttest_1samp(params[:,5], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordLen", t_test_wordlen)
print("T-test WordFreq", t_test_wordfreq)
print("T-test MList", t_test_mlist_ten)
print("T-test MPool", t_test_mpool_ten)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordlen = params[:,2]
y_wordfreq = params[:,3]
y_mlist_ten = params[:,4]
y_mpool_ten = params[:,5]

print("Y concreteness", y_concreteness)
print("Y wordlen", y_wordlen)
print("Y wordfreq", y_wordfreq)
print("Y mlist ten", y_mlist_ten)
print("Y mpool ten", y_mpool_ten)



# Plot
plt.scatter(x, y_concreteness, alpha=0.5)
plt.title("Slopes/Beta Values Concreteness")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordlen, alpha=0.5)
plt.title("Slopes/Beta Values Word Length")
plt.xticks([])
#plt.ylim(-.002, .002)
plt.show()

plt.scatter(x, y_wordfreq, alpha=0.5)
plt.title("Slopes/Beta Values Word Freq")
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist_ten, alpha=0.5)
plt.title("Slopes/Beta Values List Meaningfulness - 10th percentile")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool_ten, alpha=0.5)
plt.title("Slopes/Beta Values Pool Meaningfulness - 10th percentile")
plt.xticks([])
plt.show()

plt.scatter(x, rsquareds, alpha=0.5)
plt.title("R-Squares - Word Recall Model II")
plt.xticks([])
plt.show()

"""R-squareds [0.02478260933023968, 0.0044119628162200275, 0.012868254733254147, 0.039003389159593893, 0.018841021839008487, 0.0067999605158677312, 0.017926499091793779, 0.026054963350937155, 0.043442027378284731, 0.019944416965810063, 0.018955123854434208, 0.011480383392439508, 0.027568455365013067, 0.050486759571783613, 0.0089690067036456433, 0.028284365440651271, 0.011431625840545689, 0.016016289512555182, 0.013181838653118505, 0.02824565973007831, 0.051465830337123819, 0.013570084007277994, 0.022824074488172141, 0.0066913625899940321, 0.022489278075915942, 0.021363364209946445, 0.015276865976228571, 0.018181906854531027, 0.021413828906272414, 0.020377415177501024, 0.021058185989662581, 0.02959452490311465, 0.01240235420284086, 0.032910144033440947, 0.030546074951066426, 0.022219075026508883, 0.018947642164918066, 0.040963852794389299, 0.008757517838170803, 0.028696793473030113, 0.0059617687395018271, 0.0092452138168032105, 0.020528002640199383, 0.00532726866615163, 0.025879299390677013, 0.013860879566605644, 0.018517673954153158, 0.0023933036492127524, 0.0078684326747155975, 0.014824887450306745, 0.0091746288966978451, 0.038889800179246592, 0.013717332125134041, 0.036670945087714291, 0.015597244026775536, 0.006951744637735624, 0.0043785499167297015, 0.0020706424227440179, 0.016369096515232884, 0.01403024761174787, 0.027116232796861373, 0.016350319439512817, 0.012844009944266932, 0.021273327910049877, 0.0047632637455824467, 0.0047505588899973139, 0.026205914709600431, 0.013434773438491754, 0.028890954981552919, 0.01855100541485355, 0.0061762022039386633, 0.017980394640016883, 0.0094799372788193503, 0.027986814779632052, 0.0071822100323100102]
(75,)
T-test Intercepts Ttest_1sampResult(statistic=22.676497731143538, pvalue=4.8165737821985237e-35)
T-test Concreteness Ttest_1sampResult(statistic=1.8863978196392335, pvalue=0.063164873358237505)
T-test WordLen Ttest_1sampResult(statistic=-1.0854545327738245, pvalue=0.2812438671709841)
T-test WordFreq Ttest_1sampResult(statistic=8.7452756554518647, pvalue=5.0388321720436588e-13)
T-test MList Ttest_1sampResult(statistic=4.1444015041943496, pvalue=8.9622756007539818e-05)
T-test MPool Ttest_1sampResult(statistic=3.3119695721184326, pvalue=0.0014348743950703049)"""