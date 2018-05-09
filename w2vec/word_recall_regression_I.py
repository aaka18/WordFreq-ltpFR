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
import statsmodels.api as sm

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP22*.mat')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness = conc_freq_len.concreteness
word_freq = conc_freq_len.word_freq
word_length = conc_freq_len.word_length
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_ltpFR2

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
        all_m_list_part = all_m_list[i]
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
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)



t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,3], 0)
#t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,5], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordLen", t_test_wordlen)
print("T-test WordFreq", t_test_wordfreq)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)




# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordlen = params[:,2]
y_wordfreq = params[:,3]
y_mlist = params[:,4]
y_mpool = params[:,5]

print("Y concreteness", y_concreteness)
print("Y wordlen", y_wordlen)
print("Y wordfreq", y_wordfreq)
print("Y mlist", y_mlist)
print("Y mpool", y_mpool)


plt.scatter(x, rsquareds, alpha=0.5)
plt.title("R-Squares")
plt.xticks([])
# plt.show()

x_conc = np.full(x, 1)
x_wordfreq = np.full(x, 2)
x_wordlen = np.full(x,3)
x_mlist = np.full(x,4)
x_mpool = np.full(x,5)

plt.subplot(5,1)
plt.scatter(x_conc, y_concreteness, alpha=0.5)
plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
plt.scatter(x_mlist, y_mlist, alpha=0.5)
plt.scatter(x_mpool, y_mpool, alpha=0.5)
plt.plot()


"""
R-squareds [0.027573914997941928, 0.011217174131859853, 0.043787856150648619, 0.04585325597390566, 0.038213067008331048, 0.026139415305337144, 0.034494220133668096, 0.018234671339509734, 0.04315802331839691, 0.023482514185595549, 0.046837804467065425, 0.032028776248215873, 0.035061043396978797, 0.060674945451241524, 0.032450654973536497, 0.036875514158764888, 0.021734062831829548, 0.015045129146869396, 0.031050188785978206, 0.020422245931998706, 0.055881237497863778, 0.013146495892051302, 0.031149209241792541, 0.010044246282574942, 0.028981932283689948, 0.020331285295280277, 0.019445453612534358, 0.02559274277327872, 0.029802034282307988, 0.018087430015451433, 0.025653570906770451, 0.034283731548634289, 0.013312519732869732, 0.063957045996571482, 0.037794595949986687, 0.025131757597177229, 0.012737629417159058, 0.045387718413852141, 0.031603411443675955, 0.045812583017017405, 0.0019904630394481648, 0.022640919949006411, 0.020350781107670746, 0.021638523830864953, 0.028676206478435717, 0.026559082191639494, 0.040882503320406882, 0.015721387733519387, 0.006961787901363703, 0.014951544578495013, 0.014579374743673101, 0.034795727159953382, 0.030593989009702338, 0.066657953988266017, 0.025339433907204834, 0.040716269679664463, 0.010351534762076731, 0.011088028330285549, 0.057065462328552141, 0.026508959876307414, 0.021650365945402483, 0.017842156861939462, 0.042880859267661386, 0.030361237657022477, 0.008520071983183275, 0.028026923263188563, 0.031147906392345326, 0.012940195973714652, 0.040888670856471965, 0.029107758594536515, 0.0052049215768801282, 0.02395540716249478, 0.010040754021312881, 0.028848385067301363, 0.036829590448498983]
(75,)
T-test Intercepts Ttest_1sampResult(statistic=18.771781132334606, pvalue=7.3309056986288899e-30)
T-test Concreteness Ttest_1sampResult(statistic=-3.689871651055582, pvalue=0.0004259430638863196)
T-test WordLen Ttest_1sampResult(statistic=-0.91202664841756542, pvalue=0.36471632927820719)
T-test WordFreq Ttest_1sampResult(statistic=10.302209035073123, pvalue=6.0426207072567719e-16)
T-test MList Ttest_1sampResult(statistic=2.8653736449233844, pvalue=0.0054189401675425776)
T-test MPool Ttest_1sampResult(statistic=11.673826902792477, pvalue=1.8844026664714233e-18)
"""