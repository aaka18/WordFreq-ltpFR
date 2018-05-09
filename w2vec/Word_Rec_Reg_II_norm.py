"""1. WORD RECALL MODEL II
{Each of these vectors will have a dimension of 576 words}
1)	Word Frequency -
2)	Word Concreteness -
3)	Word Length -
4)	Mlist 10th percentile (Each word’s aver. similarity to all other words in the list) - m_list
5)	Mpool 10th percentile (Each word’s aver. similarity to all other words in the pool) - m_pool
6) P_rec each_word - p_rec_word

"""
import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from astropy.table import Table,Column

#from w2vec import P_rec_ltpFR2_words as p_rec_word
#from w2vec import Concreteness_Freq_WordLength as conc_freq_len
#from w2vec import Mlist_and_Mpool_byword_10thpercentile as m_pool

import P_rec_ltpFR2_words as p_rec_word
import Concreteness_Freq_WordLength as conc_freq_len
import Mlist_and_Mpool_byword_10thpercentile as m_pool

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP22*.mat')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_list = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m_pool = m_pool.yes_or_no_mpool

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
all_concreteness= np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []
predict = []
pvalues = []
residuals = []
f_pvalues = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_m_list_part = all_m_list
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)


        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,  all_m_list_part_norm, all_means_mpool_part_norm))

        # x-values without the m_list
        #x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        #print(x_values)
        #print(np.shape(x_values))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        f_pvalues.append(results.f_pvalue)


    return params, rsquareds, predict, pvalues

params, rsquareds, predict, pvalues = calculate_params()
print("Parameters of Word Recall Regression Model II (X-intercept, Conc coeff, WordFreq coeff, WordLen coeff, MList coeff, MPool coeff)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)


rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if f_pvalues[i] < .05:
        rsquareds_sig.append(rsquareds[i])
    else:
        rsquareds_notsig.append(rsquareds[i])
print(rsquareds_sig)
print(rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

plt.title("R-Squared Values of Word Recall Model II")
plt.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' ,label = " p < 0.05" )
plt.scatter(x_rsq_nonsig, rsquareds_notsig,  facecolors='none', edgecolors='gray', label = "Not Significant" )

plt.xticks([])
plt.legend()



t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,3], 0)
#t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mlist_ten = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool_ten = scipy.stats.ttest_1samp(params[:,5], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist_ten)
print("T-test MPool", t_test_mpool_ten)

N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordfreq = params[:,2]
y_wordlen = params[:,3]
y_mlist_10 = params[:,4]
y_mpool_10 = params[:,5]

print("Y concreteness", y_concreteness)
print("Y wordfreq", y_wordfreq)
print("Y wordlen", y_wordlen)
print("Y mlist ten", y_mlist_10)
print("Y mpool ten", y_mpool_10)

beta_concreteness = (np.array(params[:,1]))
beta_wordfreq = (np.array(params[:,2]))
beta_wordlen =(np.array(params[:,3]))
beta_mlist_10 = (np.array(params[:,4]))
beta_mpool_10 = (np.array(params[:,5]))


print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist_10), stats.sem(beta_mlist_10))
print("Beta MPool Ave, SE", np.mean(beta_mpool_10), stats.sem(beta_mpool_10))

mean_conc = np.mean(beta_concreteness)
mean_wordfreq = np.mean(beta_wordfreq)
mean_wordlen = np.mean(beta_wordlen)
mean_mlist = np.mean(beta_mlist_10)
mean_mpool = np.mean(beta_mpool_10)


print("P pvalues", pvalues)
print("Params", params)

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3), round(np.mean(beta_mlist_10),3), round(np.mean(beta_mpool_10),3)]
sem_betas = [round(stats.sem(beta_concreteness),3), round(stats.sem(beta_wordfreq),3),round(stats.sem(beta_wordlen),3),  round(stats.sem(beta_mlist_10),3),round(stats.sem(beta_mpool_10),3) ]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',  'List Meaningfulness 10%', 'Pool Meaningfulness 10%']
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names = ("Word Recall Model II", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))
#print('\033[1m' + "Table 1")
#print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")


betas_sig_conc = []
betas_nonsig_conc = []
for i in range(len(params[:,1])):
    if pvalues[:,1][i] < .05:
        betas_sig_conc.append(params[:,1][i])
    else:
        betas_nonsig_conc.append(params[:,1][i])


betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:,2])):
    if pvalues[:,2][i] < .05:
        betas_sig_wordfreq.append(params[:,2][i])
    else:
        betas_nonsig_wordfreq.append(params[:,2][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:,3])):
    if pvalues[:,3][i] < .05:
        betas_sig_wordlen.append(params[:,3][i])
    else:
        betas_nonsig_wordlen.append(params[:,3][i])




betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:,4])):
    if pvalues[:,4][i] < .05:
        betas_sig_mlist.append(params[:,4][i])
    else:
        betas_nonsig_mlist.append(params[:,4][i])


betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:,5])):
    if pvalues[:,5][i] < .05:
        betas_sig_mpool.append(params[:,5][i])
    else:
        betas_nonsig_mpool.append(params[:,5][i])


x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_mlist_sig = np.full(len(betas_sig_mlist),4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist),4)

x_mpool_sig = np.full(len(betas_sig_mpool),5)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool),5)


fig, ax = plt.subplots()
fig.canvas.draw()

plt.title("Beta Values of Word Recall Model II")
plt.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
plt.scatter(x_conc_nonsig, betas_nonsig_conc,  facecolors='none', edgecolors='gray')
plt.scatter(1, mean_conc, s=80, marker=(5, 2))

plt.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black')
plt.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq,  facecolors='none', edgecolors='gray')
plt.scatter(2, mean_wordfreq, s=80, marker=(5, 2))

plt.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black')
plt.scatter(x_wordlen_nonsig, betas_nonsig_wordlen,  facecolors='none', edgecolors='gray')
plt.scatter(3, mean_wordlen, s=80, marker=(5, 2))

plt.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color = 'black' )
plt.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
plt.scatter(4, mean_mlist, s=80, marker=(5, 2))

plt.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black')
plt.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
plt.scatter(5, mean_mpool, s=80, marker=(5, 2))

# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Concreteness'
labels[3] = 'Word Freq'
labels[5] = 'Word Length'
labels[7] = 'M List 10%'
labels[9] = 'M Pool 10%'

ax.set_xticklabels(labels)
ax.xaxis.set_ticks_position('none')


print("Rsq", rsquareds.tolist())


plt.axhline(y=0, color='gray', linestyle='--')
plt.show()

"""
N = 76

T-test Intercepts Ttest_1sampResult(statistic=-0.24252938669733309, pvalue=0.80903199928556435)
T-test Concreteness Ttest_1sampResult(statistic=1.7245694341170579, pvalue=0.088724346694877262)
T-test WordFreq Ttest_1sampResult(statistic=9.3694941873148725, pvalue=2.9605783803004205e-14)
T-test WordLen Ttest_1sampResult(statistic=-0.9828713667956972, pvalue=0.32883083407095248)
T-test MList Ttest_1sampResult(statistic=3.5200953507403461, pvalue=0.00073735142928122452)
T-test MPool Ttest_1sampResult(statistic=3.8276954073008369, pvalue=0.00026606731805647215)
R-squareds [0.024842361850816341, 0.0045225783865148195, 0.012952429615860739, 0.03910339472345381, 0.018846083585312146, 0.0066285893228189563, 0.017826197829923007, 0.024523249416468995, 0.043641713254783943, 0.020042588237329295, 0.018251386958912486, 0.011242974947804241, 0.027776278158672008, 0.049211115734778654, 0.0091443812565312754, 0.02950502146283096, 0.011139473078724982, 0.016346160539989163, 0.013693435747685001, 0.027482352347762085, 0.051790224678811336, 0.012597448580973336, 0.02317145860327241, 0.0070396871126070515, 0.020459975086000504, 0.021402656603329739, 0.015668421305177649, 0.017525878084119428, 0.021937516800389023, 0.019520537786288927, 0.020650820500893063, 0.029963574031357743, 0.012307162073015521, 0.032754595112795992, 0.03054078479294331, 0.022431304933537954, 0.018789966379183731, 0.041020979998695029, 0.0082554987599577911, 0.029028466889350124, 0.0059293691103647905, 0.0086008998293778394, 0.020252995096098791, 0.0049804028166827718, 0.024963318775572119, 0.013276290003685132, 0.01847525198339417, 0.0023913518658509325, 0.0081653752542781843, 0.016174181477784266, 0.0093648782205968395, 0.037521940331983639, 0.013885929219433435, 0.036346902575430606, 0.016017165426229329, 0.0069616524429871873, 0.0034892044823703339, 0.0023142243967160248, 0.01843286046857473, 0.013101707116167094, 0.026418008849358299, 0.016343677598151585, 0.012833572575666197, 0.021288918032139725, 0.0047145314520709025, 0.0054551742575312812, 0.02499333400133108, 0.014345937017255506, 0.029202668214636907, 0.017315655316768819, 0.0069051256962280316, 0.016295241063304933, 0.0095727395328466525, 0.028124821677471545, 0.043989548166700643, 0.0070356034651337662]
(76,)
[0.013386043210070569, 0.00036926324334599005, 0.01444868001952143, 0.0001114573046377648, 0.041169566018141888, 0.0065680438879509124, 2.4932382124566941e-05, 0.0042854720092315906, 0.007058897140646692, 1.2348445133141928e-05, 0.019919518187815226, 0.037426268928615468, 0.030122777994209971, 0.026603714506974557, 0.046349560831430672, 0.035824091332493993, 0.0038233348357473418, 0.0018955309771093783, 0.0033101856683205233, 0.023705576768860995, 0.00022318886954384939, 0.0048232530596936983, 0.039240420274419707, 0.013003266839290761, 0.00055759701249521575, 0.00075591802736503326, 0.0091515973233949211, 0.030926166577212977, 0.012909927113124869, 0.0046194657023464133, 0.0060288026213293506, 0.00010159183402496534]
[0.76280717821099597, 0.18922476160163015, 0.053951852139178844, 0.57843239235269084, 0.067692149940554974, 0.061608940402113989, 0.26384628855128056, 0.38613441573117785, 0.26905211778581839, 0.093475425057556419, 0.16298403204515638, 0.20303836763654187, 0.54404509535514678, 0.10805767427847106, 0.072320147579874164, 0.21496588219036639, 0.054634444235349633, 0.44873824219253278, 0.63876685111371323, 0.423660117387191, 0.7225747031577624, 0.17733632991738643, 0.058614640369060278, 0.92775033584757161, 0.45543492639273753, 0.096995076493202409, 0.37159696330862996, 0.15671154482322525, 0.10031374417919169, 0.5504990218786856, 0.84948248827200523, 0.93243156676323202, 0.059171010901315553, 0.18366203201064149, 0.093525397852767186, 0.19375717190292086, 0.74600891995436158, 0.68054369119084157, 0.14257662081154526, 0.075732303909298235, 0.55519641898823102, 0.094505154843921677, 0.35825870989149949, 0.54438193746807295]

"""