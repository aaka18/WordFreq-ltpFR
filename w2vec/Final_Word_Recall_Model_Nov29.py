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
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP33[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

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
fdr_pvalues = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

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


        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        #print("P values", pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4], results.pvalues[5]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
        fdr_pvalues.append(fdr_p[0])
        f_pvalues.append(results.f_pvalue)
        residuals.append(results.resid)

    return params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues


params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues = calculate_params()
# print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordFreq coeff, WordLen coeff, MList coeff, MPool coeff)", params)
# print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
print(fdr_pvalues.shape)
#print("Pvalues", pvalues)
#print(pvalues.shape)
#print("Corrected p values", fdr_pvalues)

residuals = np.array(residuals)
# print("Residuals", residuals)
# print("F pvalues", f_pvalues)


rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if f_pvalues[i] < .05:
        rsquareds_sig.append(rsquareds[i])
    else:
        rsquareds_notsig.append(rsquareds[i])
print("Significant R-Squareds", rsquareds_sig)
print("Not Significant R-Squareds", rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

print("R sig", len(rsquareds_sig))
print("R not sig", len(rsquareds_notsig))

sig = np.array(rsquareds_sig)
no = np.array(rsquareds_notsig)


# print("Sig", sig)
# print("Not_sig", no)
# combined = np.array([rsquareds_sig, rsquareds_notsig]).tolist()
# plt.hist(combined, 8, normed=1, histtype='bar', stacked=True, label = ['Significant Models', 'Not Significant Models'], color = ['black','darkgray'])
# plt.title('R-Squared Values')
# plt.xlabel("R-Squared Value", size = 14)
# plt.ylabel("Frequency", size = 13)
# plt.legend()
# #plt.savefig("Rsq_hist_word_Nov14.pdf")




t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,5], 0)

print(multipletests([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordfreq = params[:,2]
y_wordlen = params[:,3]
y_mlist = params[:,4]
y_mpool = params[:,5]


beta_concreteness = (np.array(params[:,1]))
beta_wordfreq = (np.array(params[:,2]))
beta_wordlen =(np.array(params[:,3]))
beta_mlist = (np.array(params[:,4]))
beta_mpool = (np.array(params[:,5]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))


ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3),  round(np.mean(beta_mlist),3), round(np.mean(beta_mpool),3)]
sem_betas = [round(stats.sem(beta_concreteness),3), round(stats.sem(beta_wordfreq),3), round(stats.sem(beta_wordlen),3),  round(stats.sem(beta_mlist),3),round(stats.sem(beta_mpool),3) ]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',  'List Meaningfulness', 'Pool Meaningfulness']
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names = ("Word Recall Model I", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))

print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")


betas_sig_conc = []
betas_nonsig_conc = []
for i in range(len(params[:,1])):
    if fdr_pvalues[:,1][i]:
        betas_sig_conc.append(params[:,1][i])
    else:
        betas_nonsig_conc.append(params[:,1][i])


betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:,2])):
    if fdr_pvalues[:,2][i]:
        betas_sig_wordfreq.append(params[:,2][i])
    else:
        betas_nonsig_wordfreq.append(params[:,2][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:,3])):
    if fdr_pvalues[:,3][i]:
        betas_sig_wordlen.append(params[:,3][i])
    else:
        betas_nonsig_wordlen.append(params[:,3][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:,4])):
    if fdr_pvalues[:,4][i]:
        betas_sig_mlist.append(params[:,4][i])
    else:
        betas_nonsig_mlist.append(params[:,4][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:,5])):
    if fdr_pvalues[:,5][i]:
        betas_sig_mpool.append(params[:,5][i])
    else:
        betas_nonsig_mpool.append(params[:,5][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen),3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen),3)

x_mlist_sig = np.full(len(betas_sig_mlist),4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist),4)

x_mpool_sig = np.full(len(betas_sig_mpool),5)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool),5)


# fig, ax = plt.subplots()
# ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=3, colspan = 2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=3)
# fig.canvas.draw()
from matplotlib import gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])


ax2.axhline(y=0, color='gray', linestyle='--')

#plt.scatter(x_conc, y_concreteness, alpha=0.5)
ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth = 3, color = 'black' )
ax2.scatter(1, 0.32, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq],  linewidth = 3, color = 'black' )
ax2.scatter(2, 0.32, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen],  linewidth = 3, color = 'black' )

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_mlist, mean_beta_mlist],  linewidth = 3, color = 'black' )

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([4.8,5.2], [mean_beta_mpool, mean_beta_mpool],  linewidth = 3, color = 'black' )
ax2.scatter(5, 0.32, s =65, marker = (5,2), color = 'black' )

# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'M List'
labels[5] = 'M Pool'

ax2.set_xticklabels(labels, rotation = 20, size = 10)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .37)

# print("Rsq", rsquareds.tolist())
#
# my_bins = np.arange(0, 1, .025).tolist()
#
# plt.hist(rsquareds , bins = my_bins)
# plt.title("Multiple Regressions")
# plt.xlabel("R-Squared Value")
# plt.ylabel("Frequency")


# plt.subplot(1, 2, 2)
ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("R-Squared Value")
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995,1.005], [mean_combined, mean_combined], linewidth = 3, color = 'black')



#ax3.legend()
plt.savefig("Word_recall_Dec31.pdf")
plt.show()


print("rsq", rsquareds.tolist())
"""
(87, 6)
Significant R-Squareds [0.043048040653008601, 0.032009905502706992, 0.021875495647785126, 0.043040454354505653, 0.042586890487966489, 0.0379942430474004, 0.061557058329444359, 0.034269335299770054, 0.050803271576237652, 0.03959526929537327, 0.07947845101758233, 0.058021309488747619, 0.02857730310908646, 0.079605514360271856, 0.043357287780796061, 0.020101371690638858, 0.034270917273644286, 0.026079423623591946, 0.028118436538748348, 0.029562225971357714, 0.038815786112521566, 0.036534890243150153, 0.043975586137639855, 0.026591418734479566, 0.048715555825687762, 0.09637022023217845, 0.042508522085136446, 0.050018452581795159, 0.040724736224413638, 0.038650587552461646, 0.036767789494948411, 0.038269030339431076, 0.062045802862404353, 0.024866842690677182, 0.033084555050396869, 0.034777061629101547, 0.047468922756414456, 0.032838244139923978, 0.054162764128801388, 0.046842457476787502, 0.058547547261077471, 0.035610649503591385, 0.020918793139678704, 0.034531278990072023, 0.02783996033185121, 0.027641305119227244, 0.021391369445164177, 0.07840183058254302, 0.029749285225998956, 0.028385373033572914, 0.034472506957905824, 0.026600416696771378, 0.025122738640648845, 0.022561440395819088, 0.064524990359556389, 0.045568419929424508, 0.031693882620163927, 0.020244838359965045, 0.052213321130110013, 0.026083143619622717, 0.05246867912139308, 0.031088401746193761, 0.033618370397414998, 0.022594003053817202, 0.053328880440595805, 0.041867263774299612, 0.049529769783466127, 0.063303007223416752]
Not Significant R-Squareds [0.014618746395531512, 0.015705676141283043, 0.0031959694181830089, 0.013459441901932268, 0.01488241217542241, 0.016017414918428452, 0.018328007992359896, 0.018313060676334425, 0.016413312742339481, 0.0049911591758295959, 0.0073725177594483604, 0.013347891276098589, 0.019444321922676866, 0.0081859456306370149, 0.015425740208618088, 0.013369281268905242, 0.013262289165464125, 0.007990656865498158, 0.011676017501522806]
R sig 68
R not sig 19
(array([ True,  True, False, False,  True], dtype=bool), array([  1.17808462e-02,   2.68283215e-18,   7.24048657e-01,
         5.21969833e-02,   4.41988813e-28]), 0.010206218313011495, 0.01)
T-test Intercepts Ttest_1sampResult(statistic=-10.667384103544849, pvalue=2.0439953901489734e-17)
T-test Concreteness Ttest_1sampResult(statistic=-2.7597171362249084, pvalue=0.0070685077308527845)
T-test WordFreq Ttest_1sampResult(statistic=11.3070545798019, pvalue=1.0731328611740367e-18)
T-test WordLen Ttest_1sampResult(statistic=-0.35420981536853541, pvalue=0.72404865657758877)
T-test MList Ttest_1sampResult(statistic=2.0668282534627256, pvalue=0.041757586614276178)
T-test MPool Ttest_1sampResult(statistic=16.731678671706536, pvalue=8.8397762629852942e-29)
Beta Conc Ave, SE -0.018932187006 0.0068601911252
Beta WordFreq Ave, SE 0.0792691845464 0.00701059537539
Beta Wordlen Ave, SE -0.00187460481816 0.00529235706302
Beta MList Ave, SE 0.00972731451983 0.00470639711042
Beta MPool Ave, SE 0.128151701983 0.00765922562211
[-0.019, 0.079000000000000001, -0.002, 0.01, 0.128] [-0.019, 0.079000000000000001, -0.002, 0.01, 0.128]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.019     0.007
     Word Frequency      0.079     0.007
        Word Length     -0.002     0.005
List Meaningfulness       0.01     0.005
Pool Meaningfulness      0.128     0.008
 
 
 
 
 """