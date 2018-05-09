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

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP25*.mat')
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
        print("P values", pvalues)
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


print("Sig", sig)
print("Not_sig", no)
combined = np.array([rsquareds_sig, rsquareds_notsig]).tolist()
plt.hist(combined, 8, normed=1, histtype='bar', stacked=True, label = ['Significant Models', 'Not Significant Models'], color = ['black','darkgray'])
plt.title('R-Squared Values')
plt.xlabel("R-Squared Value", size = 14)
plt.ylabel("Frequency", size = 13)
plt.legend()
#plt.savefig("Rsq_hist_word_Nov14.pdf")


# plt.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
# plt.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )
#
# plt.xticks([])
# plt.legend()

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

fig, ax = plt.subplots()
fig.canvas.draw()
plt.axhline(y=0, color='gray', linestyle='--')

#plt.scatter(x_conc, y_concreteness, alpha=0.5)
plt.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
plt.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
plt.scatter(1, mean_beta_conc, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
plt.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
plt.scatter(2, mean_beta_wordfreq, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
plt.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
plt.scatter(3, mean_beta_wordlen, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
plt.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
plt.scatter(4, mean_beta_mlist, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
plt.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
plt.scatter(5, mean_beta_mpool, s =65, marker = (5,2), color = 'black' )

# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Concreteness'
labels[3] = 'Word Freq'
labels[5] = 'Word Length'
labels[7] = 'M List'
labels[9] = 'M Pool'

ax.set_xticklabels(labels)
ax.xaxis.set_ticks_position('none')


# print("Rsq", rsquareds.tolist())
#
# my_bins = np.arange(0, 1, .025).tolist()
#
# plt.hist(rsquareds , bins = my_bins)
# plt.title("Multiple Regressions")
# plt.xlabel("R-Squared Value")
# plt.ylabel("Frequency")

#plt.savefig("Betas_Word_Rec_Model_FINAL_Nov14.pdf")
plt.show()


print("rsq", rsquareds.tolist())


"""
OUTPUT:


(78, 6)
Significant R-Squareds [0.026138257220087446, 0.050018452581795159, 0.03959526929537327, 0.043040454354505653, 0.034270917273644286, 0.043048040653008601, 0.029562225971357714, 0.036767789494948411, 0.020244838359965045, 0.058021309488747619, 0.034269335299770054, 0.038998543288016441, 0.096872579675869264, 0.042508522085136446, 0.049629736266861069, 0.026600416696771378, 0.036917213230253432, 0.028421299218508556, 0.061557058329444359, 0.029749285225998956, 0.032838244139923978, 0.021875495647785126, 0.032009905502706992, 0.043357287780796061, 0.020101371690638858, 0.028118436538748348, 0.038815786112521566, 0.079605514360271856, 0.052213321130110013, 0.031693882620163927, 0.042586890487966489, 0.027641305119227244, 0.050803271576237652, 0.038650587552461646, 0.025122738640648845, 0.026079423623591946, 0.026591418734479566, 0.034472506957905824, 0.039359095229276497, 0.021391369445164177, 0.038269030339431076, 0.043975586137639855, 0.07947845101758233, 0.030418292464419294, 0.046842457476787502, 0.061365212921937196, 0.033618370397414998, 0.023562444028675689, 0.058547547261077471, 0.033084555050396869, 0.038319563013381353, 0.035418129814612898, 0.047468922756414456, 0.034531278990072023, 0.02783996033185121, 0.022561440395819088, 0.064524990359556389, 0.046839315551415406]
Not Significant R-Squareds [0.01488241217542241, 0.015432978064635394, 0.015705676141283043, 0.016017414918428452, 0.018081844436716343, 0.014252959275488086, 0.014618746395531512, 0.0031959694181830089, 0.0049911591758295959, 0.018313060676334425, 0.016570010580434058, 0.008105131893622386, 0.013034695417723619, 0.019444321922676866, 0.013262289165464125, 0.0081859456306370149, 0.0073725177594483604, 0.015425740208618088, 0.007990656865498158, 0.0109818377828651]
R sig 58
R not sig 20
Sig [ 0.02613826  0.05001845  0.03959527  0.04304045  0.03427092  0.04304804
  0.02956223  0.03676779  0.02024484  0.05802131  0.03426934  0.03899854
  0.09687258  0.04250852  0.04962974  0.02660042  0.03691721  0.0284213
  0.06155706  0.02974929  0.03283824  0.0218755   0.03200991  0.04335729
  0.02010137  0.02811844  0.03881579  0.07960551  0.05221332  0.03169388
  0.04258689  0.02764131  0.05080327  0.03865059  0.02512274  0.02607942
  0.02659142  0.03447251  0.0393591   0.02139137  0.03826903  0.04397559
  0.07947845  0.03041829  0.04684246  0.06136521  0.03361837  0.02356244
  0.05854755  0.03308456  0.03831956  0.03541813  0.04746892  0.03453128
  0.02783996  0.02256144  0.06452499  0.04683932]
Not_sig [ 0.01488241  0.01543298  0.01570568  0.01601741  0.01808184  0.01425296
  0.01461875  0.00319597  0.00499116  0.01831306  0.01657001  0.00810513
  0.0130347   0.01944432  0.01326229  0.00818595  0.00737252  0.01542574
  0.00799066  0.01098184]
(array([ True,  True, False, False,  True], dtype=bool), array([  8.50708325e-03,   3.70538380e-15,   7.32506538e-01,
         8.93350636e-02,   7.30423095e-25]), 0.010206218313011495, 0.01)
T-test Intercepts Ttest_1sampResult(statistic=-9.9829503843030327, pvalue=1.5580472698825876e-15)
T-test Concreteness Ttest_1sampResult(statistic=-2.8829630504448436, pvalue=0.0051042499523987215)
T-test WordFreq Ttest_1sampResult(statistic=9.9943187030672043, pvalue=1.4821535185812127e-15)
T-test WordLen Ttest_1sampResult(statistic=-0.34303548462679651, pvalue=0.73250653756778927)
T-test MList Ttest_1sampResult(statistic=1.8277027630066252, pvalue=0.071468050918023898)
T-test MPool Ttest_1sampResult(statistic=15.608240858622828, pvalue=1.4608461894161775e-25)
Beta Conc Ave, SE -0.0200249507447 0.00694596163539
Beta WordFreq Ave, SE 0.0726455619843 0.00726868575464
Beta Wordlen Ave, SE -0.00196233888794 0.00572051282121
Beta MList Ave, SE 0.00918692698493 0.0050264885357
Beta MPool Ave, SE 0.124704244088 0.00798964119133
[-0.02, 0.072999999999999995, -0.002, 0.0089999999999999993, 0.125] [-0.02, 0.072999999999999995, -0.002, 0.0089999999999999993, 0.125]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness      -0.02     0.007
     Word Frequency      0.073     0.007
        Word Length     -0.002     0.006
List Meaningfulness      0.009     0.005
Pool Meaningfulness      0.125     0.008
 
 

"""