"""1. LIST RECALL MODEL I
{Each of these vectors will have a dimension of 552 lists}
# {Each of these vectors will have a dimension of N participants * 552 lists}
# 1)	Average Word Frequency Of The List - all_mean_word_freq_participants
# 2)	Average Word Concreteness Of The List - all_mean_conc_participants
# 3)	Average Word Length Of The List - all_mean_wordlen_participants
# 4)	Mlist (Each word’s aver. similarity to all other words in the list) - m_list
# 5)	Mpool (Each word’s aver. similarity to all other words in the pool) - m_pool
# 6) P_rec per list - p_rec_list

"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, MPool_byPoolSim as m_pool, \
    P_rec_ltpFR2_by_list as p_rec, semantic_similarity_bylist as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP32*.mat')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)

all_probs = np.array(p_rec_list)
# print(np.array(np.shape(all_probs)))
all_concreteness= np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_means_mpool = np.array(all_means_mpool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []
pvalues = []
predict = []
residuals = []
f_pvalues = []
fdr_pvalues = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordfreq_part = all_wordfreq[i]
        all_wordlen_part = all_wordlen[i]
        all_m_list_part = all_m_list[i]
        all_means_mpool_part = all_means_mpool


        # mask them since some p_recs have nan's

        mask = np.isfinite(all_probs_part)
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part = np.array(all_probs_part)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_m_list_part = np.array(all_m_list_part)[mask]
        all_means_mpool_part = np.array(all_means_mpool_part)[mask]

        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=1, ddof=1)
        all_concreteness_part_norm = stats.mstats.zscore(all_concreteness_part, axis=1, ddof=1)
        all_wordfreq_part_norm = stats.mstats.zscore(all_wordfreq_part,axis=1, ddof=1)
        all_wordlen_part_norm = stats.mstats.zscore(all_wordlen_part,axis=1, ddof=1)
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=1,ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part,axis=1, ddof=1)

        sm.OLS

        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,  all_m_list_part_norm, all_means_mpool_part_norm))

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

plt.title("R-Squared Values of Word Recall Model I")
plt.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
plt.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )

plt.xticks([])
plt.legend()
plt.savefig("Rsq_Word_Rec_Model.pdf")

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

plt.title("Beta Values of Word Recall Model I")
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

plt.savefig("Betas_List_Rec_Model.pdf")
plt.show()

plt.show()

print("rsq", rsquareds.tolist())


"""
N = 76
T-test Concreteness Ttest_1sampResult(statistic=-2.10786298782457, pvalue=0.038384210719160303)
T-test WordFreq Ttest_1sampResult(statistic=10.286410217297412, pvalue=5.5504947128479115e-16)
T-test WordLen Ttest_1sampResult(statistic=-0.33380816586705037, pvalue=0.73945585547747972)
T-test MList Ttest_1sampResult(statistic=2.701908807280943, pvalue=0.0085209333872856736)
T-test MPool Ttest_1sampResult(statistic=14.207023435844965, pvalue=5.7642837025963944e-23)
Beta Conc Ave, SE -0.0148290092308 0.00703509161481
Beta WordFreq Ave, SE 0.073041988774 0.00710082402228
Beta Wordlen Ave, SE -0.00195300732255 0.0058506876771
Beta MList Ave, SE 0.0140016290284 0.00518212494467
Beta MPool Ave, SE 0.109157419346 0.00768334196383 

(array([ True,  True, False,  True,  True], dtype=bool), array([  4.79802634e-02,   1.38762368e-15,   7.39455855e-01,
         1.42015556e-02,   2.88214185e-22]), 0.010206218313011495, 0.01)

"""