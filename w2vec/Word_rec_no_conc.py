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
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
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
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        # # print("Mask", mask)
        # # print(np.shape(np.array(mask)))
        # all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        # all_concreteness_part = np.array(all_concreteness_part)[mask]
        # all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        # all_wordlen_part = np.array(all_wordlen_part)[mask]
        # all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        # all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        sm.OLS

        x_values = np.column_stack(( all_wordfreq_part, all_wordlen_part,  all_m_list_part_norm, all_means_mpool_part_norm))

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
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
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
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)

print(multipletests([t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_wordfreq = params[:,1]
y_wordlen = params[:,2]
y_mlist = params[:,3]
y_mpool = params[:,4]


beta_wordfreq = (np.array(params[:,1]))
beta_wordlen =(np.array(params[:,2]))
beta_mlist = (np.array(params[:,3]))
beta_mpool = (np.array(params[:,4]))

mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))


ave_betas = [ round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3),  round(np.mean(beta_mlist),3), round(np.mean(beta_mpool),3)]
sem_betas = [ round(stats.sem(beta_wordfreq),3), round(stats.sem(beta_wordlen),3),  round(stats.sem(beta_mlist),3),round(stats.sem(beta_mpool),3) ]
predictors = [ 'Word Frequency', 'Word Length',  'List Meaningfulness', 'Pool Meaningfulness']
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names = ("Word Recall Model I", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))

print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")



betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:,1])):
    if fdr_pvalues[:,1][i]:
        betas_sig_wordfreq.append(params[:,1][i])
    else:
        betas_nonsig_wordfreq.append(params[:,1][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:,2])):
    if fdr_pvalues[:,2][i]:
        betas_sig_wordlen.append(params[:,2][i])
    else:
        betas_nonsig_wordlen.append(params[:,2][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:,3])):
    if fdr_pvalues[:,3][i]:
        betas_sig_mlist.append(params[:,3][i])
    else:
        betas_nonsig_mlist.append(params[:,3][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:,4])):
    if fdr_pvalues[:,4][i]:
        betas_sig_mpool.append(params[:,4][i])
    else:
        betas_nonsig_mpool.append(params[:,4][i])


x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 1)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 1)

x_wordlen_sig = np.full(len(betas_sig_wordlen),2)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen),2)

x_mlist_sig = np.full(len(betas_sig_mlist),3)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist),3)

x_mpool_sig = np.full(len(betas_sig_mpool),4)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool),4)


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

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([0.8, 1.2], [mean_beta_wordfreq, mean_beta_wordfreq],  linewidth = 3, color = 'black' )
ax2.scatter(1, 0.32, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordlen, mean_beta_wordlen],  linewidth = 3, color = 'black' )

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_mlist, mean_beta_mlist],  linewidth = 3, color = 'black' )

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([3.8,4.2], [mean_beta_mpool, mean_beta_mpool],  linewidth = 3, color = 'black' )
ax2.scatter(4, 0.32, s =65, marker = (5,2), color = 'black' )

labels = ['', 'Frequency', '', 'Length','', 'M List', '', 'M Pool']
ax2.set_xticklabels(labels)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .37)
ax2.set_xticklabels(labels, rotation = 20, size = 10)


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
plt.savefig("Word_recall_No_Conc_Dec31.pdf")
plt.show()
"""
(87, 5)
Significant R-Squareds [0.03869081932544427, 0.026387589398850886, 0.022571037693852158, 0.034630172391803771, 0.042224711371431467, 0.03997662753274378, 0.058888700278846962, 0.031835540355337422, 0.044055759581563492, 0.021163765738144491, 0.074668267186415616, 0.042034013082804633, 0.01654652515878019, 0.072076702694910844, 0.028882371236916948, 0.018501174867058823, 0.030907819326996266, 0.019821048890565796, 0.021919731900229666, 0.020651249970298657, 0.036457836640311325, 0.033366422883486013, 0.018941657729635519, 0.035900138286600991, 0.025369024070579838, 0.037402759067921454, 0.075013046347641166, 0.037612265906156739, 0.035198812534429469, 0.041319144429815213, 0.029186815047376524, 0.034222562774165133, 0.032670918211314404, 0.059602722901408112, 0.022899406591523896, 0.031332686946477173, 0.029687208983030788, 0.038649180984234421, 0.033253269772532734, 0.051946015673417723, 0.032953174135664676, 0.045705019852998463, 0.025300795126457443, 0.028372983056008771, 0.018913388747394677, 0.017562500725074592, 0.055268220289518721, 0.025752885997397112, 0.022397279112741364, 0.032945693007429666, 0.022741902850179296, 0.026102109010495922, 0.062757011739744017, 0.033709582813868377, 0.027473892760222496, 0.042673129224193973, 0.024951964839284524, 0.052013990230874274, 0.023043591455238821, 0.030318457202135884, 0.021372077575259274, 0.047790315839244535, 0.033427739821438207, 0.043204366970205355, 0.047087332198472276]
Not Significant R-Squareds [0.013800525574667266, 0.011030273620054376, 0.0016294236625454817, 0.01274137342323145, 0.013593515514558452, 0.0097502334525095824, 0.012414738605639242, 0.010710016236815778, 0.0023271872811333472, 0.0057225185418288138, 0.011668860714173612, 0.012267589020921954, 0.0051467668397433552, 0.011567779135188916, 0.015058602582389335, 0.010050104697289175, 0.010759416728646443, 0.012147784758565261, 0.015799245138499174, 0.010474542486266225, 0.008179320031620585, 0.0090601617165804171]
R sig 65
R not sig 22
(array([ True, False, False,  True], dtype=bool), array([  1.12074008e-20,   4.29299532e-01,   9.20792455e-02,
         2.74462798e-25]), 0.012741455098566168, 0.0125)
T-test Intercepts Ttest_1sampResult(statistic=-0.11600096397976842, pvalue=0.90792222321574345)
T-test WordFreq Ttest_1sampResult(statistic=12.467827733267194, pvalue=5.6037004163730343e-21)
T-test WordLen Ttest_1sampResult(statistic=-0.79414055075966961, pvalue=0.42929953214025607)
T-test MList Ttest_1sampResult(statistic=1.8410694307169928, pvalue=0.069059434109451076)
T-test MPool Ttest_1sampResult(statistic=15.088647986162798, pvalue=6.86156994033542e-26)
Beta WordFreq Ave, SE 0.0864182211653 0.00693129733696
Beta Wordlen Ave, SE -0.00417728466722 0.00526013268461
Beta MList Ave, SE 0.00873687591914 0.00474554396123
Beta MPool Ave, SE 0.105965487369 0.0070228616551
[0.085999999999999993, -0.0040000000000000001, 0.0089999999999999993, 0.106] [0.085999999999999993, -0.0040000000000000001, 0.0089999999999999993, 0.106]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
     Word Frequency      0.086     0.007
        Word Length     -0.004     0.005
List Meaningfulness      0.009     0.005
Pool Meaningfulness      0.106     0.007
 """