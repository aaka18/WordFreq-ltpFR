"""1. WORD RECALL MODEL I
{Each of these vectors will have a dimension of 576 words}
1)	Word Frequency -
2)	Word Concreteness -
3)	Word Length -
4)	Mlist (Each word’s aver. similarity to all other words in the list) - m_list
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

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, P_rec_ltpFR2_words as p_rec_word, \
    semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP33[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_m_list = np.array(m_list)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []

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

        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part, all_m_list_part_norm))

        # x-values without the m_list
        # x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        # print(x_values)
        # print(np.shape(x_values))

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
        fdr_p = (multipletests(
            [results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
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
# print("Pvalues", pvalues)
# print(pvalues.shape)
# print("Corrected p values", fdr_pvalues)

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




t_test_intercepts = scipy.stats.ttest_1samp(params[:, 0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:, 1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:, 2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:, 3], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 4], 0)

print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_mlist[1]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_mlist = params[:, 4]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_mlist = (np.array(params[:, 4]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_mlist), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_mlist), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'List Meaningfulness']
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names=("Word Recall Model I", "Mean Betas", "SEM Betas"),
          dtype=('str', 'float', 'float'))

print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")

betas_sig_conc = []
betas_nonsig_conc = []
for i in range(len(params[:, 1])):
    if fdr_pvalues[:, 1][i]:
        betas_sig_conc.append(params[:, 1][i])
    else:
        betas_nonsig_conc.append(params[:, 1][i])

betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:, 2])):
    if fdr_pvalues[:, 2][i]:
        betas_sig_wordfreq.append(params[:, 2][i])
    else:
        betas_nonsig_wordfreq.append(params[:, 2][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:, 3])):
    if fdr_pvalues[:, 3][i]:
        betas_sig_wordlen.append(params[:, 3][i])
    else:
        betas_nonsig_wordlen.append(params[:, 3][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:, 4])):
    if fdr_pvalues[:, 4][i]:
        betas_sig_mlist.append(params[:, 4][i])
    else:
        betas_nonsig_mlist.append(params[:, 4][i])


x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_mlist_sig = np.full(len(betas_sig_mlist), 4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 4)


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

# plt.scatter(x_conc, y_concreteness, alpha=0.5)
ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color='black')
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray')
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth=3, color='black')
ax2.scatter(1, 0.32, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
ax2.scatter(2, 0.32, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
ax2.scatter(4, 0.32, s=65, marker=(5, 2), color='black')


# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = ['', 'Concreteness', '' , 'Frequency', '', 'Length','', 'M List', '']
ax2.set_xticklabels(labels)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .37)
ax2.set_xticklabels(labels, rotation = 20, size = 10)


# print("Rsq", rsquareds.tolist())
#
# my_bins = np.arange(0, 1, .025).tolist()
#
# plt.hist(rsquareds , bins = my_bins)
# plt.title("Multiple Regressions")
# plt.xlabel("R-Squared Value")
# plt.ylabel("Frequency")


# plt.subplot(1, 2, 2)
ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color='black', label=" p < 0.05")
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label="Not Significant")
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("R-Squared Value")
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995, 1.005], [mean_combined, mean_combined], linewidth=3, color='black')

# ax3.legend()
plt.savefig("Word_No_MPool_Dec31.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""

(87, 5)
Significant R-Squareds [0.017056079520865297, 0.040196954901285831, 0.021317004377213622, 0.055913895740239639, 0.022236940316200493, 0.029329052653818244, 0.031007762711020348, 0.0467537692963782, 0.017279995972951934, 0.02623665899426797, 0.035450592103792089, 0.031007924353185445, 0.018894863482552759, 0.017710601148488103, 0.020356754249777786, 0.029752740350908691, 0.020214622082762701, 0.030869171696088604, 0.030424894037000882, 0.0338276368615823, 0.028034522676341878, 0.018280666081272545, 0.033713922518721851, 0.035283158806846959, 0.026975976842946836, 0.022719020278401891, 0.017806446847022372, 0.025726960597507675, 0.040561882064224508, 0.017049994622496123, 0.018205440298764919, 0.017201993328074039, 0.019230939753491105, 0.018804682084129776, 0.020456500008546485, 0.05579093641049726, 0.018701754205348031, 0.018492812307256634, 0.022125418874134328, 0.040671847244364123, 0.018181691222656338, 0.036500964183924234, 0.022703670629285133, 0.045685908558763089, 0.018515783051346046, 0.018324128777795989, 0.018131134564782436, 0.041039458322993494]
Not Significant R-Squareds [0.01496606644865528, 0.011803772215358888, 0.013807822122946067, 0.0066682708975687177, 0.0086985286496435998, 0.0021543835320985139, 0.015597400086987734, 0.0056241801011316372, 0.0066721440310553159, 0.0037084524753728809, 0.011810922634166499, 0.013446801042359291, 0.014167457220670809, 0.013578688664052541, 0.012556904753346254, 0.0052346664932607645, 0.0049596870156719541, 0.012259802508132189, 0.0069857833015783966, 0.0071119871080369546, 0.012797140553548081, 0.0049392125289769995, 0.0081791569828328914, 0.0098732186173425829, 0.0071213665336437026, 0.0043840417243210528, 0.0093605965486404097, 0.0089332820512552091, 0.0046775523938099584, 0.0084859376949840692, 0.011320090482150991, 0.016214483849483874, 0.015228317534749181, 0.0055847419545168586, 0.016242311203004345, 0.0087584564966221023, 0.0078360579232661376, 0.011674233493248143, 0.012384832517842503]
R sig 48
R not sig 39
(array([ True,  True, False,  True], dtype=bool), array([  2.08335537e-06,   5.14773674e-14,   9.24065387e-01,
         2.27533216e-11]), 0.012741455098566168, 0.0125)
T-test Intercepts Ttest_1sampResult(statistic=-8.8881454986147972, pvalue=8.3149719673474622e-14)
T-test Concreteness Ttest_1sampResult(statistic=5.1606180858624402, pvalue=1.5625165288078679e-06)
T-test WordFreq Ttest_1sampResult(statistic=9.28604005810055, pvalue=1.2869341850025518e-14)
T-test WordLen Ttest_1sampResult(statistic=-0.09559405406446507, pvalue=0.92406538738532684)
T-test MList Ttest_1sampResult(statistic=7.8352978360141439, pvalue=1.1376660781745649e-11)
Beta Conc Ave, SE 0.0338261673474 0.0065546736427
Beta WordFreq Ave, SE 0.0652073914049 0.00702208810181
Beta Wordlen Ave, SE -0.000506623832768 0.0052997420993
Beta MList Ave, SE 0.040521031178 0.00517160062401
[0.034000000000000002, 0.065000000000000002, -0.001, 0.041000000000000002] [0.034000000000000002, 0.065000000000000002, -0.001, 0.041000000000000002]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness      0.034     0.007
     Word Frequency      0.065     0.007
        Word Length     -0.001     0.005
List Meaningfulness      0.041     0.005
 

 """