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
    P_rec_ltpFR2_words as p_rec_word

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
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
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,
                                    all_means_mpool_part_norm))

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
t_test_mpool = scipy.stats.ttest_1samp(params[:, 4], 0)

print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_mpool[1]],
                    alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_mpool = params[:, 4]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_mpool = (np.array(params[:, 4]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
              round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_mpool), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',  'Pool Meaningfulness']
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

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 4])):
    if fdr_pvalues[:, 4][i]:
        betas_sig_mpool.append(params[:, 4][i])
    else:
        betas_nonsig_mpool.append(params[:, 4][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)


x_mpool_sig = np.full(len(betas_sig_mpool), 4)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 4)

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

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
ax2.scatter(4, 0.32, s=65, marker=(5, 2), color='black')

# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = ['', 'Concreteness', '' , 'Frequency', '', 'Length','', 'M Pool', '']
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
plt.savefig("Word_NoMList_Dec31.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
(87, 5)
Significant R-Squareds [0.042981047178569431, 0.02972806831009378, 0.021824813815780053, 0.041909168380249717, 0.04103064702573811, 0.035295498278862025, 0.059422144099958385, 0.028665319752328222, 0.04992695123075197, 0.036291552233048807, 0.077251077295061865, 0.05649250312887566, 0.028442916539922702, 0.077766448163522961, 0.043023474334723977, 0.020029871073119931, 0.033931998588267565, 0.02330418865177275, 0.027456972097850252, 0.029433944214214525, 0.038797374999065104, 0.033643366915827366, 0.017903347747555087, 0.043971170094445622, 0.025867604213746009, 0.04715075962570614, 0.096212396215087037, 0.042010384246675914, 0.040043251182922401, 0.03765927267837399, 0.0360575422467454, 0.036759644910571043, 0.038238196836703819, 0.052643163255075454, 0.023037368212377785, 0.032979841677695743, 0.028035491297123238, 0.039377834566651559, 0.032728988883527599, 0.05387894141881544, 0.041455823858442598, 0.016782687122492534, 0.057385894453119923, 0.035377306261820562, 0.019358149596709895, 0.034458530156494138, 0.023085083830296838, 0.026301811630135008, 0.019557127449048917, 0.078398052393190754, 0.028987693803085635, 0.028383158803407804, 0.031279011151743252, 0.026041822147099825, 0.023419013210025486, 0.02169021362589918, 0.064329237842591591, 0.040396710131208158, 0.031612090504052781, 0.019613225693845604, 0.052212077896034326, 0.023450764232880172, 0.046618106765715872, 0.030036422829277654, 0.032476918189855208, 0.02246239875528333, 0.053282083579138995, 0.041755365343261297, 0.049020668790872191, 0.063214105816671973]
Not Significant R-Squareds [0.0054500232764050915, 0.015611845757436682, 0.0031369091617505784, 0.012630147474262943, 0.014791456239586109, 0.015774552049927171, 0.0085310866441117295, 0.011982693515172671, 0.0040469400857443549, 0.006597347447777735, 0.005752577001991166, 0.0079884499447369084, 0.015340793189280055, 0.013093563643563066, 0.013260323139799191, 0.0079644698054911478, 0.01023916474361608]
R sig 70
R not sig 17
(array([ True,  True, False,  True], dtype=bool), array([  9.67236933e-03,   2.23836444e-18,   7.12877057e-01,
         2.50177276e-28]), 0.012741455098566168, 0.0125)
T-test Intercepts Ttest_1sampResult(statistic=-10.690123410899739, pvalue=1.8397485121391806e-17)
T-test Concreteness Ttest_1sampResult(statistic=-2.7504660537395544, pvalue=0.0072542769949979239)
T-test WordFreq Ttest_1sampResult(statistic=11.297884404589398, pvalue=1.1191822179206541e-18)
T-test WordLen Ttest_1sampResult(statistic=-0.3692121982431445, pvalue=0.71287705722693606)
T-test MPool Ttest_1sampResult(statistic=16.819486102336178, pvalue=6.2544318953230066e-29)
Beta Conc Ave, SE -0.0189251377805 0.00688070218309
Beta WordFreq Ave, SE 0.0792633437092 0.00701576869356
Beta Wordlen Ave, SE -0.00194823012647 0.00527672199278
Beta MPool Ave, SE 0.131297456919 0.00780627042467
[-0.019, 0.079000000000000001, -0.002, 0.13100000000000001] [-0.019, 0.079000000000000001, -0.002, 0.13100000000000001]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.019     0.007
     Word Frequency      0.079     0.007
        Word Length     -0.002     0.005
Pool Meaningfulness      0.131     0.008
 
"""