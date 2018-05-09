import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas

# Final Code For Word Recall Full Model

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
valence_norm = conc_freq_len.valences_norm
arousal_norm = conc_freq_len.arousals_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence_norm)
all_arousal = np.array(arousal_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)


all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_arousal_part = []
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
        print("LOOK HERE:", all_probs_part)
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_valence_part = all_valence
        all_arousal_part = all_arousal
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        # print(all_imageability)
        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part_norm = np.array(all_valence_part)[mask]
        all_arousal_part_norm = np.array(all_arousal_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]


        # print(len(all_probs_part_norm))
        # print(len(all_concreteness_part))
        # print(len(all_wordfreq_part))
        # print(len(all_wordlen_part))
        # print(len(all_valence_part_norm))
        # print(len(all_arousal_part_norm))
        # print(len(all_imag_part_norm))
        # print(len(all_m_list_part_norm))
        # print(len(all_means_mpool_part_norm))
        #
        # print((all_probs_part_norm))
        # print((all_concreteness_part))
        # print((all_wordfreq_part))
        # print((all_wordlen_part))
        # print((all_valence_part_norm))
        # print((all_arousal_part_norm))
        # print((all_imag_part_norm))
        # print((all_m_list_part_norm))
        # print((all_means_mpool_part_norm))

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part, all_valence_part_norm,
                                    all_arousal_part_norm, all_m_list_part_norm, all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        # print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3],
                                results.pvalues[4], results.pvalues[5], results.pvalues[6], results.pvalues[7]], alpha=0.05,
                               method='fdr_bh', is_sorted=False, returnsorted=False))

        fdr_pvalues.append(fdr_p[0])
        f_pvalues.append(results.f_pvalue)
        residuals.append(results.resid)

    return params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues


params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues = calculate_params()

rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
print(fdr_pvalues.shape)

residuals = np.array(residuals)


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


t_test_intercepts = scipy.stats.ttest_1samp(params[:, 0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:, 1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:, 2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:, 3], 0)
t_test_valence = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_arousal = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 7], 0)


fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                     t_test_valence[1], t_test_arousal[1],
                     t_test_mlist[1], t_test_mpool[1]],
                    alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)


# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_valence = params[:, 4]
y_arousal = params[:, 5]
y_mlist = params[:, 6]
y_mpool = params[:, 7]


beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))
beta_mpool = (np.array(params[:, 7]))


mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)


print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))


ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3) , round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal','List Meaningfulness', 'Pool Meaningfulness', ]
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names=("Word Recall Model", "Mean Betas", "SEM Betas"),
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

betas_sig_valence = []
betas_nonsig_valence = []
for i in range(len(params[:, 4])):
    if fdr_pvalues[:, 4][i]:
        betas_sig_valence.append(params[:, 4][i])
    else:
        betas_nonsig_valence.append(params[:, 4][i])

betas_sig_arousal = []
betas_nonsig_arousal = []
for i in range(len(params[:, 5])):
    if fdr_pvalues[:, 5][i]:
        betas_sig_arousal.append(params[:, 5][i])
    else:
        betas_nonsig_arousal.append(params[:, 5][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_mlist.append(params[:, 6][i])
    else:
        betas_nonsig_mlist.append(params[:, 6][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 7])):
    if fdr_pvalues[:, 7][i]:
        betas_sig_mpool.append(params[:, 7][i])
    else:
        betas_nonsig_mpool.append(params[:,7][i])


x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_valence_sig = np.full(len(betas_sig_valence), 4)
x_valence_nonsig = np.full(len(betas_nonsig_valence), 4)

x_arousal_sig = np.full(len(betas_sig_arousal), 5)
x_arousal_nonsig = np.full(len(betas_nonsig_arousal), 5)


x_mlist_sig = np.full(len(betas_sig_mlist), 6)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 6)

x_mpool_sig = np.full(len(betas_sig_mpool), 7)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 7)

from matplotlib import gridspec

fig = plt.figure(figsize=(12, 5.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

ax2.axhline(y=0, color='gray', linestyle='--')

ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color='black')
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray')
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth=3, color='black')
if fdr_correction[0][0]:
    ax2.scatter(1, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
if fdr_correction[0][1]:
    ax2.scatter(2, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')
if fdr_correction[0][2]:
    ax2.scatter(3, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
if fdr_correction[0][3]:
    ax2.scatter(4, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
if fdr_correction[0][4]:
    ax2.scatter(5, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
if fdr_correction[0][5]:
    ax2.scatter(6, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
if fdr_correction[0][6]:
    ax2.scatter(7, 0.34, s=65, marker=(5, 2), color='black')


labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Valence'
labels[5] = 'Arousal'
labels[6] = 'M List'
labels[7] = 'M Pool'


ax2.set_xticklabels(labels, rotation=18, size=13)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .41)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax2.set_ylabel("Beta Value", size = 16)

ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color='black', label=" p < 0.05")
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label="Not Significant")
ax3.yaxis.set_label_position("right")
# ax3.set_xlabel("R-Squared Value", size = 14)
ax3.set_xlabel("All Participants", size = 14)
ax3.set_ylabel("R-Squared Value", size = 16)
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995, 1.005], [mean_combined, mean_combined], linewidth=3, color='black')


plt.savefig("Final_Word_Recall_CEMS.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
(88, 8)
Significant R-Squareds [0.068061624410127086, 0.063400117799403133, 0.026298234118707908, 0.053489255744613162, 0.072851052619091305, 0.055975938194718244, 0.095265233218882139, 0.038026778754541635, 0.067622361293077526, 0.040270181798678584, 0.090139935282847916, 0.10139699055378282, 0.06829130148122331, 0.048381070785579516, 0.18816649016745091, 0.057583022330298772, 0.039380810138415767, 0.035257164147751263, 0.058410426941560623, 0.031916026758604388, 0.031130544982028496, 0.036955544231724646, 0.034821084485444853, 0.052504939697816222, 0.039633608947073018, 0.028643226417845669, 0.069069701098587788, 0.0370551356226827, 0.049997203858678785, 0.12679072107910372, 0.067433085436909956, 0.058045290508210501, 0.064305479436340129, 0.042370728574326888, 0.033501937145184679, 0.073255308412406017, 0.085549643027920896, 0.027742571795859483, 0.047288070861290277, 0.11209760312214612, 0.040093164600087716, 0.060441531937211357, 0.037010552958645415, 0.067975855573908395, 0.033130840052440158, 0.07264900699804222, 0.051368880854720245, 0.024769579391067609, 0.033994974366039932, 0.068611356047153516, 0.065738885407905068, 0.043318279560905282, 0.028357307046984581, 0.03838100589777127, 0.091173239782034177, 0.029100335076589157, 0.026898492711008859, 0.087835995518129373, 0.027202875871022436, 0.053407370320166847, 0.035401841283412061, 0.049946110834480084, 0.070747106256867642, 0.047205655206481567, 0.050337777723208466, 0.071233717026279786, 0.073675497628401976, 0.067909494068499998, 0.028893986565079022, 0.11504745648337289, 0.065532444651221544, 0.10885889469260557, 0.040020932064063763, 0.070048063044950593, 0.11360824271734826, 0.056686951570240285, 0.028793332023187679, 0.064134255798924289, 0.091839486630523659, 0.13660830064690366, 0.04411815661380647]
Not Significant R-Squareds [0.018177460195848094, 0.013781616781586847, 0.022440998256665567, 0.021099269040252056, 0.023997022353468789, 0.015934290972450249, 0.018703945970302227]
R sig 81
R not sig 7
(array([False,  True, False,  True,  True,  True,  True], dtype=bool), array([  5.00103758e-01,   5.68195240e-15,   3.32908070e-01,
         5.61190529e-20,   1.33637771e-23,   3.17376894e-02,
         5.49833040e-29]), 0.0073008319790146547, 0.0071428571428571435)
T-test Intercepts Ttest_1sampResult(statistic=-9.2294670527243703, pvalue=1.521960656645029e-14)
T-test Concreteness Ttest_1sampResult(statistic=-0.67715582196072155, pvalue=0.50010375831629517)
T-test WordFreq Ttest_1sampResult(statistic=9.5571965757957589, pvalue=3.2468299437822744e-15)
T-test WordLen Ttest_1sampResult(statistic=-1.0749938795188314, pvalue=0.28534977404630285)
T-test Valence Ttest_1sampResult(statistic=12.102084838096571, pvalue=2.4051022692046671e-20)
T-test Arousal Ttest_1sampResult(statistic=14.076250440136391, pvalue=3.8182220350800788e-24)
T-test MList Ttest_1sampResult(statistic=2.3202169159270842, pvalue=0.022669778122750547)
T-test MPool Ttest_1sampResult(statistic=17.257707856489976, pvalue=7.8547577137776364e-30)
Beta Conc Ave, SE -0.00436772572163 0.00645010436296
Beta WordFreq Ave, SE 0.0614909379076 0.00643399321338
Beta Wordlen Ave, SE -0.00566661929391 0.00527130377379
Beta Valence Ave, SE 0.108433492644 0.00895990187593
Beta Arousal Ave, SE 0.130164186862 0.00924707807775
Beta MList Ave, SE 0.0106592579281 0.00459407818938
Beta MPool Ave, SE 0.131218367016 0.00760346438284
[-0.0040000000000000001, 0.060999999999999999, -0.0060000000000000001, 0.108, 0.13, 0.010999999999999999, 0.13100000000000001] [-0.0040000000000000001, 0.060999999999999999, -0.0060000000000000001, 0.108, 0.13, 0.010999999999999999, 0.13100000000000001]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.004     0.006
     Word Frequency      0.061     0.006
        Word Length     -0.006     0.005
            Valence      0.108     0.009
            Arousal       0.13     0.009
List Meaningfulness      0.011     0.005
Pool Meaningfulness      0.131     0.008
 
 
rsq [0.06806162441012709, 0.06340011779940313, 0.02629823411870791, 0.05348925574461316, 0.0728510526190913, 0.055975938194718244, 0.09526523321888214, 0.038026778754541635, 0.06762236129307753, 0.018177460195848094, 0.040270181798678584, 0.09013993528284792, 0.10139699055378282, 0.013781616781586847, 0.06829130148122331, 0.048381070785579516, 0.1881664901674509, 0.05758302233029877, 0.03938081013841577, 0.03525716414775126, 0.05841042694156062, 0.03191602675860439, 0.031130544982028496, 0.036955544231724646, 0.03482108448544485, 0.022440998256665567, 0.05250493969781622, 0.03963360894707302, 0.02864322641784567, 0.06906970109858779, 0.0370551356226827, 0.049997203858678785, 0.12679072107910372, 0.06743308543690996, 0.0580452905082105, 0.06430547943634013, 0.04237072857432689, 0.03350193714518468, 0.021099269040252056, 0.07325530841240602, 0.0855496430279209, 0.027742571795859483, 0.04728807086129028, 0.11209760312214612, 0.040093164600087716, 0.06044153193721136, 0.037010552958645415, 0.0679758555739084, 0.03313084005244016, 0.07264900699804222, 0.051368880854720245, 0.02476957939106761, 0.03399497436603993, 0.06861135604715352, 0.06573888540790507, 0.04331827956090528, 0.02399702235346879, 0.02835730704698458, 0.03838100589777127, 0.09117323978203418, 0.029100335076589157, 0.02689849271100886, 0.08783599551812937, 0.027202875871022436, 0.05340737032016685, 0.03540184128341206, 0.049946110834480084, 0.07074710625686764, 0.04720565520648157, 0.050337777723208466, 0.07123371702627979, 0.07367549762840198, 0.0679094940685, 0.028893986565079022, 0.11504745648337289, 0.06553244465122154, 0.10885889469260557, 0.04002093206406376, 0.07004806304495059, 0.01593429097245025, 0.11360824271734826, 0.056686951570240285, 0.02879333202318768, 0.018703945970302227, 0.06413425579892429, 0.09183948663052366, 0.13660830064690366, 0.04411815661380647]

"""