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
arousal_norm = conc_freq_len.arousals_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_arousal = np.array(arousal_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
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
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
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

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,
                                    all_arousal_part_norm, all_m_list_part_norm, all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3],
                                results.pvalues[4], results.pvalues[5], results.pvalues[6]],
                               alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

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
t_test_arousal = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                                  t_test_arousal[1], t_test_mlist[1], t_test_mpool[1]],
                                alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Arousal", t_test_arousal)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_arousal = params[:, 4]
y_mlist = params[:, 5]
y_mpool = params[:, 6]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_arousal = (np.array(params[:, 4]))
beta_mlist = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_arousal), 3), round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3),  round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length',
              'Arousal', 'List Meaningfulness', 'Pool Meaningfulness', ]
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

betas_sig_arousal = []
betas_nonsig_arousal = []
for i in range(len(params[:, 4])):
    if fdr_pvalues[:, 4][i]:
        betas_sig_arousal.append(params[:, 4][i])
    else:
        betas_nonsig_arousal.append(params[:, 4][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:, 5])):
    if fdr_pvalues[:, 5][i]:
        betas_sig_mlist.append(params[:, 5][i])
    else:
        betas_nonsig_mlist.append(params[:, 5][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_mpool.append(params[:, 6][i])
    else:
        betas_nonsig_mpool.append(params[:, 6][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_arousal_sig = np.full(len(betas_sig_arousal), 4)
x_arousal_nonsig = np.full(len(betas_nonsig_arousal), 4)

x_mlist_sig = np.full(len(betas_sig_mlist), 5)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 5)

x_mpool_sig = np.full(len(betas_sig_mpool), 6)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 6)

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

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
if fdr_correction[0][3]:
    ax2.scatter(4, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
if fdr_correction[0][4]:
    ax2.scatter(5, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
if fdr_correction[0][5]:
    ax2.scatter(6, 0.34, s=65, marker=(5, 2), color='black')

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Arousal'
labels[5] = 'M List'
labels[6] = 'M Pool'

ax2.set_xticklabels(labels, rotation=18, size=13)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .41)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)


ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color='black', label=" p < 0.05")
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label="Not Significant")
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("R-Squared Value", size = 14)
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995, 1.005], [mean_combined, mean_combined], linewidth=3, color='black')

plt.savefig("Final_Word_Recall_NoValenceJan9FINAL.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""

 (88, 7)
Significant R-Squareds [0.046041406449653821, 0.056533952771510299, 0.025041898841292154, 0.049540766320627427, 0.054274157834507508, 0.03987350063876749, 0.071260102916561707, 0.038026066660288071, 0.060498892371730673, 0.077733437504966196, 0.080458765488668593, 0.065877122833715385, 0.036767957006858976, 0.091547881541734211, 0.048975969137252773, 0.023037921806187245, 0.034565295195983836, 0.052705050022069266, 0.026820032809936389, 0.033389921248812904, 0.031190931110426368, 0.051440667521362671, 0.037914002028147209, 0.048451985029031142, 0.026637768386307492, 0.049505293864095745, 0.10245993110386109, 0.067423885963900587, 0.053901002290930533, 0.04090628713366895, 0.041885549282542067, 0.060443894480218474, 0.038436506553072847, 0.10890107292478701, 0.024485079053213288, 0.05777126748786221, 0.036967102919521988, 0.047590303077852059, 0.032927677788789578, 0.061613127081002528, 0.049013168132290952, 0.033480945903123782, 0.065369367664291911, 0.061169687997283506, 0.02370129388738329, 0.02213103189227339, 0.034752752024753253, 0.070037855571891794, 0.028631379662409362, 0.022930190480443535, 0.080851640869752606, 0.025138796649264217, 0.037458254390125401, 0.028794206113399889, 0.039607310674660279, 0.043425924774522562, 0.035475983854515003, 0.022602538358369562, 0.067013479872961024, 0.071916792466035551, 0.033108289710696459, 0.02571397148882093, 0.089236500490508464, 0.045038470432867239, 0.05399273625702028, 0.033287096816760786, 0.035776205018084095, 0.057977307301195879, 0.053851312661167938, 0.026150896709008609, 0.045334530109365168, 0.082056646899760333, 0.097466659962371915, 0.029455479031194498]
Not Significant R-Squareds [0.015736469653722884, 0.01658790549604916, 0.0048798568358144889, 0.018517880297202693, 0.021738084780705047, 0.018890016679645627, 0.018824436847028392, 0.018486767647177604, 0.020185884769660323, 0.021709871925057578, 0.015550887319056073, 0.015826275256926703, 0.013080747329338616, 0.017725767460474939]
R sig 74
R not sig 14
(array([False,  True, False,  True,  True,  True], dtype=bool), array([  5.79458734e-01,   5.81493123e-19,   5.79458734e-01,
         2.78695904e-15,   4.72099438e-02,   1.73855245e-29]), 0.008512444610847103, 0.008333333333333333)
T-test Intercepts Ttest_1sampResult(statistic=-10.457766902929041, pvalue=4.7318980063872024e-17)
T-test Concreteness Ttest_1sampResult(statistic=-0.55626341046344585, pvalue=0.57945873447437735)
T-test WordFreq Ttest_1sampResult(statistic=11.644644495722847, pvalue=1.938310410780955e-19)
T-test WordLen Ttest_1sampResult(statistic=-0.69345897503878351, pvalue=0.48986884765986793)
T-test Arousal Ttest_1sampResult(statistic=9.736826368226339, pvalue=1.3934795189097175e-15)
T-test MList Ttest_1sampResult(statistic=2.1863693207103467, pvalue=0.031473295881330715)
T-test MPool Ttest_1sampResult(statistic=17.513528620778871, pvalue=2.897587419519028e-30)
Beta Conc Ave, SE -0.00359077936614 0.006455178066
Beta WordFreq Ave, SE 0.080593550648 0.00692108296458
Beta Wordlen Ave, SE -0.00363095379623 0.00523600375355
Beta Arousal Ave, SE 0.0710618429822 0.00729825512902
Beta MList Ave, SE 0.0101197103233 0.00462854570243
Beta MPool Ave, SE 0.133312868192 0.00761199362381
[-0.0040000000000000001, 0.081000000000000003, -0.0040000000000000001, 0.070999999999999994, 0.01, 0.13300000000000001] [-0.0040000000000000001, 0.081000000000000003, -0.0040000000000000001, 0.070999999999999994, 0.01, 0.13300000000000001]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.004     0.006
     Word Frequency      0.081     0.007
        Word Length     -0.004     0.005
            Arousal      0.071     0.007
List Meaningfulness       0.01     0.005
Pool Meaningfulness      0.133     0.008
 
 
rsq [0.04604140644965382, 0.0565339527715103, 0.025041898841292154, 0.04954076632062743, 0.05427415783450751, 0.03987350063876749, 0.07126010291656171, 0.03802606666028807, 0.06049889237173067, 0.015736469653722884, 0.01658790549604916, 0.0777334375049662, 0.0804587654886686, 0.004879856835814489, 0.06587712283371538, 0.036767957006858976, 0.09154788154173421, 0.04897596913725277, 0.023037921806187245, 0.034565295195983836, 0.052705050022069266, 0.02682003280993639, 0.018517880297202693, 0.033389921248812904, 0.031190931110426368, 0.021738084780705047, 0.05144066752136267, 0.03791400202814721, 0.018890016679645627, 0.04845198502903114, 0.026637768386307492, 0.049505293864095745, 0.10245993110386109, 0.06742388596390059, 0.05390100229093053, 0.04090628713366895, 0.018824436847028392, 0.018486767647177604, 0.020185884769660323, 0.04188554928254207, 0.060443894480218474, 0.021709871925057578, 0.03843650655307285, 0.10890107292478701, 0.024485079053213288, 0.05777126748786221, 0.03696710291952199, 0.04759030307785206, 0.03292767778878958, 0.06161312708100253, 0.04901316813229095, 0.015550887319056073, 0.03348094590312378, 0.06536936766429191, 0.061169687997283506, 0.015826275256926703, 0.02370129388738329, 0.02213103189227339, 0.03475275202475325, 0.0700378555718918, 0.028631379662409362, 0.022930190480443535, 0.0808516408697526, 0.025138796649264217, 0.0374582543901254, 0.02879420611339989, 0.03960731067466028, 0.04342592477452256, 0.035475983854515, 0.02260253835836956, 0.06701347987296102, 0.07191679246603555, 0.03310828971069646, 0.02571397148882093, 0.08923650049050846, 0.04503847043286724, 0.05399273625702028, 0.033287096816760786, 0.035776205018084095, 0.013080747329338616, 0.05797730730119588, 0.05385131266116794, 0.02615089670900861, 0.01772576746047494, 0.04533453010936517, 0.08205664689976033, 0.09746665996237192, 0.029455479031194498]


"""