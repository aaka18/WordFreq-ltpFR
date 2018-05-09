import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas

# Final Code For Word Recall Without Valence

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

# files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
arousal_norm = conc_freq_len.arousals_norm
imag_norm = conc_freq_len.imageability_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_arousal = np.array(arousal_norm)
all_imageability = np.array(imag_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_arousal_part = []
all_imag_part = []
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
        all_imag_part = all_imageability
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
        all_arousal_part_norm = np.array(all_arousal_part)[mask]
        all_imag_part_norm = np.array(all_imag_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,
                                    all_arousal_part_norm, all_imag_part_norm, all_m_list_part_norm,
                                    all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        fdr_p = (multipletests(
            [results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4],
             results.pvalues[5], results.pvalues[6], results.pvalues[7]], alpha=0.05,
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
t_test_arousal = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_imag = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 7], 0)

print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                      t_test_arousal[1], t_test_imag[1], t_test_mlist[1], t_test_mpool[1]],
                    alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Arousal", t_test_arousal)
print("T-test Imag", t_test_imag)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_arousal = params[:, 4]
y_imag = params[:, 5]
y_mlist = params[:, 6]
y_mpool = params[:, 7]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_arousal = (np.array(params[:, 4]))
beta_imag = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))
beta_mpool = (np.array(params[:, 7]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_imag = np.mean(beta_imag)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta Imag Ave, SE", np.mean(beta_imag), stats.sem(beta_imag))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_arousal), 3), round(np.mean(beta_imag), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3),  round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_imag), 3), round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length',
              'Arousal', 'Imageability', 'List Meaningfulness', 'Pool Meaningfulness', ]
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

betas_sig_imag = []
betas_nonsig_imag = []
for i in range(len(params[:, 5])):
    if fdr_pvalues[:, 5][i]:
        betas_sig_imag.append(params[:, 5][i])
    else:
        betas_nonsig_imag.append(params[:, 5][i])

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
        betas_nonsig_mpool.append(params[:, 7][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_arousal_sig = np.full(len(betas_sig_arousal), 4)
x_arousal_nonsig = np.full(len(betas_nonsig_arousal), 4)

x_imag_sig = np.full(len(betas_sig_imag), 5)
x_imag_nonsig = np.full(len(betas_nonsig_imag), 5)

x_mlist_sig = np.full(len(betas_sig_mlist), 6)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 6)

x_mpool_sig = np.full(len(betas_sig_mpool), 7)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 7)

from matplotlib import gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

ax2.axhline(y=0, color='gray', linestyle='--')

ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color='black')
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray')
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth=3, color='black')
ax2.scatter(1, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
ax2.scatter(2, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
ax2.scatter(4, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_imag_sig, betas_sig_imag, marker='o', color='black')
ax2.scatter(x_imag_nonsig, betas_nonsig_imag, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_imag, mean_beta_imag], linewidth=3, color='black')
ax2.scatter(5, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
ax2.scatter(6, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
ax2.scatter(7, 0.34, s=65, marker=(5, 2), color='black')


labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Arousal'
labels[5] = 'Imageability'
labels[6] = 'M List'
labels[7] = 'M Pool'

ax2.set_xticklabels(labels, rotation=20, size=10)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .41)

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

plt.savefig("Final_Word_Recall_NoValenceJan8_.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""


"""