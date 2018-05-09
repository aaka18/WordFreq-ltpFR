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
imag_norm = conc_freq_len.imageability_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence_norm)
all_arousal = np.array(arousal_norm)
all_imageability = np.array(imag_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)


all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
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
        all_valence_part = all_valence
        all_arousal_part = all_arousal
        all_imag_part = all_imageability
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        # print(all_imageability)
        mask = np.logical_not(np.isnan(all_imageability))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part_norm = np.array(all_valence_part)[mask]
        all_arousal_part_norm = np.array(all_arousal_part)[mask]
        all_imag_part_norm = np.array(all_imag_part)[mask]
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
                                    all_arousal_part_norm, all_imag_part_norm, all_m_list_part_norm, all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3],                  results.pvalues[4], results.pvalues[5], results.pvalues[6], results.pvalues[7], results.pvalues[8]], alpha=0.05,         method='fdr_bh', is_sorted=False, returnsorted=False))

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
t_test_imag = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 8], 0)


print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                     t_test_valence[1], t_test_arousal[1], t_test_imag[1],
                     t_test_mlist[1], t_test_mpool[1]],
                    alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
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
y_valence = params[:, 4]
y_arousal = params[:, 5]
y_imag = params[:, 6]
y_mlist = params[:, 7]
y_mpool = params[:, 8]


beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_imag = (np.array(params[:, 6]))
beta_mlist = (np.array(params[:, 7]))
beta_mpool = (np.array(params[:, 8]))


mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_imag = np.mean(beta_imag)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)


print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta Imag Ave, SE", np.mean(beta_imag), stats.sem(beta_imag))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))


ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3) , round(np.mean(beta_arousal), 3), round(np.mean(beta_imag), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),                       round(stats.sem(beta_imag), 3),  round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal', 'Imageability','List Meaningfulness', 'Pool Meaningfulness', ]
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

betas_sig_imag = []
betas_nonsig_imag = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_imag.append(params[:, 6][i])
    else:
        betas_nonsig_imag.append(params[:, 6][i])


betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:, 7])):
    if fdr_pvalues[:, 7][i]:
        betas_sig_mlist.append(params[:, 7][i])
    else:
        betas_nonsig_mlist.append(params[:, 7][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 8])):
    if fdr_pvalues[:, 8][i]:
        betas_sig_mpool.append(params[:, 8][i])
    else:
        betas_nonsig_mpool.append(params[:, 8][i])


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

x_imag_sig = np.full(len(betas_sig_imag), 6)
x_imag_nonsig = np.full(len(betas_nonsig_imag), 6)

x_mlist_sig = np.full(len(betas_sig_mlist), 7)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 7)

x_mpool_sig = np.full(len(betas_sig_mpool), 8)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 8)

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

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
ax2.scatter(4, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
ax2.scatter(5, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_imag_sig, betas_sig_imag, marker='o', color='black')
ax2.scatter(x_imag_nonsig, betas_nonsig_imag, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_imag, mean_beta_imag], linewidth=3, color='black')
ax2.scatter(6, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
ax2.scatter(7, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([7.8, 8.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
ax2.scatter(8, 0.34, s=65, marker=(5, 2), color='black')


labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Valence'
labels[5] = 'Arousal'
labels[6] = 'Imageability'
labels[7] = 'M List'
labels[8] = 'M Pool'


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


plt.savefig("Final_Word_Recall_Jan9_.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
(87, 9)
Significant R-Squareds [0.102127402372167, 0.08174646335017477, 0.12723192866950994, 0.072515768448813467, 0.061747273867734931, 0.14973418778793168, 0.046670730797929494, 0.11120929489491993, 0.056071580681575695, 0.1145200750256522, 0.12968916563037525, 0.076010322842440137, 0.1002338326264639, 0.22236710309920615, 0.097508458301785117, 0.061702244209691615, 0.058379391995700858, 0.12014811586467, 0.074774993502867426, 0.069970725752427576, 0.047531078987732123, 0.10783203764254135, 0.056760247761851179, 0.090695592237702738, 0.17555635343216114, 0.13081063859280884, 0.077592152125211311, 0.078678800050318798, 0.054364934902545459, 0.055996706119660233, 0.083603109349733584, 0.094921975669427838, 0.082830928677026128, 0.12509527990441394, 0.060986775225896461, 0.07226672624419006, 0.093379956894801164, 0.052726897251709759, 0.13494296657101335, 0.084662344177218407, 0.048360451274053418, 0.10128874426926049, 0.094282459128791296, 0.07498434786682262, 0.11110180318453355, 0.087793026185425616, 0.075045721274077715, 0.069120732112891381, 0.052643163313695229, 0.095139483401694425, 0.070186701232503923, 0.049555155452683519, 0.092598651303576252, 0.074743251543472056, 0.11289886928079718, 0.060743161434542947, 0.13640886960391352, 0.062200214584051672, 0.11605727874420468, 0.069045819696242572, 0.10510268545224144, 0.068850690225807587, 0.12890678710780445, 0.10101611342158956, 0.11074496919767352, 0.10980356656871648, 0.17807831308940203]
Not Significant R-Squareds [0.041128310368330978, 0.018700177578978572, 0.035331445348829482, 0.031190009232984539, 0.024536847271422912, 0.044687562323280638, 0.046118957840569519, 0.039004757399439316, 0.041517467089032034, 0.032331986464369966, 0.035658003992623533, 0.045440237155535246, 0.03592573322728887, 0.035707799896883663, 0.027225866096825979, 0.043962933784021163, 0.046059200090106223, 0.032932159597685917, 0.044454476636701989, 0.034418753111569611]
R sig 67
R not sig 20
(array([ True,  True,  True,  True,  True,  True, False,  True], dtype=bool), array([  1.74397929e-13,   2.51668184e-11,   3.41007558e-05,
         1.42149560e-09,   3.06204897e-12,   2.91877640e-25,
         7.41879547e-02,   8.01144917e-28]), 0.0063911509545450107, 0.00625)
T-test Intercepts Ttest_1sampResult(statistic=-1.5557821607106588, pvalue=0.12343245927308004)
T-test Concreteness Ttest_1sampResult(statistic=-8.9393586963179246, pvalue=6.5399223456886993e-14)
T-test WordFreq Ttest_1sampResult(statistic=7.7654717711721961, pvalue=1.5729261524792733e-11)
T-test WordLen Ttest_1sampResult(statistic=-4.4093084123703621, pvalue=2.9838161362189111e-05)
T-test Valence Ttest_1sampResult(statistic=6.8458240977351164, pvalue=1.0661216972917205e-09)
T-test Arousal Ttest_1sampResult(statistic=8.2659128911897888, pvalue=1.5310244827357368e-12)
T-test Imag Ttest_1sampResult(statistic=15.07384374216182, pvalue=7.296940997453012e-26)
T-test MList Ttest_1sampResult(statistic=1.8074544885965202, pvalue=0.074187954698108627)
T-test MPool Ttest_1sampResult(statistic=16.700076831188877, pvalue=1.0014311457009814e-28)
Beta Conc Ave, SE -0.0674947964167 0.0075502951285
Beta WordFreq Ave, SE 0.0478282389751 0.0061590899284
Beta Wordlen Ave, SE -0.0287435098957 0.00651882499647
Beta Valence Ave, SE 0.0634592112163 0.00926976946972
Beta Arousal Ave, SE 0.0788947070985 0.00954458486765
Beta Imag Ave, SE 0.124054309923 0.00822977284656
Beta MList Ave, SE 0.0111327341217 0.00615934408965
Beta MPool Ave, SE 0.136807475302 0.00819202670052
[-0.067000000000000004, 0.048000000000000001, -0.029000000000000001, 0.063, 0.079000000000000001, 0.124, 0.010999999999999999, 0.13700000000000001] [-0.067000000000000004, 0.048000000000000001, -0.029000000000000001, 0.063, 0.079000000000000001, 0.124, 0.010999999999999999, 0.13700000000000001]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.067***     0.008
     Word Frequency      0.048***     0.006
        Word Length     -0.029***     0.007
            Valence      0.063***     0.009
            Arousal      0.079***      0.01
       Imageability      0.124***     0.008
List Meaningfulness      0.011     0.006
Pool Meaningfulness      0.137***     0.008


Regression Analysis for Variables Predicting Probability of Recall

Word Recall Model I Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.016*       0.007
     Word Frequency      0.056***     0.006
        Word Length     -0.005     0.005
List Meaningfulness       0.01*     0.005
Pool Meaningfulness      0.128***     0.008
            Valence      0.105***     0.009
            Arousal      0.131***     0.009
       Imageability      0.041***     0.005 


"""