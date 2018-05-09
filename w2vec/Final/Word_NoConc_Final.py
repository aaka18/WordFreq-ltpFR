import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas

# Final Code For Word Recall No Concreteness

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob( '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
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

        x_values = np.column_stack((all_wordfreq_part, all_wordlen_part, all_valence_part_norm,
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
#print(fdr_pvalues.shape)

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
t_test_wordfreq = scipy.stats.ttest_1samp(params[:, 1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:, 2], 0)
t_test_valence = scipy.stats.ttest_1samp(params[:, 3], 0)
t_test_arousal = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)

fdr_correction = (multipletests([t_test_wordfreq[1], t_test_wordlen[1],
                                 t_test_valence[1], t_test_arousal[1],
                                 t_test_mlist[1], t_test_mpool[1]],
                                alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_wordfreq = params[:, 1]
y_wordlen = params[:, 2]
y_valence = params[:, 3]
y_arousal = params[:, 4]
y_mlist = params[:, 5]
y_mpool = params[:, 6]

beta_wordfreq = (np.array(params[:, 1]))
beta_wordlen = (np.array(params[:, 2]))
beta_valence = (np.array(params[:, 3]))
beta_arousal = (np.array(params[:, 4]))
beta_mlist = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))

mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

ave_betas = [ round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3), round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_wordfreq), 3),round(stats.sem(beta_wordlen), 3),
             round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Word Frequency', 'Word Length', 'Valence',
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


betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:, 1])):
    if fdr_pvalues[:, 1][i]:
        betas_sig_wordfreq.append(params[:, 1][i])
    else:
        betas_nonsig_wordfreq.append(params[:, 1][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:, 2])):
    if fdr_pvalues[:, 2][i]:
        betas_sig_wordlen.append(params[:, 2][i])
    else:
        betas_nonsig_wordlen.append(params[:, 2][i])

betas_sig_valence = []
betas_nonsig_valence = []
for i in range(len(params[:, 3])):
    if fdr_pvalues[:, 3][i]:
        betas_sig_valence.append(params[:, 3][i])
    else:
        betas_nonsig_valence.append(params[:, 3][i])

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

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 1)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 1)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 2)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 2)

x_valence_sig = np.full(len(betas_sig_valence), 3)
x_valence_nonsig = np.full(len(betas_nonsig_valence), 3)

x_arousal_sig = np.full(len(betas_sig_arousal), 4)
x_arousal_nonsig = np.full(len(betas_nonsig_arousal), 5)

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


ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([0.8, 1.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
if fdr_correction[0][0]:
    ax2.scatter(1, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')
if fdr_correction[0][1]:
    ax2.scatter(2, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
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
labels[1] = 'Frequency'
labels[2] = 'Length'
labels[3] = 'Valence'
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

plt.savefig("Final_Word_Recall_NoConc_Jan9FINAL.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
Significant R-Squareds [0.067615964915874738, 0.062967527868531215, 0.026298198126140093, 0.049745752522167619, 0.072149167031190675, 0.055013236679313904, 0.093475695392151059, 0.036984301035386413, 0.064405404939626876, 0.034871589080073928, 0.080604306606998133, 0.10109619868790132, 0.06299601540737787, 0.04201999161485459, 0.18739919264723692, 0.049489999714530275, 0.037604709041262452, 0.035065869116968851, 0.05805187962324343, 0.031254582274440446, 0.0306620552868897, 0.034289975151029051, 0.030119350200939587, 0.052504936365355581, 0.038874359428512584, 0.026944136071221703, 0.069012464771012194, 0.035158952833848733, 0.049186047339775496, 0.12676556249821158, 0.066784241430473901, 0.048571968599343096, 0.063662929243272659, 0.038200561722690818, 0.03086444147090095, 0.071952108451272312, 0.079259117512343269, 0.025297348600202274, 0.046943655942375706, 0.11207975227057543, 0.03868276443050811, 0.060287395701680513, 0.033311639470650634, 0.06359835847127393, 0.032986109724397972, 0.071879490544550451, 0.040397857541500493, 0.0239809564033584, 0.024184779539800805, 0.065820593646756076, 0.064545136073757892, 0.039785515066220523, 0.022157385943415431, 0.022545354320607558, 0.035699180478621639, 0.082773294329203218, 0.025039495203906914, 0.026876596680895259, 0.069145404776045249, 0.024834070653163676, 0.050919001361061977, 0.032409625938610298, 0.049319674864966356, 0.070464852700521052, 0.039675191601786008, 0.050246842940292358, 0.069304398928628497, 0.070019934091097746, 0.06646344705247309, 0.02699556761858346, 0.10602063515755933, 0.063850517505940174, 0.10810702462993049, 0.036378999665471756, 0.069955243556112823, 0.11206087196723369, 0.053525461906206662, 0.0286784167054982, 0.064047268560636805, 0.083297121868848523, 0.1324503174266155, 0.043626847188106499]
Not Significant R-Squareds [0.01733872027115313, 0.013161067074760391, 0.017909700103691528, 0.015971021196038993, 0.015888582176129873, 0.016044798280469008]
R sig 82
R not sig 6
(array([ True, False,  True,  True,  True,  True], dtype=bool), array([  4.88027940e-15,   2.97788806e-01,   4.96080126e-20,
         1.85193333e-23,   2.56656141e-02,   1.51186991e-30]), 0.008512444610847103, 0.008333333333333333)
T-test Intercepts Ttest_1sampResult(statistic=-9.2927088074279709, pvalue=1.1295015967408143e-14)
T-test WordFreq Ttest_1sampResult(statistic=9.5567596947069156, pvalue=3.2535195986775371e-15)
T-test WordLen Ttest_1sampResult(statistic=-1.0474599758820613, pvalue=0.29778880615468001)
T-test Valence Ttest_1sampResult(statistic=12.09529453776212, pvalue=2.4804006321227146e-20)
T-test Arousal Ttest_1sampResult(statistic=13.965110019521976, pvalue=6.1731110869082313e-24)
T-test MList Ttest_1sampResult(statistic=2.3434188582221243, pvalue=0.021388011714427429)
T-test MPool Ttest_1sampResult(statistic=18.149110364496938, pvalue=2.519783177078674e-31)
Beta WordFreq Ave, SE 0.061460387613 0.0064310906182
Beta Wordlen Ave, SE -0.00554434107053 0.00529312928244
Beta Valence Ave, SE 0.108406514724 0.00896270151877
Beta Arousal Ave, SE 0.130904721036 0.00937369779781
Beta MList Ave, SE 0.0108396631203 0.0046255764659
Beta MPool Ave, SE 0.12928222068 0.00712333652084
[0.060999999999999999, -0.0060000000000000001, 0.108, 0.13100000000000001, 0.010999999999999999, 0.129] [0.060999999999999999, -0.0060000000000000001, 0.108, 0.13100000000000001, 0.010999999999999999, 0.129]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
     Word Frequency      0.061     0.006
        Word Length     -0.006     0.005
            Valence      0.108     0.009
            Arousal      0.131     0.009
List Meaningfulness      0.011     0.005
Pool Meaningfulness      0.129     0.007
 
 
rsq [0.06761596491587474, 0.06296752786853121, 0.026298198126140093, 0.04974575252216762, 0.07214916703119068, 0.055013236679313904, 0.09347569539215106, 0.03698430103538641, 0.06440540493962688, 0.01733872027115313, 0.03487158908007393, 0.08060430660699813, 0.10109619868790132, 0.01316106707476039, 0.06299601540737787, 0.04201999161485459, 0.18739919264723692, 0.049489999714530275, 0.03760470904126245, 0.03506586911696885, 0.05805187962324343, 0.031254582274440446, 0.0306620552868897, 0.03428997515102905, 0.030119350200939587, 0.017909700103691528, 0.05250493636535558, 0.038874359428512584, 0.026944136071221703, 0.0690124647710122, 0.03515895283384873, 0.049186047339775496, 0.12676556249821158, 0.0667842414304739, 0.048571968599343096, 0.06366292924327266, 0.03820056172269082, 0.03086444147090095, 0.015971021196038993, 0.07195210845127231, 0.07925911751234327, 0.025297348600202274, 0.046943655942375706, 0.11207975227057543, 0.03868276443050811, 0.06028739570168051, 0.033311639470650634, 0.06359835847127393, 0.03298610972439797, 0.07187949054455045, 0.04039785754150049, 0.0239809564033584, 0.024184779539800805, 0.06582059364675608, 0.06454513607375789, 0.03978551506622052, 0.02215738594341543, 0.022545354320607558, 0.03569918047862164, 0.08277329432920322, 0.025039495203906914, 0.02687659668089526, 0.06914540477604525, 0.024834070653163676, 0.05091900136106198, 0.0324096259386103, 0.049319674864966356, 0.07046485270052105, 0.03967519160178601, 0.05024684294029236, 0.0693043989286285, 0.07001993409109775, 0.06646344705247309, 0.02699556761858346, 0.10602063515755933, 0.06385051750594017, 0.10810702462993049, 0.036378999665471756, 0.06995524355611282, 0.015888582176129873, 0.11206087196723369, 0.05352546190620666, 0.0286784167054982, 0.016044798280469008, 0.0640472685606368, 0.08329712186884852, 0.1324503174266155, 0.0436268471881065]

"""