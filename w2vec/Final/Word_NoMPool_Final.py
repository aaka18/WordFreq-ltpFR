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

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence_norm)
all_arousal = np.array(arousal_norm)
all_m_list = np.array(m_list)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_arousal_part = []
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
        all_valence_part = all_valence
        all_arousal_part = all_arousal
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)

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
                                    all_arousal_part_norm, all_m_list_part_norm))

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
                               alpha=0.05,
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

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                                 t_test_valence[1], t_test_arousal[1],
                                 t_test_mlist[1]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test MList", t_test_mlist)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_valence = params[:, 4]
y_arousal = params[:, 5]
y_mlist = params[:, 6]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3), round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal', 'List Meaningfulness']
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

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Valence'
labels[5] = 'Arousal'
labels[6] = 'M List'

ax2.set_xticklabels(labels, rotation=18, size=13)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .41)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)


ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color='black', label=" p < 0.05")
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label="Not Significant")
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("R-Squared Value", size=14)
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995, 1.005], [mean_combined, mean_combined], linewidth=3, color='black')

plt.savefig("Final_Word_Recall_NoMPoolJan9FINAL.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
(88, 7)
Significant R-Squareds [0.040724874979710224, 0.047176442688557696, 0.02308423108139579, 0.071032751388184656, 0.039522527766563731, 0.088939898280923635, 0.024542847413751567, 0.049095061546966412, 0.033423063179180867, 0.079890037805425917, 0.069899752460051645, 0.0255086943339351, 0.046625421820485569, 0.14324202250149976, 0.044531248220045283, 0.038338059402241553, 0.048712443820915441, 0.025853381805355236, 0.025604145011089874, 0.042266203427899884, 0.022833544261387284, 0.023795470729221835, 0.040601150696770949, 0.024419998879278371, 0.032678157292676424, 0.060452762126413151, 0.03426428507479562, 0.041400617552119545, 0.052131132861952256, 0.042355050082106382, 0.022381218706798878, 0.045901542748331137, 0.08192149415859451, 0.026700226261478277, 0.044415701828281584, 0.070548529650480751, 0.037767020534289042, 0.043793626687783749, 0.027722734243645575, 0.061057004624769817, 0.035377838634488179, 0.033593603336825151, 0.031803240061576421, 0.04331821883090714, 0.026161759397163697, 0.082314024553595821, 0.064967765202354699, 0.041878463720653736, 0.025731210907286584, 0.050928606651934105, 0.032032111612568137, 0.049736074394737706, 0.0469476873178023, 0.040309595509578022, 0.055058836032186642, 0.023259903750213806, 0.09507553196590024, 0.061813992936572859, 0.10222780393226605, 0.025179299181066317, 0.055201701909519896, 0.10684915377086901, 0.028338650234522311, 0.040812035804345781, 0.081784109141565309, 0.082467600630693694, 0.036986998216070766]
Not Significant R-Squareds [0.015472187859307374, 0.011165256893545084, 0.012665099058573936, 0.016998816858791566, 0.011318125031435033, 0.020221786791090768, 0.017733111157134274, 0.021082704121550999, 0.017701599703040971, 0.020523713303854829, 0.018385800987666401, 0.021335640372441289, 0.018144282469875317, 0.011521572032581084, 0.0050935382661234652, 0.013997224601072111, 0.021891057339081033, 0.019680703597462879, 0.0086852608659466757, 0.012023963494689482, 0.018673896089243192]
R sig 67
R not sig 21
(array([ True,  True, False,  True,  True,  True], dtype=bool), array([  1.53906755e-11,   2.24452904e-10,   4.38064052e-01,
         2.15990848e-20,   3.18500872e-22,   2.48754937e-12]), 0.008512444610847103, 0.008333333333333333)
T-test Intercepts Ttest_1sampResult(statistic=-7.323863089214913, pvalue=1.1458957235856418e-10)
T-test Concreteness Ttest_1sampResult(statistic=7.8439330480575169, pvalue=1.0260450361480293e-11)
T-test WordFreq Ttest_1sampResult(statistic=7.2174242878752652, pvalue=1.8704408681399753e-10)
T-test WordLen Ttest_1sampResult(statistic=-0.77905027690565687, pvalue=0.43806405248619185)
T-test Valence Ttest_1sampResult(statistic=12.368573852272913, pvalue=7.1996949207438165e-21)
T-test Arousal Ttest_1sampResult(statistic=13.471516071552333, pvalue=5.3083478697781168e-23)
T-test MList Ttest_1sampResult(statistic=8.2946321211374396, pvalue=1.243774682889004e-12)
Beta Conc Ave, SE 0.0478431631819 0.00609938444003
Beta WordFreq Ave, SE 0.0466075265866 0.00645763983488
Beta Wordlen Ave, SE -0.00411198424399 0.00527820137658
Beta Valence Ave, SE 0.111008951656 0.00897508095772
Beta Arousal Ave, SE 0.124263981059 0.00922420167107
Beta MList Ave, SE 0.0421145263876 0.00507732299305
[0.048000000000000001, 0.047, -0.0040000000000000001, 0.111, 0.124, 0.042000000000000003] [0.048000000000000001, 0.047, -0.0040000000000000001, 0.111, 0.124, 0.042000000000000003]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness      0.048     0.006
     Word Frequency      0.047     0.006
        Word Length     -0.004     0.005
            Valence      0.111     0.009
            Arousal      0.124     0.009
List Meaningfulness      0.042     0.005
 
 
rsq [0.040724874979710224, 0.047176442688557696, 0.015472187859307374, 0.02308423108139579, 0.07103275138818466, 0.03952252776656373, 0.08893989828092363, 0.024542847413751567, 0.04909506154696641, 0.011165256893545084, 0.03342306317918087, 0.07989003780542592, 0.06989975246005165, 0.012665099058573936, 0.0255086943339351, 0.04662542182048557, 0.14324202250149976, 0.04453124822004528, 0.03833805940224155, 0.016998816858791566, 0.04871244382091544, 0.011318125031435033, 0.020221786791090768, 0.025853381805355236, 0.025604145011089874, 0.017733111157134274, 0.042266203427899884, 0.022833544261387284, 0.023795470729221835, 0.04060115069677095, 0.02441999887927837, 0.032678157292676424, 0.06045276212641315, 0.03426428507479562, 0.041400617552119545, 0.052131132861952256, 0.04235505008210638, 0.022381218706798878, 0.021082704121551, 0.04590154274833114, 0.08192149415859451, 0.026700226261478277, 0.044415701828281584, 0.07054852965048075, 0.03776702053428904, 0.04379362668778375, 0.027722734243645575, 0.06105700462476982, 0.01770159970304097, 0.03537783863448818, 0.02052371330385483, 0.0183858009876664, 0.03359360333682515, 0.02133564037244129, 0.03180324006157642, 0.04331821883090714, 0.018144282469875317, 0.026161759397163697, 0.011521572032581084, 0.08231402455359582, 0.005093538266123465, 0.013997224601072111, 0.0649677652023547, 0.021891057339081033, 0.041878463720653736, 0.025731210907286584, 0.01968070359746288, 0.050928606651934105, 0.03203211161256814, 0.049736074394737706, 0.0469476873178023, 0.04030959550957802, 0.05505883603218664, 0.023259903750213806, 0.09507553196590024, 0.06181399293657286, 0.10222780393226605, 0.025179299181066317, 0.055201701909519896, 0.008685260865946676, 0.10684915377086901, 0.012023963494689482, 0.02833865023452231, 0.01867389608924319, 0.04081203580434578, 0.08178410914156531, 0.0824676006306937, 0.036986998216070766]

"""