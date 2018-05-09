import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, MPool_byPoolSim as m_pool, \
    P_rec_ltpFR2_by_list as p_rec, Session_and_list_no_list as session_nos, semantic_similarity_bylist as m_list

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants, \
all_mean_valence_participants, all_mean_arousal_participants = conc_freq_len.measures_by_list(
    files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness = np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_valence = np.array(all_mean_valence_participants)
all_arousal = np.array(all_mean_arousal_participants)
all_means_mpool = np.array(all_means_mpool)
all_trial_nos = np.array(trial_nos)
all_session_nos = np.array(session_nos)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_means_mpool_part = []
all_valence_part = []
all_arousal_part = []
all_trial_nos_part = []
all_session_nos_part = []

params = []
rsquareds = []
pvalues = []
predict = []
residuals = []
model_pvalues = []
fdr_pvalues = []


def calculate_params():
    # for each participant (= each list in list of lists)
    # get the list values f
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordfreq_part = all_wordfreq[i]
        all_wordlen_part = all_wordlen[i]
        all_valence_part = all_valence[i]
        all_arousal_part = all_arousal[i]
        all_means_mpool_part = all_means_mpool[i]
        all_trial_nos_part = all_trial_nos[i]
        all_session_nos_part = all_session_nos[i]

        # mask them since some p_recs have nan's

        mask = np.isfinite(all_probs_part)
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part = np.array(all_probs_part)[mask]
        try:
            all_concreteness_part = np.array(all_concreteness_part)[mask]
            all_wordfreq_part = np.array(all_wordfreq_part)[mask]
            all_wordlen_part = np.array(all_wordlen_part)[mask]
            all_valence_part = np.array(all_valence_part)[mask]
            all_arousal_part = np.array(all_arousal_part)[mask]
            all_means_mpool_part = np.array(all_means_mpool_part)[mask]
            all_trial_nos_part = np.array(all_trial_nos_part)[mask]
            all_session_nos_part = np.array(all_session_nos_part)[mask]
        except:
            pass

        all_probs_part_norm = stats.mstats.zscore(all_probs_part, ddof=1)
        # print(all_probs_part_norm)
        all_concreteness_part_norm = stats.mstats.zscore(all_concreteness_part, ddof=1)
        # print("Conc", all_concreteness_part_norm)
        all_wordfreq_part_norm = stats.mstats.zscore(all_wordfreq_part, ddof=1)
        all_wordlen_part_norm = stats.mstats.zscore(all_wordlen_part, ddof=1)
        all_valence_part_norm = stats.mstats.zscore(all_valence_part, ddof=1)
        all_arousal_part_norm = stats.mstats.zscore(all_arousal_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_valence_part_norm, all_arousal_part_norm,
                                    all_means_mpool_part_norm, all_trial_nos_part_norm,
                                    all_session_nos_part_norm))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        y_value = all_probs_part_norm

        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()

        # print(results)
        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        residuals.append(results.resid)
        fdr_p = (multipletests(
            [results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4],
             results.pvalues[5], results.pvalues[6], results.pvalues[7], results.pvalues[8]],
            alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
        fdr_pvalues.append(fdr_p[0])
        print(results.f_pvalue)
        model_pvalues.append(results.f_pvalue)

    return params, rsquareds, predict, pvalues, residuals, model_pvalues, fdr_pvalues


params, rsquareds, predict, pvalues, residuals, model_pvalues, fdr_pvalues = calculate_params()

rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
model_pvalues = np.array(model_pvalues)
print(model_pvalues)
print(model_pvalues.shape)

residuals = np.array(residuals)

rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if model_pvalues[i] < .05:
        rsquareds_sig.append(rsquareds[i])
    else:
        rsquareds_notsig.append(rsquareds[i])
print("Significant R-Squareds", rsquareds_sig)
print("Not Significant R-Squareds", rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

t_test_intercepts = scipy.stats.ttest_1samp(params[:, 0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:, 1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:, 2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:, 3], 0)
t_test_valence = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_arousal = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 8], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_valence[1],
                     t_test_arousal[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test MPool", t_test_mpool)
print("T-test Trial No", t_test_trial_no)
print("T-test Session No", t_test_session_no)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_valence = params[:, 4]
y_arousal = params[:, 5]
y_mpool = params[:, 6]
y_trial_no = params[:, 7]
y_session_no = params[:, 8]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))
beta_trial_no = (np.array(params[:, 7]))
beta_session_no = (np.array(params[:, 8]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3), round(np.mean(beta_arousal), 3),
              round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
              round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal',  'Pool Meaningfulness', "Trial No",
              "Session No", ]
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names=("List Recall Model", "Mean Betas", "SEM Betas"),
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

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_mpool.append(params[:, 6][i])
    else:
        betas_nonsig_mpool.append(params[:, 6][i])

betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:, 7])):
    if fdr_pvalues[:, 7][i]:
        betas_sig_trial.append(params[:, 7][i])
    else:
        betas_nonsig_trial.append(params[:, 7][i])

betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:, 8])):
    if fdr_pvalues[:, 8][i]:
        betas_sig_session.append(params[:, 8][i])
    else:
        betas_nonsig_session.append(params[:, 8][i])

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

x_mpool_sig = np.full(len(betas_sig_mpool), 6)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 6)

x_trial_sig = np.full(len(betas_sig_trial), 7)
x_trial_nonsig = np.full(len(betas_nonsig_trial), 7)

x_session_sig = np.full(len(betas_sig_session), 8)
x_session_nonsig = np.full(len(betas_nonsig_session), 8)

from matplotlib import gridspec

fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

ax2.axhline(y=0, color='gray', linestyle='--')

ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color='black')
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray')
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth=3, color='black')
if fdr_correction[0][0]:
    ax2.scatter(1, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
if fdr_correction[0][1]:
    ax2.scatter(2, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')
if fdr_correction[0][2]:
    ax2.scatter(3, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
if fdr_correction[0][3]:
    ax2.scatter(4, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
if fdr_correction[0][4]:
    ax2.scatter(5, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
if fdr_correction[0][5]:
    ax2.scatter(6, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_trial_sig, betas_sig_trial, marker='o', color='black')
ax2.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_trial_no, mean_beta_trial_no], linewidth=3, color='black')
if fdr_correction[0][6]:
    ax2.scatter(7, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_session_sig, betas_sig_session, marker='o', color='black')
ax2.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
ax2.plot([7.8, 8.2], [mean_beta_session_no, mean_beta_session_no], linewidth=3, color='black')
if fdr_correction[0][7]:
    ax2.scatter(8, 0.58, s=65, marker=(5, 2), color='black')



my_xticks = ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MPool', 'Trial No', 'Session No']
#my_xticks = ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No']
ax2.set_xticks(range(1,9))
ax2.set_xticklabels(my_xticks, rotation=14, size=13)
ax2.set_xlim(0, 9)
# ax2.set_xticks(range(10), ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No'], rotation=45)

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

plt.savefig("Final_List_Recall_Final_NoMListJan9.pdf")
plt.show()
"""
(88,)
Significant R-Squareds [0.12816843257649557, 0.14337153868475727, 0.10368260831048637, 0.066231700097787671, 0.051724630555768703, 0.05753412781765288, 0.22683343488016527, 0.060288164062162197, 0.16919281267215114, 0.084395039754070478, 0.18162857719888681, 0.072507048704801069, 0.086852382372731185, 0.2080568196325201, 0.074595237899083511, 0.13670295213684203, 0.044627505392014988, 0.065741140927961461, 0.20796096177397161, 0.077225791190438819, 0.14732572735557758, 0.13150539755935886, 0.14240000592291424, 0.032006800050662698, 0.14433721842493552, 0.075130532063916955, 0.2394306390275488, 0.18010799791339882, 0.10153695874252433, 0.18928352251826686, 0.2805480789321646, 0.1227333700826666, 0.046592284723123933, 0.034773130639489502, 0.028789397014129192, 0.034400350057923412, 0.091107342030919858, 0.11711504871579037, 0.12538452405569833, 0.23761359462900622, 0.35144361010199709, 0.15473280610388418, 0.13948838420272958, 0.051138521805884407, 0.064165620079095653, 0.13896010068961984, 0.29853381839779414, 0.34673005777525367, 0.18501400093429021, 0.15274517715324287, 0.042483630545228301, 0.13932885720512878, 0.050706034573728576, 0.12465388841423086, 0.21135165716651294, 0.095522799552956505, 0.12803072897135703, 0.059550994471753826, 0.37853049481864365, 0.069839660883145194, 0.080159959225047595, 0.028506656516315454, 0.14936476946100008, 0.05560515190105797, 0.044295009725427104, 0.12440265074858248, 0.22980692940585001, 0.14761617349626543, 0.083777817780473773, 0.11276851346270633, 0.13443214460167663, 0.24552457448313969, 0.22353105964766495, 0.18833161681239563, 0.08013840722846699, 0.14398944755446075, 0.031885714483747662]
Not Significant R-Squareds [0.01969440255028998, 0.013313766728445198, 0.0075394762651810332, 0.016435335128892836, 0.016085569530268851, 0.026968324558514944, 0.017496211798666006, 0.013595011920505717, 0.019725158152514077, 0.013041560256121465, 0.024548868952067315]
(array([False,  True, False,  True, False, False,  True, False], dtype=bool), array([  8.28777344e-01,   2.78109603e-03,   2.36024468e-01,
         2.78109603e-03,   7.31817229e-01,   1.30026019e-01,
         1.69494633e-25,   3.81926887e-01]), 0.0063911509545450107, 0.00625)
T-test Intercepts Ttest_1sampResult(statistic=-0.23771341916051034, pvalue=0.8126623045970941)
T-test Concreteness Ttest_1sampResult(statistic=-0.21692157347893165, pvalue=0.82877734383281321)
T-test WordFreq Ttest_1sampResult(statistic=3.3928652804990813, pvalue=0.001042911010671217)
T-test WordLen Ttest_1sampResult(statistic=-1.4613742072749045, pvalue=0.1475152924486694)
T-test Valence Ttest_1sampResult(statistic=3.4364356024031197, pvalue=0.00090587088094783142)
T-test Arousal Ttest_1sampResult(statistic=0.46886335956528125, pvalue=0.64034007563952478)
T-test MPool Ttest_1sampResult(statistic=1.8688189377395834, pvalue=0.065013009345575079)
T-test Trial No Ttest_1sampResult(statistic=-15.301391492583305, pvalue=2.1186829152068122e-26)
T-test Session No Ttest_1sampResult(statistic=-1.0725365919105689, pvalue=0.28644516488475308)
Beta Conc Ave, SE -0.00101945049571 0.0046996270558
Beta WordFreq Ave, SE 0.0158472351274 0.00467075283491
Beta Wordlen Ave, SE -0.00534107095736 0.00365482771679
Beta Valence Ave, SE 0.0187481907552 0.00545570845038
Beta Arousal Ave, SE 0.00270610778843 0.00577163417277
Beta MPool Ave, SE 0.00843258708432 0.00451225472625
Beta Trial No Ave, SE -0.187373561218 0.0122455242917
Beta Session No Ave, SE -0.0274517782748 0.0255951903943
[-0.001, 0.016, -0.0050000000000000001, 0.019, 0.0030000000000000001, 0.0080000000000000002, -0.187, -0.027] [-0.001, 0.016, -0.0050000000000000001, 0.019, 0.0030000000000000001, 0.0080000000000000002, -0.187, -0.027]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.001     0.005
     Word Frequency      0.016     0.005
        Word Length     -0.005     0.004
            Valence      0.019     0.005
            Arousal      0.003     0.006
Pool Meaningfulness      0.008     0.005
           Trial No     -0.187     0.012
         Session No     -0.027     0.026
 
 """