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
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness = np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_valence = np.array(all_mean_valence_participants)
all_arousal = np.array(all_mean_arousal_participants)
all_means_mpool = np.array(all_means_mpool)
all_trial_nos = np.array(trial_nos)
all_session_nos = np.array(session_nos)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
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
        all_m_list_part = all_m_list[i]
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
            all_m_list_part = np.array(all_m_list_part)[mask]
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
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_valence_part_norm, all_arousal_part_norm,
                                    all_m_list_part_norm, all_means_mpool_part_norm, all_trial_nos_part_norm,
                                    all_session_nos_part_norm))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        y_value = all_probs_part_norm

        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()

        # print(results)
        # print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        residuals.append(results.resid)
        fdr_p = (multipletests(
            [results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4],
             results.pvalues[5], results.pvalues[6], results.pvalues[7], results.pvalues[8], results.pvalues[9]],
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
t_test_mlist = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 8], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 9], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_valence[1],
                     t_test_arousal[1], t_test_mlist[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test MList", t_test_mlist)
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
y_mlist = params[:, 6]
y_mpool = params[:, 7]
y_trial_no = params[:, 8]
y_session_no = params[:, 9]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))
beta_mpool = (np.array(params[:, 7]))
beta_trial_no = (np.array(params[:, 8]))
beta_session_no = (np.array(params[:, 9]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3), round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal', 'List Meaningfulness', 'Pool Meaningfulness', "Trial No",
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

betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:, 8])):
    if fdr_pvalues[:, 8][i]:
        betas_sig_trial.append(params[:, 8][i])
    else:
        betas_nonsig_trial.append(params[:, 8][i])

betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:, 9])):
    if fdr_pvalues[:, 9][i]:
        betas_sig_session.append(params[:, 9][i])
    else:
        betas_nonsig_session.append(params[:, 9][i])

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

x_trial_sig = np.full(len(betas_sig_trial), 8)
x_trial_nonsig = np.full(len(betas_nonsig_trial), 8)

x_session_sig = np.full(len(betas_sig_session), 9)
x_session_nonsig = np.full(len(betas_nonsig_session), 9)

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

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
if fdr_correction[0][5]:
    ax2.scatter(6, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
if fdr_correction[0][6]:
    ax2.scatter(7, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_trial_sig, betas_sig_trial, marker='o', color='black')
ax2.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
ax2.plot([7.8, 8.2], [mean_beta_trial_no, mean_beta_trial_no], linewidth=3, color='black')
if fdr_correction[0][7]:
    ax2.scatter(8, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_session_sig, betas_sig_session, marker='o', color='black')
ax2.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
ax2.plot([8.8, 9.2], [mean_beta_session_no, mean_beta_session_no], linewidth=3, color='black')
if fdr_correction[0][8]:
    ax2.scatter(9, 0.58, s=65, marker=(5, 2), color='black')


my_xticks = ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No']
ax2.set_xticks(range(1,10))
ax2.set_xticklabels(my_xticks, rotation=16.5, size=14)
ax2.set_xlim(0, 10)
ax2.set_ylabel("Beta Value", size = 16)

# ax2.set_xticks(range(10), ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No'], rotation=45)

ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)

ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color='black', label=" p < 0.05")
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label="Not Significant")
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("All participants", size = 14)
ax3.set_ylabel("R-Squared Value", size = 16)
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995, 1.005], [mean_combined, mean_combined], linewidth=3, color='black')

plt.savefig("Final_List_Recall_CEMS.pdf")
plt.show()
"""
(88,)
Significant R-Squareds [0.1330469885528307, 0.15627121212831874, 0.10401486981793451, 0.080302902321659486, 0.052741022979099217, 0.061651390888436297, 0.23048051010724557, 0.065801168327984438, 0.17018462851561977, 0.084676107887849184, 0.19030187406297183, 0.072581161382151427, 0.091415993848317112, 0.20876590289745478, 0.08411316284713477, 0.13702202383865858, 0.044635532135464118, 0.066432619808533211, 0.20796328203474079, 0.078199599231113859, 0.15196654939708421, 0.13203104308921798, 0.14371839318648594, 0.032418177816486593, 0.15865962638748576, 0.079460065061540908, 0.2439817614467501, 0.18256424489637402, 0.10629317827835627, 0.19205494173937532, 0.28483278446492954, 0.12719735151662337, 0.04955338769781592, 0.034940824727050401, 0.037208612493811244, 0.091845882089011899, 0.11949526342796779, 0.1256910763075596, 0.24153227574350655, 0.35149469789976584, 0.15496028533585071, 0.13970702554571335, 0.061578436617585108, 0.064223580331057128, 0.15045556762169632, 0.29934290790202778, 0.35061599851771896, 0.18505174830437776, 0.15315052730399825, 0.042484990867583505, 0.1437953422666739, 0.05260112175227416, 0.12611118953766476, 0.21971965487823686, 0.1104012160146004, 0.12953067811444186, 0.063228440155895127, 0.37964069738645112, 0.078761800318436381, 0.080160683310834258, 0.15507531733790192, 0.055902255478507912, 0.047718265441619945, 0.12466485527086579, 0.23447359150504909, 0.15099783837569269, 0.092969461119949748, 0.11323857481993882, 0.13504963443736762, 0.24726177374733882, 0.2251408378835742, 0.18919230097096584, 0.080877455782202068, 0.14654249710586598, 0.038657680724658472]
Not Significant R-Squareds [0.020486932050903017, 0.013363346929696984, 0.02884150915700523, 0.0087907145341826753, 0.019566251961292025, 0.018583858508575135, 0.027110947355233339, 0.017655084060472559, 0.028533384925128802, 0.015354785724180586, 0.021398874414903957, 0.016563218197356289, 0.024556655323412335]
(array([False,  True, False,  True, False,  True,  True,  True, False], dtype=bool), array([  1.57629657e-01,   9.90637299e-04,   2.25015937e-01,
         1.00684978e-03,   4.97482212e-01,   1.19618038e-14,
         1.00684978e-03,   1.98199152e-25,   3.24447093e-01]), 0.0056830449880480582, 0.005555555555555556)
T-test Intercepts Ttest_1sampResult(statistic=-0.25883593256349779, pvalue=0.7963732458730024)
T-test Concreteness Ttest_1sampResult(statistic=-1.6377475714062124, pvalue=0.10508643800732309)
T-test WordFreq Ttest_1sampResult(statistic=3.7389610933377977, pvalue=0.00033021243303855963)
T-test WordLen Ttest_1sampResult(statistic=-1.3674318086038453, pvalue=0.17501239569316035)
T-test Valence Ttest_1sampResult(statistic=3.5829613134703466, pvalue=0.00055936098982867211)
T-test Arousal Ttest_1sampResult(statistic=0.68131428131332217, pvalue=0.49748221210630206)
T-test MList Ttest_1sampResult(statistic=9.5996608180081768, pvalue=2.6581786286623814e-15)
T-test MPool Ttest_1sampResult(statistic=-3.5921228385035953, pvalue=0.00054252295775277491)
T-test Trial No Ttest_1sampResult(statistic=-15.292103330186288, pvalue=2.2022128040262889e-26)
T-test Session No Ttest_1sampResult(statistic=-1.0681730683613502, pvalue=0.2883974158092576)
Beta Conc Ave, SE -0.00756954947085 0.00462192684819
Beta WordFreq Ave, SE 0.0177776765559 0.00475471022887
Beta Wordlen Ave, SE -0.00501049058461 0.00366416120576
Beta Valence Ave, SE 0.0194937267796 0.00544067464707
Beta Arousal Ave, SE 0.00396453680241 0.0058189544989
Beta MList Ave, SE 0.0513837542371 0.0053526635171
Beta MPool Ave, SE -0.0200616260138 0.00558489420204
Beta Trial No Ave, SE -0.187431206821 0.012256731646
Beta Session No Ave, SE -0.027347287633 0.0256019257956
[-0.0080000000000000002, 0.017999999999999999, -0.0050000000000000001, 0.019, 0.0040000000000000001, 0.050999999999999997, -0.02, -0.187, -0.027] [-0.0080000000000000002, 0.017999999999999999, -0.0050000000000000001, 0.019, 0.0040000000000000001, 0.050999999999999997, -0.02, -0.187, -0.027]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.008     0.005
     Word Frequency      0.018     0.005
        Word Length     -0.005     0.004
            Valence      0.019     0.005
            Arousal      0.004     0.006
List Meaningfulness      0.051     0.005
Pool Meaningfulness      -0.02     0.006
           Trial No     -0.187     0.012
         Session No     -0.027     0.026
 
 
 
 """