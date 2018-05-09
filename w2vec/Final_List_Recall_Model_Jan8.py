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
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants,\
    all_mean_valence_participants, all_mean_arousal_participants, all_mean_imag_participants = conc_freq_len.measures_by_list(
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
all_imag = np.array(all_mean_imag_participants)
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
all_imag_part = []
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
        all_imag_part = all_imag[i]
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
            all_imag_part = np.array(all_imag_part)[mask]
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
        all_imag_part_norm = stats.mstats.zscore(all_imag_part, ddof=1)
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_valence_part_norm, all_arousal_part_norm, all_imag_part_norm,
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
        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        residuals.append(results.resid)
        fdr_p = (multipletests(
            [results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4],
             results.pvalues[5], results.pvalues[6], results.pvalues[7], results.pvalues[8], results.pvalues[9],
             results.pvalues[10] ], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
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
t_test_imag = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 8], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 9], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 10], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])
print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_valence[1],
                     t_test_arousal[1], t_test_imag[1], t_test_mlist[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test Arousal", t_test_arousal)
print("T-test Imag", t_test_imag)
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
y_imag = params[:, 6]
y_mlist = params[:, 7]
y_mpool = params[:, 8]
y_trial_no = params[:, 9]
y_session_no = params[:, 10]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_imag = (np.array(params[:, 6]))
beta_mlist = (np.array(params[:, 7]))
beta_mpool = (np.array(params[:, 8]))
beta_trial_no = (np.array(params[:, 9]))
beta_session_no = (np.array(params[:, 10]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_imag = np.mean(beta_imag)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta Imag Ave, SE", np.mean(beta_imag), stats.sem(beta_imag))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
            round(np.mean(beta_valence), 3) , round(np.mean(beta_arousal), 3), round(np.mean(beta_imag), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3),  round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3), round(stats.sem(beta_imag), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal', 'Imageability', 'List Meaningfulness', 'Pool Meaningfulness', "Trial No",
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

betas_sig_imag = []
betas_nonsig_imag = []
for i in range(len(params[:,6 ])):
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

betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:, 9])):
    if fdr_pvalues[:, 9][i]:
        betas_sig_trial.append(params[:, 9][i])
    else:
        betas_nonsig_trial.append(params[:, 9][i])

betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:, 10])):
    if fdr_pvalues[:, 10][i]:
        betas_sig_session.append(params[:, 10][i])
    else:
        betas_nonsig_session.append(params[:, 10][i])

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

x_trial_sig = np.full(len(betas_sig_trial), 9)
x_trial_nonsig = np.full(len(betas_nonsig_trial), 9)

x_session_sig = np.full(len(betas_sig_session), 10)
x_session_nonsig = np.full(len(betas_nonsig_session), 10)

from matplotlib import gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

ax2.axhline(y=0, color='gray', linestyle='--')

ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth = 3, color = 'black' )
ax2.scatter(1, 0.58, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq],  linewidth = 3, color = 'black' )
ax2.scatter(2, 0.58, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen],  linewidth = 3, color = 'black' )

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
ax2.scatter(4, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')

ax2.scatter(x_imag_sig, betas_sig_imag, marker='o', color='black')
ax2.scatter(x_imag_nonsig, betas_nonsig_imag, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_imag, mean_beta_imag], linewidth=3, color='black')
ax2.scatter(6, 0.58, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
ax2.plot([6.8, 7.2], [mean_beta_mlist, mean_beta_mlist],  linewidth = 3, color = 'black' )
ax2.scatter(7, 0.58, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([7.8,8.2], [mean_beta_mpool, mean_beta_mpool],  linewidth = 3, color = 'black' )
ax2.scatter(8, 0.58, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_trial_sig, betas_sig_trial, marker='o', color='black')
ax2.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
ax2.plot([8.8,9.2], [mean_beta_trial_no, mean_beta_trial_no],  linewidth = 3, color = 'black' )
ax2.scatter(9, 0.58, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_session_sig, betas_sig_session, marker='o', color='black')
ax2.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
ax2.plot([6.8,7.2], [mean_beta_session_no, mean_beta_session_no],  linewidth = 3, color = 'black' )



labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Valence'
labels[5] = 'Arousal'
labels[6] = 'M List'
labels[7] = 'M Pool'
labels[8] = 'Trial Number'
labels[9] = 'Session Number'

ax2.set_xticklabels(labels, rotation=20, size=9)
# ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.6, .6)


ax3.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
ax3.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("R-Squared Value")
ax3.xaxis.labelpad = 14
ax3.set_xticks([])
ax3.axes.get_xaxis().set_ticks([])
ax3.yaxis.tick_right()
combined = rsquareds_sig + rsquareds_notsig
combined = np.array(combined)
mean_combined = np.mean(combined)
ax3.plot([0.995,1.005], [mean_combined, mean_combined], linewidth = 3, color = 'black')

plt.savefig("Final_List_Recall_Jan8.pdf")
plt.show()
"""
(87,)
Significant R-Squareds [0.13809856230842732, 0.15639261831116957, 0.10489590995728071, 0.083245983437897131, 0.052809794636629381, 0.091062725668246691, 0.23863780271566903, 0.063796652671064935, 0.17420237103977099, 0.086861167553247243, 0.19051713104772794, 0.071475940407752181, 0.091942518268360751, 0.20506076176952981, 0.084585721198390029, 0.13710303111910249, 0.054293510288822899, 0.063775354671852469, 0.20823767955409334, 0.078478985333607532, 0.14530852391169857, 0.13497679680260033, 0.14671724876250758, 0.15894677030874982, 0.078023329214117809, 0.24399522468626356, 0.18305194668868507, 0.10646015295744682, 0.19347205965681025, 0.28857130033012668, 0.13234926661159241, 0.05438321172197591, 0.036533216495587939, 0.038003757199745691, 0.09208499669705239, 0.12200888378386132, 0.12916535087857584, 0.24182993815449949, 0.35256757514519499, 0.16492450079737353, 0.14350931075566553, 0.061920941215635517, 0.067678274157468854, 0.14656521804161415, 0.29103815481669115, 0.35105348873774656, 0.18475308028206316, 0.14987185901353894, 0.043125676862064366, 0.13980052692210299, 0.05901747876359198, 0.12738198756582153, 0.21405900765160424, 0.11116391703616668, 0.13370336821787576, 0.063930153952909574, 0.38027981251230403, 0.08768362364012916, 0.080410003582168987, 0.15589157736737602, 0.056473769583982869, 0.050232828343915958, 0.12617925112460093, 0.2345865139002673, 0.14577811051009881, 0.043619314628929207, 0.11308397104038126, 0.13557679897105168, 0.24671774980406835, 0.22721955267956007, 0.18985186560449896, 0.080688572775958978, 0.14560521733958043, 0.038801467155585523]
Not Significant R-Squareds [0.020688059236015133, 0.014020462509172105, 0.03244421344389703, 0.028618068877879987, 0.0096129154190059962, 0.019764124377397829, 0.018585920187637273, 0.025920424463014902, 0.018142630725710651, 0.028533529690764725, 0.015577196103033542, 0.029021417852162057, 0.022562854204169147]
(array([ True,  True, False,  True, False,  True,  True,  True,  True, False], dtype=bool), array([  4.56709215e-02,   1.13774752e-03,   2.59411002e-01,
         1.23004690e-03,   3.89883982e-01,   9.63316273e-03,
         1.45387240e-14,   1.13774752e-03,   2.31425193e-25,
         3.61603923e-01]), 0.0051161968918237433, 0.005)
T-test Intercepts Ttest_1sampResult(statistic=-0.39891393942199643, pvalue=0.69094542369770418)
T-test Concreteness Ttest_1sampResult(statistic=-2.1802636255409822, pvalue=0.0319696450697977)
T-test WordFreq Ttest_1sampResult(statistic=3.6517421778940062, pvalue=0.00044665677843764223)
T-test WordLen Ttest_1sampResult(statistic=-1.2699473232472005, pvalue=0.20752880141939156)
T-test Valence Ttest_1sampResult(statistic=3.5559891709723801, pvalue=0.0006150234520971964)
T-test Arousal Ttest_1sampResult(statistic=0.86419647092652729, pvalue=0.38988398189691975)
T-test Imag Ttest_1sampResult(statistic=2.8307980571680784, pvalue=0.005779897640192498)
T-test MList Ttest_1sampResult(statistic=9.6034958447068277, pvalue=2.9077448095223381e-15)
T-test MPool Ttest_1sampResult(statistic=-3.6461802939851453, pvalue=0.00045509900985103469)
T-test Trial No Ttest_1sampResult(statistic=-15.351305307455688, pvalue=2.3142519330146727e-26)
T-test Session No Ttest_1sampResult(statistic=-0.9889877931812453, pvalue=0.3254435304299812)
Beta Conc Ave, SE -0.0105613701009 0.00484407939349
Beta WordFreq Ave, SE 0.0172361047438 0.00471996759468
Beta Wordlen Ave, SE -0.00471519521728 0.00371290614261
Beta Valence Ave, SE 0.0192458137697 0.0054122250784
Beta Arousal Ave, SE 0.00510177370442 0.00590348824146
Beta Imag Ave, SE 0.0112176520242 0.00396271715526
Beta MList Ave, SE 0.0515389634604 0.00536668774516
Beta MPool Ave, SE -0.0206264760733 0.00565700936602
Beta Trial No Ave, SE -0.188595074813 0.0122852793972
Beta Session No Ave, SE -0.0255454201733 0.0258298639775
[-0.010999999999999999, 0.017000000000000001, -0.0050000000000000001, 0.019, 0.0050000000000000001, 0.010999999999999999, 0.051999999999999998, -0.021000000000000001, -0.189, -0.025999999999999999] [-0.010999999999999999, 0.017000000000000001, -0.0050000000000000001, 0.019, 0.0050000000000000001, 0.010999999999999999, 0.051999999999999998, -0.021000000000000001, -0.189, -0.025999999999999999]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.011*     0.005
     Word Frequency      0.017**     0.005
        Word Length     -0.005     0.004
            Valence      0.019**     0.005
            Arousal      0.005     0.006
       Imageability      0.011**     0.004
List Meaningfulness      0.052***     0.005
Pool Meaningfulness     -0.021**     0.006
           Trial No     -0.189***     0.012
         Session No     -0.026     0.026
 
 """