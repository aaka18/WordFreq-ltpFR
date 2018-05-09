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
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_valence_part_norm, all_m_list_part_norm, all_means_mpool_part_norm,
                                    all_trial_nos_part_norm, all_session_nos_part_norm))

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
t_test_mlist = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 8], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_valence[1],
                      t_test_mlist[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
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
y_mlist = params[:, 5]
y_mpool = params[:, 6]
y_trial_no = params[:, 7]
y_session_no = params[:, 8]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_mlist = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))
beta_trial_no = (np.array(params[:, 7]))
beta_session_no = (np.array(params[:, 8]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
               'List Meaningfulness', 'Pool Meaningfulness', "Trial No",
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

x_mlist_sig = np.full(len(betas_sig_mlist), 5)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 5)

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

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
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


my_xticks = ['Concreteness','Frequency','Length','Valence', 'MList', 'MPool', 'Trial No', 'Session No']
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

plt.savefig("Final_List_Recall_Final_NoArousalJan9.pdf")
plt.show()
"""
(88,)
Significant R-Squareds [0.13295631704155286, 0.15595335088622475, 0.10247802062187172, 0.079894409609251316, 0.050842829205920648, 0.06101739751099422, 0.21617219689313649, 0.065600399094738138, 0.16468833758538948, 0.081483133098004035, 0.18938220516099047, 0.072286227031138184, 0.090697406962892191, 0.20870390031488673, 0.084011143305572022, 0.13698766812819585, 0.044612932526284732, 0.066038831214497273, 0.20568435066598179, 0.070850738179326522, 0.13891620333646193, 0.13111535490471415, 0.14142854145904438, 0.032417125820477732, 0.1579081192953069, 0.07904592363856211, 0.23886774469310312, 0.17637095552770199, 0.10629008331874756, 0.18762560744103818, 0.2823289671915401, 0.12707807699421736, 0.047233718244156453, 0.032543846442093294, 0.028672180671362679, 0.035602641419428815, 0.088311565462729558, 0.11940738989396693, 0.12392855120481938, 0.23948882599724386, 0.34603127321125948, 0.15377493022152555, 0.13966779713117727, 0.060104214349084528, 0.061139782834142453, 0.13515599692798186, 0.29899868145784725, 0.3504051237472352, 0.18490149051893334, 0.15269059531485429, 0.042466898079930093, 0.1437953362811597, 0.052601119937600638, 0.11859361677299218, 0.2195913993418428, 0.11039485664435389, 0.12846355718987701, 0.061271688684899517, 0.37875291810691036, 0.077353593677388033, 0.078930030601842338, 0.1498753084504747, 0.055900446036963203, 0.046266645665266837, 0.12381379393124015, 0.2344594709406278, 0.1507181529573739, 0.092933140194065178, 0.10959831488361693, 0.13499264703784608, 0.24720714353895779, 0.22435729728177367, 0.1849008660301722, 0.080361960336241989, 0.14639958285713484, 0.038492082306346154]
Not Significant R-Squareds [0.019767891488328804, 0.01273858366777636, 0.0086303519682706664, 0.019507305606112912, 0.017012244258880926, 0.026790867357563752, 0.01726039747366015, 0.022620494343974396, 0.015177141307074371, 0.019623784155549551, 0.016381139084590934, 0.022694475427319394]
(array([False,  True, False,  True,  True,  True,  True, False], dtype=bool), array([  9.98955229e-02,   7.64353248e-04,   1.97552558e-01,
         5.09744767e-04,   1.03259687e-14,   7.64353248e-04,
         1.72576216e-25,   2.88588188e-01]), 0.0063911509545450107, 0.00625)
T-test Intercepts Ttest_1sampResult(statistic=-0.094835464845096293, pvalue=0.92466360402474035)
T-test Concreteness Ttest_1sampResult(statistic=-1.8025506981827115, pvalue=0.074921642150460974)
T-test WordFreq Ttest_1sampResult(statistic=3.6300951301412727, pvalue=0.00047772078018078673)
T-test WordLen Ttest_1sampResult(statistic=-1.3743649169122616, pvalue=0.17285848807310217)
T-test Valence Ttest_1sampResult(statistic=3.8966338300414955, pvalue=0.00019115428774578046)
T-test MList Ttest_1sampResult(statistic=9.605875790839459, pvalue=2.5814921820096142e-15)
T-test MPool Ttest_1sampResult(statistic=-3.6656517958327206, pvalue=0.00042375619776710815)
T-test Trial No Ttest_1sampResult(statistic=-15.297063266206093, pvalue=2.1572027046516711e-26)
T-test Session No Ttest_1sampResult(statistic=-1.0677477569169895, pvalue=0.28858818810993059)
Beta Conc Ave, SE -0.00825291559507 0.00457846517349
Beta WordFreq Ave, SE 0.0178736517199 0.00492374196244
Beta Wordlen Ave, SE -0.00506205100811 0.00368319283024
Beta Valence Ave, SE 0.0175203155334 0.00449626942062
Beta MList Ave, SE 0.0510489217296 0.00531434330832
Beta MPool Ave, SE -0.0203793126749 0.0055595331499
Beta Trial No Ave, SE -0.187186061355 0.0122367318548
Beta Session No Ave, SE -0.0273352369304 0.0256008376073
[-0.0080000000000000002, 0.017999999999999999, -0.0050000000000000001, 0.017999999999999999, 0.050999999999999997, -0.02, -0.187, -0.027] [-0.0080000000000000002, 0.017999999999999999, -0.0050000000000000001, 0.017999999999999999, 0.050999999999999997, -0.02, -0.187, -0.027]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.008     0.005
     Word Frequency      0.018     0.005
        Word Length     -0.005     0.004
            Valence      0.018     0.004
List Meaningfulness      0.051     0.005
Pool Meaningfulness      -0.02     0.006
           Trial No     -0.187     0.012
         Session No     -0.027     0.026
 


 """