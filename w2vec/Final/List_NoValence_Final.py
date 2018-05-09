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
        all_arousal_part_norm = stats.mstats.zscore(all_arousal_part, ddof=1)
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_arousal_part_norm, all_m_list_part_norm, all_means_mpool_part_norm,
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
t_test_arousal = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 8], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                     t_test_arousal[1], t_test_mlist[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
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
y_arousal = params[:, 4]
y_mlist = params[:, 5]
y_mpool = params[:, 6]
y_trial_no = params[:, 7]
y_session_no = params[:, 8]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_arousal = (np.array(params[:, 4]))
beta_mlist = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))
beta_trial_no = (np.array(params[:, 7]))
beta_session_no = (np.array(params[:, 8]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
              round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3),  round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',
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

x_arousal_sig = np.full(len(betas_sig_arousal), 4)
x_arousal_nonsig = np.full(len(betas_nonsig_arousal), 4)

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

ax2.scatter(x_arousal_sig, betas_sig_arousal, marker='o', color='black')
ax2.scatter(x_arousal_nonsig, betas_nonsig_arousal, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_arousal, mean_beta_arousal], linewidth=3, color='black')
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

my_xticks = ['Concreteness','Frequency','Length', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No']
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

plt.savefig("Final_List_Recall_Final_NoValenceJan9.pdf")
plt.show()
"""
(88,)
Significant R-Squareds [0.13062379286363845, 0.15576051937990065, 0.099671001620712407, 0.074764537444121815, 0.049009220445973223, 0.06121240044581977, 0.22262804794980373, 0.065321462014511722, 0.16878922630986282, 0.082760322719095614, 0.18914528759607563, 0.072367676265502001, 0.087281437588993205, 0.20822809507833417, 0.075444329263900878, 0.13699574656708358, 0.043171394626964177, 0.066124901910819212, 0.20795085206788277, 0.071623581346398235, 0.15025657455399077, 0.13195316617584918, 0.14299250184478307, 0.032405232084229296, 0.15858628324653667, 0.077786042286437818, 0.24066790747151146, 0.16981326086474635, 0.10354807638340013, 0.18690995677658295, 0.27849265112632104, 0.12718692232285755, 0.046676286802187716, 0.03250285266557229, 0.036254143745029332, 0.090759036121544456, 0.11914170873658758, 0.12345434316033044, 0.24153098494266112, 0.34791973549683175, 0.15495226040777987, 0.13953855710901764, 0.060412913218041098, 0.064179357816503302, 0.15045515957798272, 0.29688986163761599, 0.35054478619090568, 0.18500325938330997, 0.15170985272740312, 0.042477509338072816, 0.14379437773130155, 0.051499656294584972, 0.1254688745689051, 0.21871365515226959, 0.11016934085625174, 0.12771432195847299, 0.057310658339770004, 0.37940351834083774, 0.076589630895186223, 0.077946253963209977, 0.15372612186369383, 0.052760187357834942, 0.047655766856300352, 0.12181385836017256, 0.23341389005144497, 0.14923865143842374, 0.092779776453167484, 0.10940028854728678, 0.1348155318877261, 0.24708756289898637, 0.22468423712072005, 0.18890463026366922, 0.07896078891539815, 0.14288111849319163, 0.038081940387030389]
Not Significant R-Squareds [0.019855092057443735, 0.011496022412607498, 0.025452395376299286, 0.0086654996640173065, 0.019242625869613672, 0.0090998864310478655, 0.026714717088970774, 0.013973002731641704, 0.026401429898920581, 0.015002129576365153, 0.021385583080750448, 0.01485497123410795, 0.023749617450246618]
(array([False,  True, False, False,  True,  True,  True, False], dtype=bool), array([  1.59554886e-01,   1.26771075e-04,   2.45581016e-01,
         2.29193110e-01,   1.63553044e-14,   2.06046344e-03,
         1.49636146e-25,   2.88287633e-01]), 0.0063911509545450107, 0.00625)
T-test Intercepts Ttest_1sampResult(statistic=-0.099554746939532174, pvalue=0.92092686456108386)
T-test Concreteness Ttest_1sampResult(statistic=-1.6639435301563616, pvalue=0.099721803830289349)
T-test WordFreq Ttest_1sampResult(statistic=4.2819311519314107, pvalue=4.7539153226392342e-05)
T-test WordLen Ttest_1sampResult(statistic=-1.2493624724835062, pvalue=0.21488338924130551)
T-test Arousal Ttest_1sampResult(statistic=-1.3774880645398109, pvalue=0.17189483262147365)
T-test MList Ttest_1sampResult(statistic=9.5082595081304024, pvalue=4.0888261056029088e-15)
T-test MPool Ttest_1sampResult(statistic=-3.3966626693779576, pvalue=0.0010302317179661841)
T-test Trial No Ttest_1sampResult(statistic=-15.331342145542274, pvalue=1.8704518233877414e-26)
T-test Session No Ttest_1sampResult(statistic=-1.068417908295562, pvalue=0.28828763275351627)
Beta Conc Ave, SE -0.00779961950373 0.00468743040997
Beta WordFreq Ave, SE 0.0214587247193 0.00501145953961
Beta Wordlen Ave, SE -0.00460766026843 0.00368800918061
Beta Arousal Ave, SE -0.00664781475509 0.00482604163783
Beta MList Ave, SE 0.0510386006044 0.00536781737612
Beta MPool Ave, SE -0.0190034733128 0.00559474848182
Beta Trial No Ave, SE -0.187495154993 0.0122295330189
Beta Session No Ave, SE -0.0273528829887 0.0256012958752
[-0.0080000000000000002, 0.021000000000000001, -0.0050000000000000001, -0.0070000000000000001, 0.050999999999999997, -0.019, -0.187, -0.027] [-0.0080000000000000002, 0.021000000000000001, -0.0050000000000000001, -0.0070000000000000001, 0.050999999999999997, -0.019, -0.187, -0.027]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.008     0.005
     Word Frequency      0.021     0.005
        Word Length     -0.005     0.004
            Arousal     -0.007     0.005
List Meaningfulness      0.051     0.005
Pool Meaningfulness     -0.019     0.006
           Trial No     -0.187     0.012
         Session No     -0.027     0.026

 """