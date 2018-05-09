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
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness = np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_valence = np.array(all_mean_valence_participants)
all_arousal = np.array(all_mean_arousal_participants)
all_trial_nos = np.array(trial_nos)
all_session_nos = np.array(session_nos)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
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
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_valence_part_norm, all_arousal_part_norm,
                                    all_m_list_part_norm, all_trial_nos_part_norm,
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
t_test_mlist = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 7], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 8], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_valence[1],
                     t_test_arousal[1], t_test_mlist[1],
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
y_trial_no = params[:, 7]
y_session_no = params[:, 8]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))
beta_trial_no = (np.array(params[:, 7]))
beta_session_no = (np.array(params[:, 8]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta Arousal Ave, SE", np.mean(beta_arousal), stats.sem(beta_arousal))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3), round(np.mean(beta_arousal), 3),
             round(np.mean(beta_mlist), 3),  round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3), round(stats.sem(beta_arousal), 3),
             round(stats.sem(beta_mlist), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'Arousal', 'List Meaningfulness',  "Trial No", "Session No", ]
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

x_mlist_sig = np.full(len(betas_sig_mlist), 6)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 6)

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

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
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

my_xticks = ['Concreteness','Frequency','Length','Valence', 'Arousal', 'MList', 'MPool', 'Trial No', 'Session No']
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

plt.savefig("Final_List_Recall_Final_NoMPoolJan9.pdf")
plt.show()
"""

(88,)
Significant R-Squareds [0.13257044957076547, 0.15626946207426717, 0.10301790735108718, 0.076804260948138747, 0.051266621657931832, 0.060991125736433061, 0.22942599715797807, 0.063275337245035201, 0.16275327322039401, 0.084227339084492203, 0.18611825730872811, 0.072579534449952354, 0.090696624559040151, 0.2087407277977843, 0.068356127838372749, 0.13544139140109268, 0.044635525657876429, 0.066375493846575484, 0.20791875876958377, 0.078036929233011088, 0.15196647620815729, 0.13181261641874442, 0.14351411376367917, 0.031586907906109074, 0.15395404623961517, 0.079247474707809573, 0.24384847677574339, 0.18189505470486533, 0.10302669125621589, 0.19002850554704864, 0.28239559091362842, 0.12704911936273566, 0.047845605214407838, 0.034352457566411809, 0.028769919419573853, 0.036370961872913443, 0.091262506976793389, 0.11690195369511258, 0.1251264361554737, 0.23841664328990675, 0.35147907941207923, 0.15495985191137107, 0.13878021959769127, 0.061358928838032623, 0.063619177159157214, 0.14670415746276744, 0.29320865450882649, 0.34791181386626469, 0.18489436732428288, 0.14565978684646319, 0.042095919254924508, 0.13791507046099383, 0.052011773425290952, 0.12573368192202106, 0.21378187979257834, 0.10145547956919387, 0.12477748708045577, 0.062488872411932461, 0.37880523431485336, 0.078712150635862788, 0.078509783412925138, 0.1519290368373204, 0.055543302676468898, 0.042695928491555679, 0.12459628877344753, 0.23209182089066327, 0.14620312529348733, 0.08826562020514972, 0.11004593724627276, 0.13481039453051691, 0.24604615687918574, 0.22032730922355592, 0.18612005476066085, 0.080814293216816768, 0.14529339984704071, 0.038234939533979051]
Not Significant R-Squareds [0.020443873081620212, 0.012234596389458674, 0.0087819875540686709, 0.018455272609799067, 0.014128505560103366, 0.026957288890961717, 0.017180517177807442, 0.027860017625525146, 0.013745688382341692, 0.021287542727292941, 0.015781966908969491, 0.024550886291460072]
(array([False,  True, False,  True, False,  True,  True, False], dtype=bool), array([  9.95534245e-02,   4.99270068e-04,   2.09602691e-01,
         1.12372366e-03,   4.19550094e-01,   2.33215079e-13,
         1.71338853e-25,   3.29379954e-01]), 0.0063911509545450107, 0.00625)
T-test Intercepts Ttest_1sampResult(statistic=-0.22799183138431819, pvalue=0.82018757771057627)
T-test Concreteness Ttest_1sampResult(statistic=-1.8890173843910794, pvalue=0.062220890301508916)
T-test WordFreq Ttest_1sampResult(statistic=3.9025458768706334, pvalue=0.00018722627541474128)
T-test WordLen Ttest_1sampResult(statistic=-1.4268507121609535, pvalue=0.15720201844525741)
T-test Valence Ttest_1sampResult(statistic=3.5816229531559309, pvalue=0.00056186183186143204)
T-test Arousal Ttest_1sampResult(statistic=0.81105022644833868, pvalue=0.41955009405277088)
T-test MList Ttest_1sampResult(statistic=8.9446688429686603, pvalue=5.8303769677313296e-14)
T-test Trial No Ttest_1sampResult(statistic=-15.29879177323304, pvalue=2.1417356612490756e-26)
T-test Session No Ttest_1sampResult(statistic=-1.0685967523624351, pvalue=0.28820745950875082)
Beta Conc Ave, SE -0.00865093357945 0.00457959447644
Beta WordFreq Ave, SE 0.018726061001 0.00479842174616
Beta Wordlen Ave, SE -0.00524694970616 0.00367729410053
Beta Valence Ave, SE 0.0193931091097 0.00541461492831
Beta Arousal Ave, SE 0.00471376772614 0.00581193071948
Beta MList Ave, SE 0.0398011067055 0.00444970153778
Beta Trial No Ave, SE -0.187402420704 0.01224949156
Beta Session No Ave, SE -0.0273595433177 0.0256032439339
[-0.0089999999999999993, 0.019, -0.0050000000000000001, 0.019, 0.0050000000000000001, 0.040000000000000001, -0.187, -0.027] [-0.0089999999999999993, 0.019, -0.0050000000000000001, 0.019, 0.0050000000000000001, 0.040000000000000001, -0.187, -0.027]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.009     0.005
     Word Frequency      0.019     0.005
        Word Length     -0.005     0.004
            Valence      0.019     0.005
            Arousal      0.005     0.006
List Meaningfulness       0.04     0.004
           Trial No     -0.187     0.012
         Session No     -0.027     0.026
 
 

 """