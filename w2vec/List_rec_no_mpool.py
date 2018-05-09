# 2.	LIST RECALL MODEL I
# {Each of these vectors will have a dimension of N participants * 552 lists}
# 1)	Average Word Frequency Of The List - all_mean_word_freq_participants
# 2)	Average Word Concreteness Of The List - all_mean_conc_participants
# 3)	Average Word Length Of The List - all_mean_wordlen_participants
# 4)	Mlist (Each word’s aver. similarity to all other words in the list) - m_list
# 5)	Mpool (Each word’s aver. similarity to all other words in the pool) - m_pool
# 6) P_rec per list - p_rec_list

import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, P_rec_ltpFR2_by_list as p_rec, \
    Session_and_list_no_list as session_nos, semantic_similarity_bylist as m_list

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(
    files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness = np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_trial_nos = np.array(trial_nos)
all_session_nos = np.array(session_nos)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
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
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
                                    all_m_list_part_norm,  all_trial_nos_part_norm,
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
             results.pvalues[5], results.pvalues[6]], alpha=0.05, method='fdr_bh', is_sorted=False,
            returnsorted=False))
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
t_test_mlist = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 6], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])
print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_mlist[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist)
print("T-test Trial No", t_test_trial_no)
print("T-test Session No", t_test_session_no)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_mlist = params[:, 4]
y_trial_no = params[:, 5]
y_session_no = params[:, 6]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_mlist = (np.array(params[:, 4]))
beta_trial_no = (np.array(params[:, 5]))
beta_session_no = (np.array(params[:, 6]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_mlist), 3),  round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_mlist), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'List Meaningfulness', "Trial No",
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

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:, 4])):
    if fdr_pvalues[:, 4][i]:
        betas_sig_mlist.append(params[:, 4][i])
    else:
        betas_nonsig_mlist.append(params[:, 4][i])

betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:, 5])):
    if fdr_pvalues[:, 5][i]:
        betas_sig_trial.append(params[:, 5][i])
    else:
        betas_nonsig_trial.append(params[:, 5][i])

betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_session.append(params[:, 6][i])
    else:
        betas_nonsig_session.append(params[:, 6][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_mlist_sig = np.full(len(betas_sig_mlist), 4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 4)

x_trial_sig = np.full(len(betas_sig_trial), 5)
x_trial_nonsig = np.full(len(betas_nonsig_trial), 5)

x_session_sig = np.full(len(betas_sig_session), 6)
x_session_nonsig = np.full(len(betas_nonsig_session), 6)

from matplotlib import gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax2 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

ax2.axhline(y=0, color='gray', linestyle='--')

ax2.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
ax2.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
ax2.plot([.8, 1.2], [mean_beta_conc, mean_beta_conc], linewidth = 3, color = 'black' )

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq],  linewidth = 3, color = 'black' )
ax2.scatter(2, 0.61, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen],  linewidth = 3, color = 'black' )

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_mlist, mean_beta_mlist],  linewidth = 3, color = 'black' )
ax2.scatter(4, 0.61, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_trial_sig, betas_sig_trial, marker='o', color='black')
ax2.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
ax2.plot([4.8,5.2], [mean_beta_trial_no, mean_beta_trial_no],  linewidth = 3, color = 'black' )
ax2.scatter(5, 0.61, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_session_sig, betas_sig_session, marker='o', color='black')
ax2.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
ax2.plot([5.8,6.2], [mean_beta_session_no, mean_beta_session_no],  linewidth = 3, color = 'black' )


labels = ['','Concreteness', 'Frequency', 'Length', 'M List', 'Trial No', 'Session No']
ax2.set_xticklabels(labels, rotation = 15, size = 9.5)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.6, .67)


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

plt.savefig("List_No_MPool_Dec31.pdf")
plt.show()

"""
(87,)
Significant R-Squareds [0.13184697175142235, 0.15572695799302783, 0.098643632382716895, 0.071188167615561349, 0.047439218385605275, 0.085765136879720205, 0.21512439538906258, 0.058034088690147256, 0.15239973886699731, 0.074200467234005751, 0.18454527139481791, 0.069804835598386794, 0.086128173249495621, 0.20418302383764786, 0.056636622141598814, 0.13519871822726659, 0.046001627992411365, 0.061780027200490584, 0.20196813051863305, 0.06858710369797183, 0.13236855107186518, 0.13106236073982336, 0.14078931611171164, 0.031576441143131184, 0.15179436829615811, 0.076084987612959454, 0.23821139041841932, 0.16924125524266054, 0.099803108045248168, 0.18395052191699224, 0.27596766572260789, 0.12692759933288378, 0.044645661270538928, 0.031256726775361621, 0.02487135464037693, 0.034376788340043207, 0.08059944443517264, 0.11772361323668223, 0.12628205347841392, 0.23488138737744568, 0.34581738026533348, 0.15958869786730023, 0.13963727578395846, 0.060034557269564415, 0.05815586839623299, 0.11470929451396905, 0.28145332005540347, 0.34781692260087438, 0.18395300689300476, 0.14047762958005905, 0.04233421750142663, 0.1343433832592339, 0.050507439288533962, 0.11747721761622398, 0.20401434579732924, 0.023537877523689854, 0.10116703379651437, 0.12111433214339673, 0.056607301901194895, 0.37797934312848402, 0.082991800227594337, 0.07026528436346724, 0.14638254420608288, 0.05080633594588535, 0.0411002404262516, 0.1225072000319869, 0.23046597854741913, 0.13599924561812604, 0.038517610383532075, 0.10493112085265288, 0.13461673700371624, 0.24525943089869784, 0.21950963545526669, 0.18215916112683916, 0.078657703752720742, 0.13948127746900096, 0.037670864447851171]
Not Significant R-Squareds [0.017594114601162558, 0.010736779366095517, 0.0081467477788518261, 0.018358917141853337, 0.0047324482682996605, 0.013114400910036683, 0.021954595549044664, 0.013343756681285623, 0.020027068840647755, 0.013860202565760815]
(array([False,  True, False,  True,  True, False], dtype=bool), array([  1.70300087e-01,   3.66320118e-05,   2.02178037e-01,
         6.97066453e-14,   1.35373505e-25,   3.24906747e-01]), 0.008512444610847103, 0.008333333333333333)
T-test Intercepts Ttest_1sampResult(statistic=-0.16880477832244822, pvalue=0.86634659406163239)
T-test Concreteness Ttest_1sampResult(statistic=-1.5987973249045357, pvalue=0.1135333914751301)
T-test WordFreq Ttest_1sampResult(statistic=4.5379891644624548, pvalue=1.8316005894015967e-05)
T-test WordLen Ttest_1sampResult(statistic=-1.3887938206627368, pvalue=0.16848169758075765)
T-test MList Ttest_1sampResult(statistic=9.1600314961962486, pvalue=2.3235548420327536e-14)
T-test Trial No Ttest_1sampResult(statistic=-15.357467506205683, pvalue=2.2562250809137958e-26)
T-test Session No Ttest_1sampResult(statistic=-0.99009188939593329, pvalue=0.32490674683106435)
Beta Conc Ave, SE -0.00724154569332 0.00452937065913
Beta WordFreq Ave, SE 0.0227753117996 0.00501881141056
Beta Wordlen Ave, SE -0.00522691640196 0.00376363742709
Beta MList Ave, SE 0.0404386420425 0.00441468373327
Beta Trial No Ave, SE -0.18792466586 0.0122366963033
Beta Session No Ave, SE -0.0255768879408 0.0258328426025
[-0.0070000000000000001, 0.023, -0.0050000000000000001, 0.040000000000000001, -0.188, -0.025999999999999999] [-0.0070000000000000001, 0.023, -0.0050000000000000001, 0.040000000000000001, -0.188, -0.025999999999999999]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.007     0.005
     Word Frequency      0.023     0.005
        Word Length     -0.005     0.004
List Meaningfulness       0.04     0.004
           Trial No     -0.188     0.012
         Session No     -0.026     0.026
 """