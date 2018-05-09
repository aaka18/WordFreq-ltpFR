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

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, MPool_byPoolSim as m_pool, \
    P_rec_ltpFR2_by_list as p_rec, Session_and_list_no_list as session_nos, semantic_similarity_bylist as m_list

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(
    files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness = np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_means_mpool = np.array(all_means_mpool)
all_trial_nos = np.array(trial_nos)
all_session_nos = np.array(session_nos)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
all_means_mpool_part = []
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
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part, ddof=1)

        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm,
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
             results.pvalues[5], results.pvalues[6], results.pvalues[7]], alpha=0.05, method='fdr_bh', is_sorted=False,
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
t_test_mpool = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:, 6], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:, 7], 0)
# print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])
print(multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1], t_test_mlist[1], t_test_mpool[1],
                     t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False,
                    returnsorted=False))

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
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
y_mlist = params[:, 4]
y_mpool = params[:, 5]
y_trial_no = params[:, 6]
y_session_no = params[:, 7]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_mlist = (np.array(params[:, 4]))
beta_mpool = (np.array(params[:, 5]))
beta_trial_no = (np.array(params[:, 6]))
beta_session_no = (np.array(params[:, 7]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)
mean_beta_trial_no = np.mean(beta_trial_no)
mean_beta_session_no = np.mean(beta_session_no)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))
print("Beta Trial No Ave, SE", np.mean(beta_trial_no), stats.sem(beta_trial_no))
print("Beta Session No Ave, SE", np.mean(beta_session_no), stats.sem(beta_session_no))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3), round(np.mean(beta_trial_no), 3),
             round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3),
             round(stats.sem(beta_trial_no), 3), round(stats.sem(beta_session_no), 3)]
predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'List Meaningfulness', 'Pool Meaningfulness', "Trial No",
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

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:, 5])):
    if fdr_pvalues[:, 5][i]:
        betas_sig_mpool.append(params[:, 5][i])
    else:
        betas_nonsig_mpool.append(params[:, 5][i])

betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:, 6])):
    if fdr_pvalues[:, 6][i]:
        betas_sig_trial.append(params[:, 6][i])
    else:
        betas_nonsig_trial.append(params[:, 6][i])

betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:, 7])):
    if fdr_pvalues[:, 7][i]:
        betas_sig_session.append(params[:, 7][i])
    else:
        betas_nonsig_session.append(params[:, 7][i])

x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen), 3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen), 3)

x_mlist_sig = np.full(len(betas_sig_mlist), 4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist), 4)

x_mpool_sig = np.full(len(betas_sig_mpool), 5)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool), 5)

x_trial_sig = np.full(len(betas_sig_trial), 6)
x_trial_nonsig = np.full(len(betas_nonsig_trial), 6)

x_session_sig = np.full(len(betas_sig_session), 7)
x_session_nonsig = np.full(len(betas_nonsig_session), 7)

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

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([4.8,5.2], [mean_beta_mpool, mean_beta_mpool],  linewidth = 3, color = 'black' )
ax2.scatter(5, 0.61, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_trial_sig, betas_sig_trial, marker='o', color='black')
ax2.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
ax2.plot([5.8,6.2], [mean_beta_trial_no, mean_beta_trial_no],  linewidth = 3, color = 'black' )
ax2.scatter(6, 0.61, s =65, marker = (5,2), color = 'black' )

ax2.scatter(x_session_sig, betas_sig_session, marker='o', color='black')
ax2.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
ax2.plot([6.8,7.2], [mean_beta_session_no, mean_beta_session_no],  linewidth = 3, color = 'black' )


labels = ['','Concreteness', 'Frequency', 'Length', 'M List', 'M Pool', 'Trial No', 'Session No']
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

plt.savefig("List_Recall_Dec31.pdf")
plt.show()

"""
(87,)
Significant R-Squareds [0.13217443541076568, 0.15572912238757242, 0.09966386809254224, 0.074050318260113435, 0.0488557547436137, 0.086528950086470435, 0.2165546022013084, 0.060761155364449038, 0.15724606909028371, 0.075232298347218451, 0.18896186923659908, 0.069807277429495773, 0.087214825348524427, 0.20418723891729518, 0.070483588102902117, 0.13686358802403042, 0.046120701285562449, 0.061987780471021381, 0.20196813054695595, 0.068621638141008745, 0.13266235784791525, 0.1313323394080752, 0.14088274318554894, 0.032403857610454834, 0.15705022803818214, 0.076318810981591101, 0.23835936693733539, 0.16963808420132109, 0.10220157692769194, 0.18599576798008655, 0.27882082993480117, 0.12706434835674985, 0.04621330968284787, 0.031864154938264111, 0.035524126301453629, 0.081838421988659982, 0.12118721502658736, 0.12655515524733485, 0.23850255746740412, 0.34583502524608689, 0.15960557209645454, 0.1403118021906804, 0.060271424633692172, 0.0589062510460725, 0.12126898779847239, 0.28826706306676475, 0.350404646048325, 0.18408490740186478, 0.14766547519585649, 0.042751092845087157, 0.13864489042263306, 0.051074588008484967, 0.11794004581701911, 0.20925869500695304, 0.11011956854716087, 0.12635940018752512, 0.057310522593221225, 0.37875291270186118, 0.083068968606767379, 0.071983791449067858, 0.14983441286259092, 0.050988518495190038, 0.045962042422512339, 0.1225470546563292, 0.23273548186276904, 0.14090740280676917, 0.041620497014491331, 0.10829331794411456, 0.13486427399003198, 0.24655097855105645, 0.22432054255944234, 0.18542517919003054, 0.078692105520262512, 0.14084184052233806, 0.038078942597911447]
Not Significant R-Squareds [0.017594337604473931, 0.012555408047215111, 0.024872850756441744, 0.008179430435938273, 0.019391821145256838, 0.0087681856651082857, 0.02380660308058502, 0.013691351357341452, 0.022611717415099153, 0.014982539978076415, 0.020106610691056126, 0.014546002690792537]
(array([False,  True, False,  True,  True,  True, False], dtype=bool), array([  2.22344910e-01,   7.44853628e-05,   2.22344910e-01,
         8.57397028e-15,   1.76948188e-03,   1.65100554e-25,
         3.25258226e-01]), 0.0073008319790146547, 0.0071428571428571435)
T-test Intercepts Ttest_1sampResult(statistic=-0.17368940300577015, pvalue=0.86251775185602242)
T-test Concreteness Ttest_1sampResult(statistic=-1.3192677737632275, pvalue=0.19058135175924767)
T-test WordFreq Ttest_1sampResult(statistic=4.3913444267228385, pvalue=3.1922298336590299e-05)
T-test WordLen Ttest_1sampResult(statistic=-1.3390942993961603, pvalue=0.1840686276424387)
T-test MList Ttest_1sampResult(statistic=9.6401092061917719, pvalue=2.4497057949634265e-15)
T-test MPool Ttest_1sampResult(statistic=-3.4038469229760424, pvalue=0.0010111325042874741)
T-test Trial No Ttest_1sampResult(statistic=-15.346701913527841, pvalue=2.3585793497671717e-26)
T-test Session No Ttest_1sampResult(statistic=-0.98936880419163575, pvalue=0.32525822624581757)
Beta Conc Ave, SE -0.00604403667582 0.00458135701942
Beta WordFreq Ave, SE 0.02188707564 0.00498414005215
Beta Wordlen Ave, SE -0.00502107103196 0.00374960227538
Beta MList Ave, SE 0.0515414932952 0.00534656736691
Beta MPool Ave, SE -0.0191908946059 0.00563800166109
Beta Trial No Ave, SE -0.187925316691 0.0122453226596
Beta Session No Ave, SE -0.0255546675859 0.0258292635442
[-0.0060000000000000001, 0.021999999999999999, -0.0050000000000000001, 0.051999999999999998, -0.019, -0.188, -0.025999999999999999] [-0.0060000000000000001, 0.021999999999999999, -0.0050000000000000001, 0.051999999999999998, -0.019, -0.188, -0.025999999999999999]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.006     0.005
     Word Frequency      0.022     0.005
        Word Length     -0.005     0.004
List Meaningfulness      0.052     0.005
Pool Meaningfulness     -0.019     0.006
           Trial No     -0.188     0.012
         Session No     -0.026     0.026
 
 
"""