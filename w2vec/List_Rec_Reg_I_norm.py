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
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)
session_nos, trial_nos = session_nos.measures_by_list(files_ltpFR2)

all_probs = np.array(p_rec_list)
all_concreteness= np.array(all_mean_conc_participants)
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
        #print("Mask", mask)
        #print(np.shape(np.array(mask)))
        all_probs_part = np.array(all_probs_part)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_m_list_part = np.array(all_m_list_part)[mask]
        all_means_mpool_part = np.array(all_means_mpool_part)[mask]
        all_trial_nos_part = np.array(all_trial_nos_part)[mask]
        all_session_nos_part = np.array(all_session_nos_part)[mask]


        all_probs_part_norm = stats.mstats.zscore(all_probs_part,  ddof=1)
        #print(all_probs_part_norm)
        all_concreteness_part_norm = stats.mstats.zscore(all_concreteness_part, ddof=1)
        #print("Conc", all_concreteness_part_norm)
        all_wordfreq_part_norm = stats.mstats.zscore(all_wordfreq_part, ddof=1)
        all_wordlen_part_norm = stats.mstats.zscore(all_wordlen_part, ddof=1)
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part,  ddof=1)
        all_trial_nos_part_norm = stats.mstats.zscore(all_trial_nos_part, ddof=1)
        all_session_nos_part_norm = stats.mstats.zscore(all_session_nos_part,  ddof=1)



        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part_norm,  all_wordfreq_part_norm, all_wordlen_part_norm, all_m_list_part_norm, all_means_mpool_part_norm, all_trial_nos_part_norm, all_session_nos_part_norm))

        # x-values without the m_list
        #x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        #print(x_values)
        #print(np.shape(x_values))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        y_value = all_probs_part_norm

        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()


        #print(results)
        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        residuals.append(results.resid)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3], results.pvalues[4], results.pvalues[5], results.pvalues[6], results.pvalues[7]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
        fdr_pvalues.append(fdr_p[0])
        print(results.f_pvalue)
        model_pvalues.append(results.f_pvalue)

    return  params, rsquareds, predict, pvalues, residuals, model_pvalues, fdr_pvalues

params, rsquareds, predict, pvalues, f_pvalues, residuals, fdr_pvalues = calculate_params()
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordFreq coeff, WordLen coeff,  MList coeff, MPool coeff)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)

rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
model_pvalues = np.array(model_pvalues)
print(model_pvalues)
print(model_pvalues.shape)
#print(fdr_pvalues.shape)
#print("Pvalues", pvalues)
#print(pvalues.shape)
#print("Corrected p values", fdr_pvalues)

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

# x_rsq_sig = np.full(len(rsquareds_sig), 1)
# x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)
#
# print("R sig", len(rsquareds_sig))
# print("R not sig", len(rsquareds_notsig))

sig = np.array(rsquareds_sig)
no = np.array(rsquareds_notsig)

combined = np.array([sig, no]).tolist()
plt.hist(combined, 8, normed=1, histtype='bar', stacked=True, label = ['Significant Models', 'Not Significant Models'], color = ['black','darkgray'])
plt.title('R-Squared Values')
plt.xlabel("R-Squared Value", size = 14)
plt.ylabel("Frequency", size = 13)
plt.legend()
# plt.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
# plt.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )
#
# plt.xticks([])
# plt.legend()
plt.savefig("Rsq_hist_list_Nov14.pdf")
plt.show()

t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,5], 0)
t_test_trial_no = scipy.stats.ttest_1samp(params[:,6], 0)
t_test_session_no = scipy.stats.ttest_1samp(params[:,7], 0)
#print([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1]])
print(multipletests([t_test_concreteness[1],t_test_wordfreq[1], t_test_wordlen[1],t_test_mlist[1],t_test_mpool[1],t_test_trial_no[1], t_test_session_no[1]], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))

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
y_concreteness = params[:,1]
y_wordfreq = params[:,2]
y_wordlen = params[:,3]
y_mlist = params[:,4]
y_mpool = params[:,5]
y_trial_no = params[:, 6]
y_session_no = params[:, 7]

beta_concreteness = (np.array(params[:,1]))
beta_wordfreq = (np.array(params[:,2]))
beta_wordlen =(np.array(params[:,3]))
beta_mlist = (np.array(params[:,4]))
beta_mpool = (np.array(params[:,5]))
beta_trial_no = (np.array(params[:,6]))
beta_session_no = (np.array(params[:,7]))

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

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3),  round(np.mean(beta_mlist),3), round(np.mean(beta_mpool),3), round(np.mean(beta_trial_no), 3), round(np.mean(beta_session_no), 3)]
sem_betas = [round(stats.sem(beta_concreteness),3), round(stats.sem(beta_wordfreq),3), round(stats.sem(beta_wordlen),3),  round(stats.sem(beta_mlist),3),round(stats.sem(beta_mpool),3), round(stats.sem(beta_trial_no), 3),round(stats.sem(beta_session_no), 3) ]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',  'List Meaningfulness', 'Pool Meaningfulness',"Trial No", "Session No", ]
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names = ("List Recall Model", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))

print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")


betas_sig_conc = []
betas_nonsig_conc = []
for i in range(len(params[:,1])):
    if fdr_pvalues[:,1][i]:
        betas_sig_conc.append(params[:,1][i])
    else:
        betas_nonsig_conc.append(params[:,1][i])


betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:,2])):
    if fdr_pvalues[:,2][i]:
        betas_sig_wordfreq.append(params[:,2][i])
    else:
        betas_nonsig_wordfreq.append(params[:,2][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:,3])):
    if fdr_pvalues[:,3][i]:
        betas_sig_wordlen.append(params[:,3][i])
    else:
        betas_nonsig_wordlen.append(params[:,3][i])

betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:,4])):
    if fdr_pvalues[:,4][i]:
        betas_sig_mlist.append(params[:,4][i])
    else:
        betas_nonsig_mlist.append(params[:,4][i])

betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:,5])):
    if fdr_pvalues[:,5][i]:
        betas_sig_mpool.append(params[:,5][i])
    else:
        betas_nonsig_mpool.append(params[:,5][i])



betas_sig_trial = []
betas_nonsig_trial = []
for i in range(len(params[:,6])):
    if fdr_pvalues[:,6][i]:
        betas_sig_trial.append(params[:,6][i])
    else:
        betas_nonsig_trial.append(params[:,6][i])


betas_sig_session = []
betas_nonsig_session = []
for i in range(len(params[:,7])):
    if fdr_pvalues[:,7][i]:
        betas_sig_session.append(params[:,7][i])
    else:
        betas_nonsig_session.append(params[:,7][i])


x_conc_sig = np.full(len(betas_sig_conc), 1)
x_conc_nonsig = np.full(len(betas_nonsig_conc), 1)

x_wordfreq_sig = np.full(len(betas_sig_wordfreq), 2)
x_wordfreq_nonsig = np.full(len(betas_nonsig_wordfreq), 2)

x_wordlen_sig = np.full(len(betas_sig_wordlen),3)
x_wordlen_nonsig = np.full(len(betas_nonsig_wordlen),3)

x_mlist_sig = np.full(len(betas_sig_mlist),4)
x_mlist_nonsig = np.full(len(betas_nonsig_mlist),4)

x_mpool_sig = np.full(len(betas_sig_mpool),5)
x_mpool_nonsig = np.full(len(betas_nonsig_mpool),5)

x_trial_sig = np.full(len(betas_sig_trial),6)
x_trial_nonsig = np.full(len(betas_nonsig_trial),6)

x_session_sig = np.full(len(betas_sig_session),7)
x_session_nonsig = np.full(len(betas_nonsig_session),7)



fig, ax = plt.subplots()
fig.canvas.draw()
plt.axhline(y=0, color='gray', linestyle='--')

#plt.scatter(x_conc, y_concreteness, alpha=0.5)
plt.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
plt.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
plt.scatter(1, mean_beta_conc, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black', label = "Word Frequency"  )
plt.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
plt.scatter(2, mean_beta_wordfreq, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black')
plt.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
plt.scatter(3, mean_beta_wordlen, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
plt.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
plt.scatter(4, mean_beta_mlist, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
plt.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
plt.scatter(5, mean_beta_mpool, s =65, marker = (5,2), color = 'black' )


plt.scatter(x_trial_sig, betas_sig_trial, marker='o', color = 'black' )
plt.scatter(x_trial_nonsig, betas_nonsig_trial, facecolors='none', edgecolors='gray')
plt.scatter(6, mean_beta_trial_no, s =65, marker = (5,2), color = 'black' )

plt.scatter(x_session_sig, betas_sig_session, marker='o', color = 'black' )
plt.scatter(x_session_nonsig, betas_nonsig_session, facecolors='none', edgecolors='gray')
plt.scatter(7, mean_beta_session_no, s =65, marker = (5,2), color = 'black' )


# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels[1] = 'Concreteness'
# labels[3] = 'Word Freq'
# labels[4] = 'Word Length'
# labels[5] = 'M List'
# labels[6] = 'M Pool'
# labels[7] = 'Session No'
# labels[8] = 'Trial No'

labels = [" ", "Concreteness", "Word \n Frequency", "Word \n Length", "M List", "M Pool", "Trial \n Number", "Session \n Number"]
ax.set_xticklabels(labels)
# #ax.xaxis.set_tick_params(rotation=50)
plt.xticks(size = 6, rotation=35)



# print("Rsq", rsquareds.tolist())
#
# my_bins = np.arange(0, 1, .025).tolist()
#
# plt.hist(rsquareds , bins = my_bins)
# plt.title("Multiple Regressions")
# plt.xlabel("R-Squared Value")
# plt.ylabel("Frequency")
plt.savefig("Betas_LIST_Rec_Model_Nov14.pdf")

plt.show()


print("rsq", rsquareds.tolist())


"""
OUTPUT:
(78,)
Significant R-Squareds [0.13266235784791525, 0.04621330968284787, 0.069807277429495773, 0.074050318260113435, 0.061987780471021381, 0.13217443541076568, 0.14088274318554894, 0.12118721502658736, 0.12606868701847529, 0.060761155364449038, 0.031717466274735373, 0.27870161166887675, 0.12706434835674985, 0.1891638891452847, 0.086395279754086229, 0.062375368716092283, 0.2165546022013084, 0.18896186923659908, 0.050480653001431008, 0.032403857610454834, 0.0589062510460725, 0.09966386809254224, 0.24913922129456334, 0.15572912238757242, 0.13686358802403042, 0.046120701285562449, 0.1313323394080752, 0.15705022803818214, 0.20246430713528485, 0.070483588102902117, 0.23545580168020441, 0.050734119313448178, 0.075232298347218451, 0.0488557547436137, 0.15724606909028371, 0.20418723891729518, 0.081838421988659982, 0.068621638141008745, 0.10220157692769194, 0.082070053849853108, 0.085358552856575232, 0.11122835768829009, 0.035546339827756568, 0.16963808420132109, 0.087214825348524427, 0.043029972014742124, 0.28826706306676475, 0.38327780222349228, 0.35912530609228499, 0.23654524365505736, 0.11315615002209112, 0.34622338310824918, 0.18802630659949149, 0.15443216698815687, 0.15960557209645454, 0.029680186984117229, 0.044523793373498655, 0.14059092326806477, 0.060271424633692172, 0.11858842437870509, 0.12655515524733485, 0.20967414061802736, 0.13884263905022165, 0.15913036613357889, 0.27159122970296767, 0.22228774993265543, 0.20193888467604793]
Not Significant R-Squareds [0.02235294572630786, 0.017594337604473931, 0.013137201475323046, 0.011528460517602923, 0.023176385796554011, 0.021278990010261456, 0.008179430435938273, 0.024872850756441744, 0.019391821145256838, 0.018425319335723001, 0.019359529293771827]
(array([False,  True, False,  True,  True,  True, False], dtype=bool), array([  3.61643929e-01,   6.27799972e-13,   8.07514225e-01,
         2.65317406e-12,   2.01474596e-02,   2.82778343e-23,
         2.42350792e-01]), 0.0073008319790146547, 0.0071428571428571435)
T-test Intercepts Ttest_1sampResult(statistic=0.16316121131504693, pvalue=0.87081897587788593)
T-test Concreteness Ttest_1sampResult(statistic=-1.0220013306030189, pvalue=0.30998051043785746)
T-test WordFreq Ttest_1sampResult(statistic=8.9092864915163847, pvalue=1.7937142049500206e-13)
T-test WordLen Ttest_1sampResult(statistic=0.24447373987440654, pvalue=0.80751422498613312)
T-test MList Ttest_1sampResult(statistic=8.4935459791218477, pvalue=1.1370745989884469e-12)
T-test MPool Ttest_1sampResult(statistic=-2.5887097646122732, pvalue=0.011512834062446133)
T-test Trial No Ttest_1sampResult(statistic=-14.738925143781342, pvalue=4.03969060888311e-24)
T-test Session No Ttest_1sampResult(statistic=-1.3750373685435988, pvalue=0.17310770831724789)
Beta Conc Ave, SE -0.00513979363643 0.00502914573838
Beta WordFreq Ave, SE 0.0438715300452 0.0049242473106
Beta Wordlen Ave, SE 0.00084569280089 0.00345923779513
Beta MList Ave, SE 0.0522265663706 0.00614897081843
Beta MPool Ave, SE -0.0154395697819 0.00596419497967
Beta Trial No Ave, SE -0.189523103129 0.0128586787218
Beta Session No Ave, SE -0.0381114224723 0.027716644903
[-0.0050000000000000001, 0.043999999999999997, 0.001, 0.051999999999999998, -0.014999999999999999, -0.19, -0.037999999999999999] [-0.0050000000000000001, 0.043999999999999997, 0.001, 0.051999999999999998, -0.014999999999999999, -0.19, -0.037999999999999999]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 List Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.005     0.005
     Word Frequency      0.044     0.005
        Word Length      0.001     0.003
List Meaningfulness      0.052     0.006
Pool Meaningfulness     -0.015     0.006
           Trial No      -0.19     0.013
         Session No     -0.038     0.028
 
 
rsq [0.02235294572630786, 0.13266235784791525, 0.04621330968284787, 0.06980727742949577, 0.07405031826011343, 0.06198778047102138, 0.13217443541076568, 0.14088274318554894, 0.12118721502658736, 0.1260686870184753, 0.01759433760447393, 0.06076115536444904, 0.03171746627473537, 0.27870161166887675, 0.12706434835674985, 0.1891638891452847, 0.08639527975408623, 0.013137201475323046, 0.06237536871609228, 0.011528460517602923, 0.2165546022013084, 0.18896186923659908, 0.05048065300143101, 0.032403857610454834, 0.0589062510460725, 0.09966386809254224, 0.24913922129456334, 0.15572912238757242, 0.13686358802403042, 0.04612070128556245, 0.1313323394080752, 0.15705022803818214, 0.20246430713528485, 0.07048358810290212, 0.2354558016802044, 0.05073411931344818, 0.07523229834721845, 0.0488557547436137, 0.02317638579655401, 0.1572460690902837, 0.20418723891729518, 0.08183842198865998, 0.021278990010261456, 0.06862163814100875, 0.10220157692769194, 0.08207005384985311, 0.08535855285657523, 0.11122835768829009, 0.008179430435938273, 0.024872850756441744, 0.03554633982775657, 0.019391821145256838, 0.1696380842013211, 0.08721482534852443, 0.043029972014742124, 0.28826706306676475, 0.3832778022234923, 0.359125306092285, 0.23654524365505736, 0.11315615002209112, 0.3462233831082492, 0.1880263065994915, 0.15443216698815687, 0.15960557209645454, 0.02968018698411723, 0.044523793373498655, 0.14059092326806477, 0.018425319335723, 0.06027142463369217, 0.11858842437870509, 0.12655515524733485, 0.20967414061802736, 0.13884263905022165, 0.019359529293771827, 0.1591303661335789, 0.2715912297029677, 0.22228774993265543, 0.20193888467604793]

"""