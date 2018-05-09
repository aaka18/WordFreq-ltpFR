"""2.	LIST RECALL MODEL II
{Each of these vectors will have a dimension of N participants * 552 lists}
1)	Average Word Frequency Of The List - all_mean_wordfreq_participants
2)	Average Word Concreteness Of The List - all_mean_conc_participants
3)	Average Word Length Of The List - all_mean_wordlen_participants
4)	Mlist (Each word’s aver. similarity to all other words in the list) 10th percentile - m_list_ten
5)	Mpool (Each word’s aver. similarity to all other words in the pool) 10th percentile - m_pool_ten
6) P_rec per list

"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats

from w2vec import Mlist10th_percentile as m_list
from w2vec import Mpool10th_percentile as m_pool
from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, P_rec_ltpFR2_by_list as p_rec

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(files_ltpFR2)
m_list_larger_90 = m_list.get_larger_90(files_ltpFR2)
m_list = m_list.get_m_list(m_list_larger_90, files_ltpFR2)
all_means_mpool = m_pool.get_m_pool(files_ltpFR2)


all_probs = np.array(p_rec_list)
all_concreteness= np.array(all_mean_conc_participants)
all_wordfreq = np.array(all_mean_wordfreq_participants)
all_wordlen = np.array(all_mean_wordlen_participants)
all_m_list = np.array(m_list)
all_means_mpool = np.array(all_means_mpool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
all_means_mpool_part = []

# print("P_rec_list", p_rec_list)
# print("Shape of P_rec_list", np.shape(p_rec_list))
# print(" ")
# print("Aver conc per list", all_mean_conc_participants)
# print("Shape of aver conc per list", np.shape(all_mean_conc_participants))
# print("  ")
# print("Aver freq per list", all_mean_wordfreq_participants)
# print("Shape of aver conc per list", np.shape(all_mean_wordfreq_participants))
# print("  ")
# print("Aver word len per list", all_mean_wordlen_participants)
# print("Shape of aver word len per list", np.shape(all_mean_wordlen_participants))
# print("  ")
# print("M list per list", m_list)
# print("Shape of m list", np.shape(m_list))
# print("  ")
# print("M pool per list", m_pool)
# print("Shape of m pool", np.shape(m_pool))

params = []
rsquareds = []
pvalues = []
predict = []
f_pvalues = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordfreq_part = all_wordfreq[i]
        all_wordlen_part = all_wordlen[i]
        all_m_list_part = all_m_list[i]
        all_means_mpool_part = all_means_mpool[i]

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


        # all_probs_part_norm = stats.mstats.zscore(all_probs_part, ddof=1)
        # all_concreteness_part_norm = stats.mstats.zscore(all_concreteness_part, ddof=1)
        # all_wordfreq_part_norm = stats.mstats.zscore(all_wordfreq_part, ddof=1)
        # all_wordlen_part_norm = stats.mstats.zscore(all_wordlen_part, ddof=1)
        # all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, ddof=1)
        # all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part,  ddof=1)

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,
                                    all_m_list_part, all_means_mpool_part))

        # x_values = np.column_stack((all_concreteness_part_norm, all_wordfreq_part_norm, all_wordlen_part_norm , all_m_list_part_norm, all_means_mpool_part_norm))
        # # x-values without the m_list
        # x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        # print(x_values)
        # print(np.shape(x_values))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        #y_value = all_probs_part_norm
        y_value = all_probs_part
        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(np.array(results))
        print(results.summary())
        print(results.rsquared)

        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        f_pvalues.append(results.f_pvalue)

    return params, rsquareds, predict, pvalues, f_pvalues

params, rsquareds, predict, pvalues, f_pvalues = calculate_params()
print("Parameters of List Recall Regression Model II (X-intercept, Conc coeff, WordFreq coeff,WordLen coeff,  MPool 10,  MList 10%)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)
pvalues = np.array(pvalues)
f_pvalues = np.array(f_pvalues)

print("F pvalues", f_pvalues)

rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if f_pvalues[i] < .05:
        rsquareds_sig.append(f_pvalues[i])
    else:
        rsquareds_notsig.append(f_pvalues[i])
print(rsquareds_sig)
print(rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

plt.title("R-Squared Values of List Recall Model II")
plt.scatter(x_rsq_sig, rsquareds_sig ,marker='o', color = 'black', label = " p < 0.05" )
plt.scatter(x_rsq_nonsig, rsquareds_notsig,  facecolors='none', edgecolors='gray', label = "Not Significant" )
plt.ylim(-.01, .5)

plt.xticks([])
plt.legend()


t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mlist_10 = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool_10 = scipy.stats.ttest_1samp(params[:,5], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist_10)
print("T-test MPool", t_test_mpool_10)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordfreq = params[:,2]
y_wordlen = params[:,3]
y_mlist_10 = params[:,4]
y_mpool_10 = params[:,5]

print("Y concreteness", y_concreteness)
print("Y wordfreq", y_wordfreq)
print("Y wordlen", y_wordlen)
print("Y mlist", y_mlist_10)
print("Y mpool", y_mpool_10)

beta_concreteness = (np.array(params[:,1]))
beta_wordfreq = (np.array(params[:,2]))
beta_wordlen =(np.array(params[:,3]))
beta_mlist_10 = (np.array(params[:,4]))
beta_mpool_10 = (np.array(params[:,5]))

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist_10), stats.sem(beta_mlist_10))
print("Beta MPool Ave, SE", np.mean(beta_mpool_10), stats.sem(beta_mpool_10))

print("P pvalues", pvalues)
print("Params", params)

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3), round(np.mean(beta_mlist_10),3), round(np.mean(beta_mpool_10),3)]
sem_betas = [round(stats.sem(beta_concreteness),3),  round(stats.sem(beta_wordfreq),3), round(stats.sem(beta_wordlen),3), round(stats.sem(beta_mlist_10),3),round(stats.sem(beta_mpool_10),3) ]
predictors = ['Concreteness',  'Word Frequency','Word Length', 'List Meaningfulness 10%', 'Pool Meaningfulness 10%']
print(ave_betas, ave_betas)

t = Table([predictors, ave_betas, sem_betas], names = ("List Recall Model II", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))
print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")
betas_sig_conc = []
betas_nonsig_conc = []
for i in range(len(params[:,1])):
    if pvalues[:,1][i] < .05:
        betas_sig_conc.append(params[:,1][i])
    else:
        betas_nonsig_conc.append(params[:,1][i])

betas_sig_wordfreq = []
betas_nonsig_wordfreq = []
for i in range(len(params[:,2])):
    if pvalues[:,2][i] < .05:
        betas_sig_wordfreq.append(params[:,2][i])
    else:
        betas_nonsig_wordfreq.append(params[:,2][i])

betas_sig_wordlen = []
betas_nonsig_wordlen = []
for i in range(len(params[:,3])):
    if pvalues[:,3][i] < .05:
        betas_sig_wordlen.append(params[:,3][i])
    else:
        betas_nonsig_wordlen.append(params[:,3][i])



betas_sig_mlist = []
betas_nonsig_mlist = []
for i in range(len(params[:,4])):
    if pvalues[:,4][i] < .05:
        betas_sig_mlist.append(params[:,4][i])
    else:
        betas_nonsig_mlist.append(params[:,4][i])


betas_sig_mpool = []
betas_nonsig_mpool = []
for i in range(len(params[:,5])):
    if pvalues[:,5][i] < .05:
        betas_sig_mpool.append(params[:,5][i])
    else:
        betas_nonsig_mpool.append(params[:,5][i])


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


fig, ax = plt.subplots()
fig.canvas.draw()

plt.title("Beta Values of List Recall Model II")
plt.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
plt.scatter(x_conc_nonsig, betas_nonsig_conc,  facecolors='none', edgecolors='gray')

plt.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black', )
plt.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')

plt.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black')
plt.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')

plt.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color = 'black')
plt.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')

plt.scatter(x_mpool_sig, betas_sig_mpool, marker='o' , color = 'black')
plt.scatter(x_mpool_nonsig, betas_nonsig_mpool,facecolors='none', edgecolors='gray')


# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Concreteness'
labels[3] = 'Word Freq'
labels[5] = 'Word Length'
labels[7] = 'M List 10%'
labels[9] = 'M Pool 10%'

ax.set_xticklabels(labels)
ax.xaxis.set_ticks_position('none')

print("Rsq", rsquareds.tolist())

plt.axhline(y=0, color='gray', linestyle='--')
plt.show()

"""
T-test Concreteness Ttest_1sampResult(statistic=0.52983174020549173, pvalue=0.59779362454561391)
T-test WordFreq Ttest_1sampResult(statistic=7.3528282950056028, pvalue=2.0137081009630842e-10)
T-test WordLen Ttest_1sampResult(statistic=-0.28450838876345763, pvalue=0.77680488320972996)
T-test MList Ttest_1sampResult(statistic=1.2589922420371424, pvalue=0.2119376712310502)
T-test MPool Ttest_1sampResult(statistic=2.9911258790740147, pvalue=0.0037593649078834003)


Beta Conc Ave, SE 0.00223433565629 0.00421706645099
Beta WordFreq Ave, SE 0.0392993817093 0.00534479796515
Beta Wordlen Ave, SE -0.000966104142979 0.00339569651066
Beta MList Ave, SE 0.00666252936685 0.00529195426659
Beta MPool Ave, SE 0.0129354857446 0.00432462098474

R-squareds [0.0082574358919518254, 0.0042348012776325383, 0.0027102390305421276, 0.021751744976822196, 0.0051202026493474273, 0.00091612542286745313, 0.0034881122142602639, 0.011301865384469534, 0.010926518764458715, 0.0054783274596921228, 0.0084784506632149537, 0.014764607418083608, 0.0010006516083443939, 0.010224655484158296, 0.0034221411241376343, 0.0071346922930111134, 0.016094989678579585, 0.0024630871856239356, 0.011385844600353856, 0.0037881073802726872, 0.022688801804353398, 0.0045454557026136655, 0.0082426245240929674, 0.011601478692765288, 0.0055176210806766601, 0.0045597499607383796, 0.011446400616133112, 0.0069873475850305011, 0.0087053163133871125, 0.0078127550763077425, 0.0086728435724094943, 0.013147878297298909, 0.0096727570164109045, 0.015630660424637122, 0.01289107580465354, 0.010221116572985189, 0.013760940357541718, 0.0050622744550412602, 0.0053190109096985516, 0.0087616024683401239, 0.0044692818978802062, 0.0074840429512476048, 0.0019288320567840689, 0.0045697155893114072, 0.0052960409667993646, 0.0062023149679962808, 0.0032410959815796225, 0.0033914142192810193, 0.0076052725939603771, 0.0021711262560148992, 0.011471895873647053, 0.0099505450517608507, 0.013894917918379712, 0.024449258357805981, 0.0066972449333378981, 0.006034379353434538, 0.0039908214846176016, 0.0028717050650614029, 0.01158612505383616, 0.014856494625890559, 0.0045415960583651271, 0.0064150912406503169, 0.010982290064321054, 0.0079291105567582676, 0.021510173096997653, 0.0029464746370492234, 0.0051052090314148701, 0.0098377362385891898, 0.0017334020040177256, 0.0040320457971592116, 0.004132536238579898, 0.00092851011423322749, 0.0082351996333225319, 0.019472516502486381, 0.021792077919670416, 0.0066387402926155747]
(76,)
F pvalues [ 0.48457082  0.802814    0.9146952   0.03423817  0.72917281  0.99205698
  0.86103762  0.28512369  0.30474667  0.69881134  0.45854167  0.14848442
  0.99033681  0.34533948  0.86591215  0.56199889  0.11379665  0.93382257
  0.2957418   0.83879647  0.02781354  0.77735321  0.47567771  0.27019384
  0.69547621  0.77616941  0.27887906  0.57282362  0.44242375  0.50787302
  0.44470837  0.2029672   0.37901648  0.12497508  0.21304092  0.34445218
  0.18055214  0.73406962  0.71233054  0.43848164  0.78364506  0.53327777
  0.95783415  0.77534353  0.71427862  0.63760645  0.88068995  0.86816461
  0.52383296  0.94583505  0.27760852  0.36067505  0.175949    0.01871854
  0.60049616  0.65172698  0.82237454  0.91143441  0.27300587  0.14581612
  0.77835752  0.61982556  0.30176556  0.49903996  0.03610964  0.90363807
  0.73122721  0.36760228  0.96650785  0.8191009   0.8110658   0.99180524
  0.47622342  0.05621579  0.0339348   0.60229493]
  
"""