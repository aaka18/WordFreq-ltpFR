import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas

# Final Code For Word Recall Full Model

from w2vec.Final  Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
valence_norm = conc_freq_len.valences_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []
predict = []
pvalues = []
residuals = []
f_pvalues = []
fdr_pvalues = []


def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_valence_part = all_valence
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        # print(all_imageability)
        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part_norm = np.array(all_valence_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        # print(len(all_probs_part_norm))
        # print(len(all_concreteness_part))
        # print(len(all_wordfreq_part))
        # print(len(all_wordlen_part))
        # print(len(all_valence_part_norm))
        # print(len(all_arousal_part_norm))
        # print(len(all_imag_part_norm))
        # print(len(all_m_list_part_norm))
        # print(len(all_means_mpool_part_norm))
        #
        # print((all_probs_part_norm))
        # print((all_concreteness_part))
        # print((all_wordfreq_part))
        # print((all_wordlen_part))
        # print((all_valence_part_norm))
        # print((all_arousal_part_norm))
        # print((all_imag_part_norm))
        # print((all_m_list_part_norm))
        # print((all_means_mpool_part_norm))

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part, all_valence_part_norm,
                                     all_m_list_part_norm, all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3],
                                results.pvalues[4], results.pvalues[5], results.pvalues[6]],
                               alpha=0.05,
                               method='fdr_bh', is_sorted=False, returnsorted=False))

        fdr_pvalues.append(fdr_p[0])
        f_pvalues.append(results.f_pvalue)
        residuals.append(results.resid)

    return params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues


params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues = calculate_params()

rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
print(fdr_pvalues.shape)

residuals = np.array(residuals)

rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if f_pvalues[i] < .05:
        rsquareds_sig.append(rsquareds[i])
    else:
        rsquareds_notsig.append(rsquareds[i])
print("Significant R-Squareds", rsquareds_sig)
print("Not Significant R-Squareds", rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

print("R sig", len(rsquareds_sig))
print("R not sig", len(rsquareds_notsig))

sig = np.array(rsquareds_sig)
no = np.array(rsquareds_notsig)

t_test_intercepts = scipy.stats.ttest_1samp(params[:, 0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:, 1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:, 2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:, 3], 0)
t_test_valence = scipy.stats.ttest_1samp(params[:, 4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:, 5], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:, 6], 0)

fdr_correction = (multipletests([t_test_concreteness[1], t_test_wordfreq[1], t_test_wordlen[1],
                                 t_test_valence[1], t_test_mlist[1], t_test_mpool[1]],
                                alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False))
print(fdr_correction)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test Valence", t_test_valence)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:, 1]
y_wordfreq = params[:, 2]
y_wordlen = params[:, 3]
y_valence = params[:, 4]
y_mlist = params[:, 5]
y_mpool = params[:, 6]

beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_mlist = (np.array(params[:, 5]))
beta_mpool = (np.array(params[:, 6]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta Valence Ave, SE", np.mean(beta_valence), stats.sem(beta_valence))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq), 3), round(np.mean(beta_wordlen), 3),
             round(np.mean(beta_valence), 3),
             round(np.mean(beta_mlist), 3), round(np.mean(beta_mpool), 3)]
sem_betas = [round(stats.sem(beta_concreteness), 3), round(stats.sem(beta_wordfreq), 3),
             round(stats.sem(beta_wordlen), 3), round(stats.sem(beta_valence), 3),
             round(stats.sem(beta_mlist), 3), round(stats.sem(beta_mpool), 3)]

predictors = ['Concreteness', 'Word Frequency', 'Word Length', 'Valence',
              'List Meaningfulness', 'Pool Meaningfulness', ]
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names=("Word Recall Model", "Mean Betas", "SEM Betas"),
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
    ax2.scatter(1, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordfreq_sig, betas_sig_wordfreq, marker='o', color='black')
ax2.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
ax2.plot([1.8, 2.2], [mean_beta_wordfreq, mean_beta_wordfreq], linewidth=3, color='black')
if fdr_correction[0][1]:
    ax2.scatter(2, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color='black')
ax2.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
ax2.plot([2.8, 3.2], [mean_beta_wordlen, mean_beta_wordlen], linewidth=3, color='black')
if fdr_correction[0][2]:
    ax2.scatter(3, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_valence_sig, betas_sig_valence, marker='o', color='black')
ax2.scatter(x_valence_nonsig, betas_nonsig_valence, facecolors='none', edgecolors='gray')
ax2.plot([3.8, 4.2], [mean_beta_valence, mean_beta_valence], linewidth=3, color='black')
if fdr_correction[0][3]:
    ax2.scatter(4, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mlist_sig, betas_sig_mlist, marker='o', color='black')
ax2.scatter(x_mlist_nonsig, betas_nonsig_mlist, facecolors='none', edgecolors='gray')
ax2.plot([4.8, 5.2], [mean_beta_mlist, mean_beta_mlist], linewidth=3, color='black')
if fdr_correction[0][4]:
    ax2.scatter(5, 0.34, s=65, marker=(5, 2), color='black')

ax2.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color='black')
ax2.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
ax2.plot([5.8, 6.2], [mean_beta_mpool, mean_beta_mpool], linewidth=3, color='black')
if fdr_correction[0][5]:
    ax2.scatter(6, 0.34, s=65, marker=(5, 2), color='black')

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = 'Concreteness'
labels[2] = 'Frequency'
labels[3] = 'Length'
labels[4] = 'Valence'
labels[5] = 'M List'
labels[6] = 'M Pool'

ax2.set_xticklabels(labels, rotation=18, size=13)
ax2.xaxis.set_ticks_position('none')
ax2.set_ylim(-.3, .41)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)

ax2.set_ylim(-.3, .41)

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

plt.savefig("Final_Word_Recall_NoArousal_Jan9FINAL.pdf")
plt.show()

print("rsq", rsquareds.tolist())
"""
(88, 7)
Significant R-Squareds [0.051845663038716072, 0.032247383452923328, 0.043126273809791282, 0.045751660209730427, 0.050710996922091134, 0.068016945599565171, 0.035477316444522566, 0.066223376133525913, 0.028564444382670917, 0.038993941497246665, 0.09871040340723658, 0.065978045045646416, 0.048124054767580304, 0.1205841533192562, 0.04476463391194152, 0.029565346389720526, 0.035178584318963813, 0.031646770074628128, 0.027801752193128637, 0.029871770559143584, 0.039925850986659683, 0.037192972919374179, 0.023269517287829555, 0.050905424929094667, 0.034603564730138259, 0.048727466766515737, 0.10486565468074016, 0.050216499518160207, 0.050451761421403507, 0.05550148458005999, 0.032163227800812377, 0.052286082019056157, 0.040261716913662093, 0.044267948461808038, 0.067171528455511553, 0.03342211741178891, 0.033105987872992348, 0.034760865837140553, 0.059046232581577374, 0.033130774005121211, 0.056701667577541803, 0.049196871527676711, 0.026550272173860967, 0.05784514393055995, 0.036046303314410943, 0.023180093080513942, 0.036569930986722232, 0.026515611310630294, 0.028675212023714369, 0.026897459983349026, 0.080197597487639993, 0.03320069526856162, 0.03168038719305033, 0.035682703024809026, 0.031350670446015694, 0.026421032434477132, 0.04121978193043585, 0.065293175093689815, 0.048318144745513103, 0.050339765339665443, 0.05318174568145706, 0.027673837326308348, 0.085437579915233819, 0.038935954343903734, 0.050303947137573357, 0.031600723441100098, 0.05627295036983071, 0.049012983283258649, 0.049467398663959772, 0.067750030533353711, 0.024049895399598342]
Not Significant R-Squareds [0.021875790690552344, 0.018176032170796219, 0.0065206258081678126, 0.015740484273196187, 0.018669818459748133, 0.016362806880156233, 0.022071317545247937, 0.013467698157593633, 0.0077910042355932019, 0.01641626334287416, 0.016692148995673395, 0.016225010118427918, 0.013783358431069592, 0.020396810631273521, 0.01524107146757514, 0.0088651796887433409, 0.016352101422665966]
R sig 71
R not sig 17
(array([ True,  True, False,  True,  True,  True], dtype=bool), array([  1.24121448e-03,   1.14616972e-17,   7.32247871e-01,
         8.25667215e-07,   3.57512786e-02,   7.91297696e-28]), 0.008512444610847103, 0.008333333333333333)
T-test Intercepts Ttest_1sampResult(statistic=-10.550782790479282, pvalue=3.0643077363351922e-17)
T-test Concreteness Ttest_1sampResult(statistic=-3.4642454179684297, pvalue=0.00082747631868942723)
T-test WordFreq Ttest_1sampResult(statistic=10.99817991634044, pvalue=3.8205657293733285e-18)
T-test WordLen Ttest_1sampResult(statistic=-0.34323744462106515, pvalue=0.73224787077943287)
T-test Valence Ttest_1sampResult(statistic=5.4775114098574997, pvalue=4.1283360728326066e-07)
T-test MList Ttest_1sampResult(statistic=2.2091321610086316, pvalue=0.029792732173920203)
T-test MPool Ttest_1sampResult(statistic=16.54524495842637, pvalue=1.3188294937172993e-28)
Beta Conc Ave, SE -0.0231092136474 0.00667077844069
Beta WordFreq Ave, SE 0.072417454097 0.00658449440251
Beta Wordlen Ave, SE -0.0017886601769 0.00521114524343
Beta Valence Ave, SE 0.0386071120208 0.00704829422195
Beta MList Ave, SE 0.0102333030564 0.00463227290654
Beta MPool Ave, SE 0.125631205558 0.007593191027
[-0.023, 0.071999999999999995, -0.002, 0.039, 0.01, 0.126] [-0.023, 0.071999999999999995, -0.002, 0.039, 0.01, 0.126]
 
Table 1
Regression Analysis for Variables Predicting Probability of Recall

 Word Recall Model  Mean Betas SEM Betas
------------------- ---------- ---------
       Concreteness     -0.023     0.007
     Word Frequency      0.072     0.007
        Word Length     -0.002     0.005
            Valence      0.039     0.007
List Meaningfulness       0.01     0.005
Pool Meaningfulness      0.126     0.008
 
 
rsq [0.05184566303871607, 0.03224738345292333, 0.021875790690552344, 0.04312627380979128, 0.04575166020973043, 0.050710996922091134, 0.06801694559956517, 0.035477316444522566, 0.06622337613352591, 0.01817603217079622, 0.028564444382670917, 0.038993941497246665, 0.09871040340723658, 0.0065206258081678126, 0.06597804504564642, 0.048124054767580304, 0.1205841533192562, 0.04476463391194152, 0.029565346389720526, 0.03517858431896381, 0.015740484273196187, 0.03164677007462813, 0.018669818459748133, 0.027801752193128637, 0.029871770559143584, 0.016362806880156233, 0.03992585098665968, 0.03719297291937418, 0.023269517287829555, 0.05090542492909467, 0.03460356473013826, 0.04872746676651574, 0.10486565468074016, 0.05021649951816021, 0.05045176142140351, 0.05550148458005999, 0.03216322780081238, 0.022071317545247937, 0.013467698157593633, 0.05228608201905616, 0.04026171691366209, 0.007791004235593202, 0.04426794846180804, 0.06717152845551155, 0.03342211741178891, 0.03310598787299235, 0.03476086583714055, 0.059046232581577374, 0.03313077400512121, 0.0567016675775418, 0.04919687152767671, 0.01641626334287416, 0.026550272173860967, 0.05784514393055995, 0.036046303314410943, 0.016692148995673395, 0.01622501011842792, 0.02318009308051394, 0.03656993098672223, 0.026515611310630294, 0.02867521202371437, 0.026897459983349026, 0.08019759748764, 0.013783358431069592, 0.03320069526856162, 0.03168038719305033, 0.035682703024809026, 0.031350670446015694, 0.026421032434477132, 0.04121978193043585, 0.06529317509368981, 0.0483181447455131, 0.05033976533966544, 0.02039681063127352, 0.05318174568145706, 0.027673837326308348, 0.08543757991523382, 0.038935954343903734, 0.05030394713757336, 0.01524107146757514, 0.0316007234411001, 0.05627295036983071, 0.008865179688743341, 0.016352101422665966, 0.04901298328325865, 0.04946739866395977, 0.06775003053335371, 0.024049895399598342]

 
"""