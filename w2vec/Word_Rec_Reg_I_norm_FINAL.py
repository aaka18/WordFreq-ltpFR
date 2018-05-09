"""1. WORD RECALL MODEL I
{Each of these vectors will have a dimension of 576 words}
1)	Word Frequency -
2)	Word Concreteness -
3)	Word Length -
4)	Mlist (Each word’s aver. similarity to all other words in the list) - m_list
5)	Mpool (Each word’s aver. similarity to all other words in the pool) - m_pool
6) P_rec each_word - p_rec_word

"""
from scipy.stats.stats import pearsonr
import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table,Column


import statsmodels.api as sm
import statsmodels.formula.api as smf

#from w2vec 
import P_rec_ltpFR2_words as p_rec_word
#from w2vec 
import Concreteness_Freq_WordLength as conc_freq_len
#from w2vec 
import semanticsim_each_word as m_list
#from w2vec 
import ltpFR2_word_to_allwords as m_pool


files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_ltpFR2

all_probs = np.array(p_rec_word)
all_concreteness= np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []
predict = []
pvalues = []
residuals = []
f_pvalues = []

def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)


        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part,  all_m_list_part_norm, all_means_mpool_part_norm))

        # x-values without the m_list
        #x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        #print(x_values)
        #print(np.shape(x_values))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()


        #print(np.array(results))
        print(results.summary())
        #print(results.rsquared)
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict())
        pvalues.append(results.pvalues)
        f_pvalues.append(results.f_pvalue)
        #print(results.resid)
        residuals.append(results.resid)

    return params, rsquareds, predict, pvalues, residuals, f_pvalues


params, rsquareds, predict, pvalues, residuals, f_pvalues = calculate_params()
# print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordFreq coeff, WordLen coeff, MList coeff, MPool coeff)", params)
# print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
residuals = np.array(residuals)
# print("Residuals", residuals)
# print("F pvalues", f_pvalues)

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

plt.title("R-Squared Values of Word Recall Model I")
plt.scatter(x_rsq_sig, rsquareds_sig, marker='o', color = 'black' , label = " p < 0.05" )
plt.scatter(x_rsq_nonsig, rsquareds_notsig, facecolors='none', edgecolors='gray', label = "Not Significant" )

plt.xticks([])
plt.legend()
plt.savefig("Rsq_Word_Rec_Model.pdf")


t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,3], 0)
#t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,5], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordFreq", t_test_wordfreq)
print("T-test WordLen", t_test_wordlen)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)



# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordfreq = params[:,2]
y_wordlen = params[:,3]
y_mlist = params[:,4]
y_mpool = params[:,5]

print("Y concreteness", y_concreteness)
print("Y wordfreq", y_wordfreq)
print("Y wordlen", y_wordlen)
print("Y mlist", y_mlist)
print("Y mpool", y_mpool)

beta_concreteness = (np.array(params[:,1]))
beta_wordfreq = (np.array(params[:,2]))
beta_wordlen =(np.array(params[:,3]))
beta_mlist = (np.array(params[:,4]))
beta_mpool = (np.array(params[:,5]))

mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Beta Conc Ave, SE", np.mean(beta_concreteness), stats.sem(beta_concreteness))
print("Beta WordFreq Ave, SE", np.mean(beta_wordfreq), stats.sem(beta_wordfreq))
print("Beta Wordlen Ave, SE", np.mean(beta_wordlen), stats.sem(beta_wordlen))
print("Beta MList Ave, SE", np.mean(beta_mlist), stats.sem(beta_mlist))
print("Beta MPool Ave, SE", np.mean(beta_mpool), stats.sem(beta_mpool))

print("P pvalues", pvalues)
print("Params", params)
print("P-values of models")

ave_betas = [round(np.mean(beta_concreteness), 3), round(np.mean(beta_wordfreq),3), round(np.mean(beta_wordlen), 3),  round(np.mean(beta_mlist),3), round(np.mean(beta_mpool),3)]
sem_betas = [round(stats.sem(beta_concreteness),3), round(stats.sem(beta_wordfreq),3), round(stats.sem(beta_wordlen),3),  round(stats.sem(beta_mlist),3),round(stats.sem(beta_mpool),3) ]
predictors = ['Concreteness', 'Word Frequency', 'Word Length',  'List Meaningfulness', 'Pool Meaningfulness']
print(ave_betas, ave_betas)

print(" ")
t = Table([predictors, ave_betas, sem_betas], names = ("Word Recall Model I", "Mean Betas", "SEM Betas"), dtype=('str', 'float', 'float'))

print('\033[1m' + "Table 1")
print("Regression Analysis for Variables Predicting Probability of Recall")
print('\033[0m')
print(t)
print(" ")
print(" ")
plt.savefig("Table_Word_Rec_Model.pdf")

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

plt.title("Beta Values of Word Recall Model I")
#plt.scatter(x_conc, y_concreteness, alpha=0.5)
plt.scatter(x_conc_sig, betas_sig_conc, marker='o', color = 'black' )
plt.scatter(x_conc_nonsig, betas_nonsig_conc, facecolors='none', edgecolors='gray' )
plt.scatter(1, mean_beta_conc, s =80, marker = (5,2), color = 'black' )

plt.scatter(x_wordfreq_sig, betas_sig_wordfreq,  marker='o', color = 'black'  )
plt.scatter(x_wordfreq_nonsig, betas_nonsig_wordfreq, facecolors='none', edgecolors='gray')
plt.scatter(2, mean_beta_wordfreq, s =80, marker = (5,2), color = 'black' )

plt.scatter(x_wordlen_sig, betas_sig_wordlen, marker='o', color = 'black' )
plt.scatter(x_wordlen_nonsig, betas_nonsig_wordlen, facecolors='none', edgecolors='gray')
plt.scatter(3, mean_beta_wordlen, s =80, marker = (5,2), color = 'black' )

plt.scatter(x_mlist_sig, betas_sig_mlist, marker='o' , color = 'black' )
plt.scatter(x_mlist_nonsig, betas_nonsig_mlist,facecolors='none', edgecolors='gray')
plt.scatter(4, mean_beta_mlist, s =80, marker = (5,2), color = 'black' )

plt.scatter(x_mpool_sig, betas_sig_mpool, marker='o', color = 'black' )
plt.scatter(x_mpool_nonsig, betas_nonsig_mpool, facecolors='none', edgecolors='gray')
plt.scatter(5, mean_beta_mpool, s =80, marker = (5,2), color = 'black' )

# plt.scatter(x_wordfreq, y_wordfreq, alpha=0.5)
# plt.scatter(x_wordlen, y_wordlen, alpha=0.5)
# plt.scatter(x_mlist, y_mlist, alpha=0.5)
# plt.scatter(x_mpool, y_mpool, alpha=0.5)


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Concreteness'
labels[3] = 'Word Freq'
labels[5] = 'Word Length'
labels[7] = 'M List'
labels[9] = 'M Pool'

ax.set_xticklabels(labels)
ax.xaxis.set_ticks_position('none')


# print("Rsq", rsquareds.tolist())
#
# my_bins = np.arange(0, 1, .025).tolist()
#
# plt.hist(rsquareds , bins = my_bins)
# plt.title("Multiple Regressions")
# plt.xlabel("R-Squared Value")
# plt.ylabel("Frequency")

plt.axhline(y=0, color='gray', linestyle='--')
plt.show()
plt.savefig("Betas_Word_Rec_Model.pdf")
print("rsq", rsquareds.tolist())


"""
N = 76

T-test Intercepts Ttest_1sampResult(statistic=-0.24252938669733309, pvalue=0.80903199928556435)
T-test Concreteness Ttest_1sampResult(statistic=-3.6763960696775571, pvalue=0.00044202218402400245)
T-test WordFreq Ttest_1sampResult(statistic=11.440247262086499, pvalue=4.0971524850590462e-18)
T-test WordLen Ttest_1sampResult(statistic=-0.77289050288527816, pvalue=0.44201620728378033)
T-test MList Ttest_1sampResult(statistic=2.4873313193806399, pvalue=0.015092652137559044)
T-test MPool Ttest_1sampResult(statistic=13.205846109459262, pvalue=2.9498056262354163e-21)

F pvalues [0.0069022944286167304, 0.26513627328278649, 0.00010720223853809549, 6.170806390150392e-05, 0.00046586018663840744, 0.0097916596737174025, 0.001217074501182285, 0.061838147873134561, 0.00012676312720643303, 0.018507610620107084, 4.7358107540999932e-05, 0.0022775235089380116, 0.0010525400012348462, 1.0550914316018963e-06, 0.0020471961571603085, 0.00065934538330889416, 0.027893514822790233, 0.1232590839903196, 0.0029135677429922378, 0.037750935555364941, 4.0073151544212687e-06, 0.18202082652098375, 0.0028420427551321321, 0.32932007932763424, 0.004879154027631334, 0.038544655241009057, 0.047143193445653167, 0.011175646961706368, 0.003980312673497079, 0.063891745482808487, 0.011012777993880967, 0.001284381791567009, 0.17604757226269308, 4.1927585681316252e-07, 0.00051946023642213503, 0.012487754109951756, 0.19748354315159861, 6.9913212426152074e-05, 0.0025353235131181486, 6.2385297384322076e-05, 0.95062318830998871, 0.022568710072007437, 0.038373218333604422, 0.028519528671732523, 0.0052623935213965412, 0.0088431078537475772, 0.00023148404466857481, 0.1068476027902308, 0.55048778751746419, 0.12570081387658144, 0.13584604811806283, 0.0011266448348853507, 0.0032664164566585732, 1.9515540489038591e-07, 0.011879230304455259, 0.00024184362274147069, 0.31143935674475276, 0.27167041539356362, 2.8863292277336462e-06, 0.0089515418947931687, 0.028441214081694092, 0.067454081953925771, 0.000136449005403692, 0.0034621537074018397, 0.42944429605620182, 0.0061757638418609056, 0.0028429726195273153, 0.18968701659687992, 0.00023110821409350369, 0.0047294331627921442, 0.70270407826389603, 0.016543394296696219, 0.32952771225975025, 0.0050431017362067279, 1.8978147487268534e-06, 0.0006672306467817268]


Beta Conc Ave, SE -0.0260882348742 0.00709614371786
Beta WordFreq Ave, SE 0.0813706725268 0.00711266729316
Beta Wordlen Ave, SE -0.00447888691854 0.005794982474
Beta MList Ave, SE 0.0129564148569 0.00520896221423
Beta MPool Ave, SE 0.0996263532943 0.00754410981837

[0.26513627328278649, 0.061838147873134561, 0.1232590839903196, 0.18202082652098375, 0.32932007932763424, 0.063891745482808487, 0.17604757226269308, 0.19748354315159861, 0.95062318830998871, 0.1068476027902308, 0.55048778751746419, 0.12570081387658144, 0.13584604811806283, 0.31143935674475276, 0.27167041539356362, 0.067454081953925771, 0.42944429605620182, 0.18968701659687992, 0.70270407826389603, 0.32952771225975025]


"""