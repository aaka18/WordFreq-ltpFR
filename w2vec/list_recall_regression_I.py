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

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, MPool_byPoolSim as m_pool, \
    P_rec_ltpFR2_by_list as p_rec, semantic_similarity_bylist as m_list

# Importing all of the data
files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP22*.mat')
p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = conc_freq_len.measures_by_list(files_ltpFR2)
all_means, all_sems, m_list = m_list.all_parts_list_correlations(files_ltpFR2)
all_means_mpool = m_pool.all_parts_list_correlations(files_ltpFR2)

all_probs = np.array(p_rec_list)
print(np.shape(all_probs))
all_concreteness= np.array(all_mean_conc_participants)
print(np.shape(all_concreteness))
all_wordlen = np.array(all_mean_wordlen_participants)
print(np.shape(all_wordlen))
all_wordfreq = np.array(all_mean_wordfreq_participants)
print(np.shape(all_wordfreq))
all_m_list = np.array(m_list)
print(np.shape(all_m_list))
all_means_mpool = np.array(all_means_mpool)
print(np.shape(all_means_mpool))

all_probs_part = []
all_concreteness_part = []
all_wordlen_part = []
all_wordfreq_part = []
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

def calculate_params():
    # for each participant (= each list in list of lists)
    # get the list values f
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        all_concreteness_part = all_concreteness[i]
        all_wordlen_part = all_wordlen[i]
        all_wordfreq_part = all_wordfreq[i]
        all_m_list_part = all_m_list[i]
        all_means_mpool_part = all_means_mpool[i]

        # mask them since some p_recs have nan's

        mask = np.isfinite(all_probs_part)
        print("Mask", mask)
        print(np.shape(np.array(mask)))
        all_probs_part = np.array(all_probs_part)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_m_list_part = np.array(all_m_list_part)[mask]
        all_means_mpool_part = np.array(all_means_mpool_part)[mask]


        sm.OLS

        # add the predictor values as columns of the x-values
        x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_m_list_part, all_means_mpool_part))

        # x-values without the m_list
        #x_values = np.column_stack((all_concreteness_part, all_wordlen_part, all_wordfreq_part, all_means_mpool_part))

        #print(x_values)
        #print(np.shape(x_values))

        # also add the constant
        x_values = sm.add_constant(x_values)

        # y-value is the independent variable = probability of recall
        y_value = all_probs_part

        # run the regression model for each participant
        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(np.array(results))
        print(results.summary())
        print(results.rsquared)

        params.append(results.params)

        rsquareds.append(results.rsquared)
    return params, rsquareds

params, rsquareds = calculate_params()
print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff, MList coeff, MPool coeff)", params)
#print("Parameters of List Recall Regression Model I (X-intercept, Conc coeff, WordLen coeff, WordFreq coeff,  MPool coeff)", params)

# turn list of lists into a np array
print("R-squareds", rsquareds)
rsquareds = np.array(rsquareds)
print(np.shape(rsquareds))
params = np.array(params)

# t-test between slopes of each participant for each parameter & 0
t_test_intercepts = scipy.stats.ttest_1samp(params[:,0], 0)
t_test_concreteness = scipy.stats.ttest_1samp(params[:,1], 0)
t_test_wordlen = scipy.stats.ttest_1samp(params[:,2], 0)
t_test_wordfreq = scipy.stats.ttest_1samp(params[:,3], 0)
t_test_mlist = scipy.stats.ttest_1samp(params[:,4], 0)
t_test_mpool = scipy.stats.ttest_1samp(params[:,5], 0)
#t_test_mpool = scipy.stats.ttest_1samp(params[:,4], 0)

print("T-test Intercepts", t_test_intercepts)
print("T-test Concreteness", t_test_concreteness)
print("T-test WordLen", t_test_wordlen)
print("T-test WordFreq", t_test_wordfreq)
print("T-test MList", t_test_mlist)
print("T-test MPool", t_test_mpool)

# MAKE SCATTER PLOTS

# Create data
N = len(all_probs)
x = np.zeros(N)
y_concreteness = params[:,1]
y_wordlen = params[:,2]
y_wordfreq = params[:,3]
y_mlist = params[:,4]
y_mpool = params[:,5]
#y_mpool = params[:,4]

print("Y concreteness", y_concreteness)
print("Y wordlen", y_wordlen)
print("Y wordfreq", y_wordfreq)
print("Y mlist", y_mlist)
print("Y mpool", y_mpool)



# Plot
plt.scatter(x, y_concreteness, alpha=0.5)
plt.title("Slopes/Beta Values Concreteness (List)")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordlen, alpha=0.5)
plt.title("Slopes/Beta Values WordLen (List)")
plt.xticks([])
plt.show()

plt.scatter(x, y_wordfreq, alpha=0.5)
plt.title("Slopes/Beta Values Word Freq (List)")
plt.ylim(-.0001, .0001)
plt.xticks([])
plt.show()

plt.scatter(x, y_mlist, alpha=0.5)
plt.title("Slopes/Beta Values List Meaningfulness (List)")
plt.xticks([])
plt.show()

plt.scatter(x, y_mpool, alpha=0.5)
plt.title("Slopes/Beta Values Pool Meaningfulness List")
plt.xticks([])
plt.show()

plt.scatter(x, rsquareds, alpha=0.5)
plt.title("R-Squares - List Recall Model I")
plt.xticks([])
plt.show()


# plt.scatter(x, y_mlist_ten, alpha=0.5)
# plt.title("Slopes: Highly Meaningful (List) Words & P-Rec")
# plt.xticks([])
# plt.show()

# plt.scatter(x, y_mpool_ten, alpha=0.5)
# plt.title("Slopes: Highly Meaningful (Pool) Words & P-Rec")
# plt.xticks([])
# plt.show()




"""Notes on how to do regression model
For each row/each participant:
Concatenate vector (np.concatenate) and add each of them to columns of x
Transverse them (axis = o) (552 rows * 5 predictors)
then add sm.add_constant(x)
write y = all_probs_part
run the model model = sm.OLS(y, x) → gives
results = model.fit()
results.params → matrix(one
column

Six value for each subject, append them to a betas list (6 times 71 participants)
t-test whole matrix over all the predictors
p-value/t-value for each predictor

sm.OLS

x = matrix of all predictors
y = p_rec (values predicted)
observations as rows
predictors as columns

sm.add_constant(x)
model = sm.OLS(y, x) → gives the model
results = model.fit()
results.params → matrix (one column for each predictor) x intercept+ 5 columns for 5 predictors
"""


"""
Output is:

R-squareds [0.011548334772991109, 0.0054420681833505791, 0.0025315752186695351, 0.021939405709030302, 0.016250249194673327, 0.0027641353902198018, 0.0030035734183189167, 0.013366349466248706, 0.01753449610443325, 0.009704360527275413, 0.0084865835045419002, 0.020560989754005998, 0.0011128523753243247, 0.015083768902978023, 0.0065393459895597417, 0.00886375498867642, 0.020546526973825197, 0.0025390194249503839, 0.0170365584538269, 0.0070889471012490768, 0.044485367904616768, 0.004029758841297082, 0.0085671247882153079, 0.010445861259299805, 0.005377268690253123, 0.0042025117741860685, 0.016061810972155532, 0.025053895964420092, 0.0091227792757937465, 0.0070199921972333712, 0.012224439665123876, 0.0145135868897871, 0.0087005472746408685, 0.014489165194342957, 0.011935543800810744, 0.0104494190039921, 0.0070040483999810332, 0.0039889181110837058, 0.0028078721354189984, 0.010515570005823593, 0.0061260267399129154, 0.0080257563299173107, 0.00096369650583094657, 0.0057633452026897292, 0.0069163247844971076, 0.035832946896958817, 0.0052382641124950613, 0.012503845347357045, 0.0085178404700243515, 0.0025467424764979985, 0.015898571432928166, 0.0098148030857668767, 0.019456360322665289, 0.034295138397935632, 0.0061564192365211046, 0.010069249433228178, 0.0056680150270803642, 0.0030867555879104946, 0.01095961002463619, 0.0129583815928348, 0.005257096022168728, 0.005826223850455281, 0.01215983575654267, 0.0043640187955363841, 0.030950440978089855, 0.0019665417539489471, 0.0068309078697674774, 0.012779897655050387, 0.019870953805372515, 0.0064398320591174674, 0.0043936613850336093, 0.0072190488632114524, 0.0054326664136734593, 0.01543082369712423, 0.0066077095586157819]

T-test Intercepts Ttest_1sampResult(statistic=12.357067064682258, pvalue=1.1455376411848381e-19)
T-test Concreteness Ttest_1sampResult(statistic=-1.593996196321811, pvalue=0.11519948964747011)
T-test WordLen Ttest_1sampResult(statistic=-0.17711439476737312, pvalue=0.85990241999343642)
T-test WordFreq Ttest_1sampResult(statistic=7.5204120736071056, pvalue=1.0407453051606479e-10)
T-test MList Ttest_1sampResult(statistic=6.1289728392541596, pvalue=3.9332084919011747e-08)
T-test MPool Ttest_1sampResult(statistic=-2.5949627388834329, pvalue=0.011400653435620264)"""