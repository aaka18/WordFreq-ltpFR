import numpy as np
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec import Mpool10th_percentile as m_pool_10
from w2vec.Final import P_rec_ltpFR2_by_list as prbl

nan = np.nan

# List of lists with all participants; 552 p_recs of lists per participant

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_probs_imported = prbl.get_ltpFR2_precall(files_ltpFR2)
m_pool_10 = m_pool_10.get_m_pool(files_ltpFR2)

all_probs = np.array(all_probs_imported)
m_pool_10 = np.array(m_pool_10)



all_probs_final = []
m_pool_10_final = []
slopes_participants = []

# For every participant, get their lists' values (552), mask NaNs from P_recs, find the slope through the 552 lists, add slopes into a new list
# Conduct a t-test to the slopes to see whether they're different from a zero distribution

for i in range(len(all_probs)):
    all_probs_part = all_probs[i]
    all_list_ten_part = m_pool_10[i]

    mask = np.isfinite(all_probs_part)
    all_probs_part = np.array(all_probs_part)[mask]
    all_list_ten_part = np.array(all_list_ten_part)[mask]

    #print(len(all_probs_part))
    #print(len(all_num_top_ten_part))

    slope, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_list_ten_part, all_probs_part)
    print(slope)

    slopes_participants.append(slope)
    all_probs_final.append(all_probs_part)
    m_pool_10_final.append(all_list_ten_part)


#print((np.array(all_probs_final)))
#print((np.array(all_num_top_ten_final)))
print(len(slopes_participants))
print(slopes_participants)

t_test = scipy.stats.ttest_1samp(slopes_participants, 0, axis=0)
print(t_test)


# Create data
N = len(all_probs)
x = np.zeros(N)
y = slopes_participants

# Plot
plt.scatter(x, y, alpha=0.5)
#plt.title('Slopes')
plt.title("Slopes: Number of Highly Meaningful Words (MPool) In Each List & P-Rec")
plt.xticks([])
#plt.title("Slopes: Semantic Meaningfulness of List & P-Rec")
plt.show()

"""Output:
"""