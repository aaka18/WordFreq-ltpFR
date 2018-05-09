import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec.Final import P_rec_ltpFR2_words as prbl, semanticsim_each_word as sem_list

nan = np.nan

# List of lists with all participants; 552 p_recs of lists per participant

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_probs_imported = prbl.get_ltpFR2_precall(files_ltpFR2)
word_id_all_participants = sem_list.all_parts_list_correlations(files_ltpFR2)

all_probs = np.array(all_probs_imported)
all_sem_list = np.array(word_id_all_participants)

all_probs_final = []
all_sems_final = []
slopes_participants = []

print(np.shape(all_sem_list))



# For every participant, get their lists' values (552), mask NaNs from P_recs, find the slope through the 552 lists, add slopes into a new list
# Conduct a t-test to the slopes to see whether they're different from a zero distribution

for i in range(len(all_probs)):
    all_probs_part = all_probs[i]
    all_sem_list_part = all_sem_list[i]

    mask = np.isfinite(all_probs_part)
    all_probs_part = np.array(all_probs_part)[mask]
    all_sem_list_part = np.array(all_sem_list_part)[mask]

    #print(len(all_probs_part))
    #print(len(all_num_top_ten_part))

    slope, intercept, r_value, p_value, std_err = scipy.stats.mstats.linregress(all_sem_list_part, all_probs_part)
    print(slope)

    slopes_participants.append(slope)
    all_probs_final.append(all_probs_part)
    all_sems_final.append(all_sem_list_part)


#print((np.array(all_probs_final)))
#print((np.array(all_num_top_ten_final)))
print(len(slopes_participants))
print(slopes_participants)

t_test = scipy.stats.ttest_1samp(slopes_participants, 0, axis=0)
print(t_test)


# Create data
N = len(all_probs)
x = np.zeros(len(all_probs))
y = slopes_participants

# Plot
plt.scatter(x, y, alpha=0.5)
#plt.title('Slopes')
plt.title("Slopes: Semantic Meaningfulness of Words to Other List Words & P-Rec (By Word)")
#plt.xlabel('x')
#plt.ylabel('y')
#plt.ylim(-.01, .01)
plt.xticks([])
plt.show()
