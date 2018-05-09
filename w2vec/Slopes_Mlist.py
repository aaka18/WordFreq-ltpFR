import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from w2vec.Final import P_rec_ltpFR2_by_list as prbl, semantic_similarity_bylist as sem_list

nan = np.nan

# List of lists with all participants; 552 p_recs of lists per participant

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
all_probs_imported = prbl.get_ltpFR2_precall(files_ltpFR2)
all_means, all_sems, all_parts_each_list_ave_sim = sem_list.all_parts_list_correlations(files_ltpFR2)
all_sem_list = all_parts_each_list_ave_sim

all_probs = np.array(all_probs_imported)
all_sem_list = np.array(all_sem_list)

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
plt.title("Slopes: Semantic Meaningfulness of Words to Other List Words & P-Rec")
#plt.xlabel('x')
#plt.ylabel('y')
#plt.ylim(-.01, .01)
plt.xticks([])
plt.show()

"""Output
75
[-1.2810837722802706, 0.6382747653700257, 0.10751142057047927, -0.18945827248165789, 0.93166110542240888, 0.62850692355722171, 0.29489526077134365, 0.26532458475407078, 0.0020308504420848651, 0.087043285950096541, 0.1132081951116637, 1.0825258866188898, -0.024307199535131302, 0.80222459067887841, 0.79853131188089488, 0.79309583033328901, 0.67099656349652903, 0.33359785635090344, 0.71718206224538672, -0.44379474037912153, 0.22237838709907387, 0.27875483947049284, 0.47269134101958932, 0.88380756283970197, -0.12270329795485647, -0.29941067621066647, 0.90136208360360803, 1.2394035514100261, 0.61258580325857237, 0.58636706684085405, 0.8366041654984494, 0.33360955599403963, -0.24122299111441331, 0.46229951470969449, 0.72991676220648405, 0.18344679791034493, 0.27975819789738537, -0.20195957781177687, 0.32737141011685383, -0.18737839247592827, 0.49526055943280023, 0.33587851213768732, -0.080400303981225607, 0.49577816144168818, 0.33510290868485049, 2.313179427602861, 0.45366561794450982, 1.0807755300317328, -0.79817781688639311, -0.11974390190049349, 0.084378858945514904, 0.43825789071799914, 0.64255606928777753, 0.97369759572052084, 0.78674763782951573, 0.51604838908648742, 0.38973406921874582, 0.85575107271067596, 1.0874778932841382, 0.1787859479451967, -0.33164973609558979, 0.19263200726100779, 0.34963103416365465, 0.14000418327000416, 0.36469488804771749, 0.23402007603630254, 0.41557188761038799, 0.16361613719520274, 1.4321526289870361, 0.31483081516365696, -0.4782430035609147, 0.7646885577114706, 0.24229366824745643, 0.18099825918521625, 0.28418533844427846]
Ttest_1sampResult(statistic=6.3246138759487343, pvalue=1.7348319589136614e-08)

"""