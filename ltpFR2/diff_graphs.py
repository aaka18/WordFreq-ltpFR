
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statistics
import math
#from ltpFR2.ltpFR_ltpFR2 import sem_ltpFR, sem_ltpFR2

# FINDING TOTAL PROBABILITY OF RECALL FOR LTPFR2

# files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
#
# # a is the file for a single participant
# def count_presentations(a):
#      counts_pres = []
#
# #  HOW TO GET THE PRESENTED ITEMS IF THERE'S NO TASK
#      return np.array([len(np.where((pres_mat < 0))[0])])
# #
# #
# # counting the number of occurences a word is recalled
# def count_recalls(recalled):
#      counts_rec = []
# #
#      return np.array([len(np.where((recalled == 1))[0])])
#
#
# # How to get items recalled once in each list & not get the repetitions or intrusions within session so the shapes would match?
#
# #  Get the recall probabilities of each bin
# def prob(pres_counts, rec_counts):
#     probabilities = rec_counts / pres_counts
#     return probabilities
#
# pres_counts = np.zeros(1)
# rec_counts = np.zeros(1)
#
# all_probs = []
#
# # Do this for every participant
# for f in files_ltpFR2:
#     print(f)
#     # Read in data
#     test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
#     if isinstance(test_mat_file['data'], np.ndarray):
#         print('Skipping...')
#         continue
#     else:
#         pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
#         rec_mat = test_mat_file['data'].pres.recalled
#         #print(pres_mat)
#         #print(rec_mat)
#
#     #print(rec_mat.shape)
#
#     # For each list of this participant's data
#     this_participant_probs = []
#     n =0
#     for idx, row in enumerate(rec_mat):
#         # sum up the 1s in the recall matrix because that is everything they've remembered
#         num_recalled = sum(row)
#         # length of the row of presented items
#         num_presented = len(pres_mat[idx])
#
#         #finding each lists' probability of recall
#         list_prob = num_recalled/num_presented
#
#         # Add these list probs to this participant's probabilities list
#         this_participant_probs.append(list_prob)
#         n += 1
#
# #    get their mean for the participants overall p_rec
#     #print(this_participant_probs)
#     #print(type(this_participant_probs))
#
#
#     mean_pt_probs = np.nanmean(this_participant_probs)
#
#     # Append this partic p_rec to everyone else
#     all_probs.append(mean_pt_probs)
#
# #Get the average of all participants
# p_rec_all = np.nanmean(all_probs)
# sd_prec = np.std(all_probs)
# print(sd_prec)
# print(p_rec_all)


# CREATING THE DIFFERENCE IN EACH BINS AVERAGE & ALL PREC GRAPHS

# Overall average prec subtracted by each bin's prec in ltpFR2
array_ltpFR2 = ([-0.00258963, -0.02253316, -0.01147334, -0.02066432, -0.00653289,
        0.00331013, -0.02256423, -0.00056285,  0.01049548,  0.02532051])

all_prob_ltpFR2 = [ 0.47376608,  0.45382255,  0.46488237,  0.45569139,  0.46982281,  0.47966583,
   0.45379148,  0.47579285 , 0.48685119,  0.50167621]
# sd_ltpFR_2 = [ 0.17807775,  0.17476771,  0.17762841,  0.18496205,  0.18492326,  0.18517886,
#   0.1832743,   0.1872125,   0.18588065,  0.18156173]

# overall_sd_ltpFR_2 = (0.17901285666 * 0.17901285666)

# final_sd_ltpFR2 = []
# for i in sd_ltpFR_2:
#     final_sd_ltpFR2.append(math.sqrt(((i*i) + overall_sd_ltpFR_2)))

differences_ltpFR2 = []
for i in all_prob_ltpFR2:
    differences_ltpFR2.append(i - 0.476355704148 )
bins = range(1,11)


fig = plt.figure(1)
y_pos = np.arange(len(array_ltpFR2))
plt.bar(y_pos, array_ltpFR2, align='center')
plt.suptitle('Avg Bin P-Rec - Overall ltpFR2 P-Rec', fontsize=18)
plt.xlabel('Bin Number', fontsize=16)
plt.ylabel('Difference in Recall Probability', fontsize=16)
ind = range(0, 11)    # the x locations for the groups
plt.xticks(ind, bins)
plt.ylim((-.05, .05))
plt.autoscale(enable=True, axis='x', tight=None)



# P_rec of each bin in ltpFR task
all_prob_ltpFR_task =  [ 0.61052376,  0.58914622,  0.59610668 , 0.58424441,  0.57938536,  0.58957954,
   0.58460922,  0.59617755,  0.60191177,  0.64560948]

# sd_ltpFR_task = [ 0.13963395,  0.14107845,  0.14119727,  0.14320495, 0.14819052,  0.14838155,
#    0.14906467,  0.14634006,  0.14544815,  0.13717661]
# overall_sd_ltpFR_task = 0.59690371 * 0.59690371

# Subtracting overall prec from each bin's average prec
differences_ltpFR_task = []
for i in all_prob_ltpFR_task:
    differences_ltpFR_task.append(i - 0.59690371 )

# final_sd_ltpFR_task = []
# for i in sd_ltpFR_task:
#     final_sd_ltpFR_task.append(math.sqrt(((i*i) + overall_sd_ltpFR_task)))

plt.figure(2)
z_pos = np.arange(len(differences_ltpFR_task))
plt.ylim((-0.05, 0.05))
plt.bar(z_pos, differences_ltpFR_task,  align='center')
plt.suptitle('Avg Bin P-Rec - Overall ltpFR Task P-Rec', fontsize=18)
plt.xlabel('Bin Number', fontsize=16)
plt.ylabel('Difference in Recall Probability', fontsize=16)
ind = range(0, 11)    # the x locations for the groups
plt.xticks(ind, bins)
plt.autoscale(enable=True, axis='x', tight=None)
# P_rec of each bin in ltpFR no task

all_prob_ltpFR_notask = [ 0.68886441,  0.67348541,  0.67608663,  0.67417885,  0.67563323,  0.67135942,
   0.68277894,  0.68554571,  0.69580249,  0.7057427]
# std_ltpFR_no_task = [ 0.14494571,  0.15590852,  0.15253789,  0.16392974,  0.17409283,  0.1592174,
#    0.16627234,  0.15630211,  0.15741896,  0.15813334]
#
# overall_sd_ltpFR_notask = 0.68095845 * 0.68095845
#
# final_sd_ltpFR_notask = []
# for i in std_ltpFR_no_task:
#     final_sd_ltpFR_notask.append(math.sqrt(((i*i) + overall_sd_ltpFR_notask)))

# Subtracting overall prec from each bin's average prec

differences_ltpFR_notask = []
for i in all_prob_ltpFR_notask:
    differences_ltpFR_notask.append(i -  0.68095845)

plt.figure(3)
a_pos = np.arange(len(differences_ltpFR_notask))
plt.ylim((-.05, .05))
plt.bar(a_pos, differences_ltpFR_notask,  align='center')
plt.suptitle('Avg Bin P-Rec - Overall ltpFR No Task P-Rec', fontsize=18)
plt.xlabel('Bin Number', fontsize=16)
plt.ylabel('Difference in Recall Probability', fontsize=16)
ind = range(0, 11)    # the x locations for the groups
plt.xticks(ind, bins)
plt.autoscale(enable=True, axis='x', tight=None)
plt.show()
