import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np

# THIS IS THE FINAL CODE TO FIND SEMANTIC W2V MEANINGFULNESS MEASURE FOR EACH PARTICIPANT IN LTPFR2 TO OTHER WORDS IN ITS LIST



import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np


def get_mini_matrix(big_matrix, list_indices):
    """Method to extract a mini matrix from a bigger matrix that
    contains the semantic similarity values for just the words
    in this individual list."""

    # initialize mini matrix
    mini_mat = []

    # for each row in bigger matrix,
    for row_idx, row in enumerate(big_matrix):

        # if this row index is in our list of indices,
        if row_idx in list_indices:
            # get out the items that are in the list,
            mini_row = row[list_indices]
            # append them to the mini matrix
            mini_mat.append(mini_row)

    # reshape into a matrix format & print
    return np.array(mini_mat)


def get_list_similarities(similarity_matrix):
    """This method takes out similarity of a word to itself out from each row,
    sums the other similarities in each row and averages them"""

    means_of_lists = []
    sems_of_lists = []
    for item_row in similarity_matrix:

        # take out words similarity to itself (i.e., 1)
        items_row = item_row[item_row < 1.0]

        # np.mean of the item's row
        mean_items_row = np.mean(items_row)

        # find the sem
        sem_row = scipy.stats.sem(items_row)

        # save it to means of lists
        means_of_lists.append(mean_items_row)
        sems_of_lists.append(sem_row)


    return means_of_lists, sems_of_lists

# w2v_path = "/Users/adaaka/Desktop/w2v.mat"
# w2v = scipy.io.loadmat(w2v_path, squeeze_me=False, struct_as_record=False)['w2v']
# w2v = np.array(w2v)
# print("Word to vector matrix", w2v)



def one_participants_list_means(w2v, presented_items):
    """This is the loop to find one participant's means and sems of lists"""
    this_parts_means_of_lists = []
    this_parts_sems_of_lists = []
    for single_list in presented_items:
        # Comment this out for testing, keep it for an actual participant matlab to python indexing
        single_list -= 1
        #print("Single list:", single_list)
        item_to_mini_matrix = get_mini_matrix(w2v, single_list)
        #print("Item to mini matrix:", item_to_mini_matrix)
        means_of_lists, sems_of_lists = get_list_similarities(item_to_mini_matrix)
        this_parts_means_of_lists.append(means_of_lists)
        this_parts_sems_of_lists.append(sems_of_lists)
        #print("This list's means of lists:", means_of_lists)
        #print("This list's means of sems:", sems_of_lists)
    # print("This part's means:")
    # print(np.array(this_parts_means_of_lists))
    # print("This part's sems")
    # print(np.array(this_parts_sems_of_lists))
    # print("This participant's all lists:")
    # print(np.array(all_lists))
    return this_parts_means_of_lists, this_parts_sems_of_lists




w2v_path = "/Users/adaaka/Desktop/w2v.mat"
w2v = scipy.io.loadmat(w2v_path, squeeze_me=False, struct_as_record=False)['w2v']
w2v = np.array(w2v)
word_to_pool = get_list_similarities(w2v)
# print("W2V Similarities of Each Word to the Other Words in the Pool + SEMS", word_to_pool )



def all_parts_list_correlations(files_ltpFR2):

    """Adding each participants' w2v similarity of presented items in each list to other words in the
    list into a gigantic, multidimentional matrix (participant # * 552 lists * 24 items in each list  """

    all_means = []
    all_sems = []
    word_id_average_all_sub = []

    # for f in files_ltpFR2:
    #     # Read in data
    #     test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
    #
    #     # Skip if there is no data for a participant
    #     if isinstance(test_mat_file['data'], np.ndarray):
    #         #print('Skipping...')
    #         continue
    #
    #     # Get the session matrix for the participant
    #     session_mat = test_mat_file['data'].session
    #     # print(len(session_mat))
    #     # print(np.bincount(session_mat))
    #
    #     # Skip if the participant did not finish
    #     if len(session_mat) < 552:
    #         # print('Skipping because participant did not finish...')
    #         continue
    #
    #     # Get the presented items matrix for the participant
    #     else:
    #         print(f)
    #         pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    #         # Get just the lists in sessions 1-23rd sessions
    #         # print(np.unique(pres_mat[np.where(session_mat != 24)]).shape)
    for f in files_ltpFR2:
        if f not in ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP325.json', '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP341.json', '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP250.json']:
        # Read in data
        #test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
            with open(f, 'r') as jsfile:
                data = json.load(jsfile)
            """
            if isinstance(test_mat_file['data'], np.ndarray):
                #print('Skipping...')
                continue
            session_mat = test_mat_file['data'].session
            """
            session_mat = np.array(data['session'])[data['good_trial']]
            # print(len(session_mat))
            # print(np.bincount(session_mat))
            #if len(session_mat) < 576:
            #    print('Skipping because participant did not finish...')
            #    continue
            #else:
            print(f)
            pres_mat = np.array(data['pres_nos'], dtype='int16')
            pres_mat = pres_mat[np.where(session_mat != 23)]
            rec_mat = np.array(data['recalled'])
            rec_mat = rec_mat[np.where(session_mat != 23)]

        # Get all means and all sems for this participant
        all_means_participant, all_sems_participant = one_participants_list_means(w2v,pres_mat[np.where(session_mat != 24)])


        # print("This part's means:")
        # print(np.array(all_means_participant))
        # print(np.shape(all_means_participant))
        # print("This part's sems")
        # print(np.array(all_sems_participant))
        # print(np.shape(all_sems_participant))
        # print("This participant's all presented lists:")
        # print(np.array(pres_mat))

        all_means.append(all_means_participant)
        all_sems.append(all_sems_participant)

        # Get their all_means matrix, which has each word's sim to other words in the list across 552 lists
        # dimensions of all_means are (552 lists x 24 words)

        # Use boolean indexing (or another way of saying it, use pres_mat as a map, to find which word each
        # similarity in all_means is matched up to
        # Use code: all_means_part[np.where((pres_mat) == word_id)

        # once we match them, we grab out the numbers (the means) from all_means that correspond to each word
        # in the list of words that were presented to thsi participant (whcch happens to be the same for all pt's b/c ltpFR2)
        # dimensions are: 576 * (23 * N) participants

        # double check how many unique words but probably 576
        # (Number of words, 576 x how many times it has been presented)

        # Take the average of each row
        pres_mat = pres_mat[np.where(session_mat != 24)]
        all_means_participant = np.array(all_means_participant)
        # print("All means participant", all_means_participant.shape)

        word_id_corr_this_part = []



        for word_id in np.unique(pres_mat):
        # print("Word ID:", word_id)
        # print(pres_mat.dtype)
        # print(np.where((pres_mat)==word_id))
            word_id_corr_this_part.append(all_means_participant[np.where((pres_mat) == word_id)])
        # print("Shape Word id this part", np.shape(word_id_corr_this_part))

        each_word_aver_sim_part = np.mean(word_id_corr_this_part, axis = 1)
        # print(each_word_aver_sim_part)
        # print("Each word aver sim part shape:", len(each_word_aver_sim_part))

        word_id_average_all_sub.append(each_word_aver_sim_part)

    # print("Word id corr length", np.shape(word_id_average_all_sub))

    return all_means, all_sems, word_id_average_all_sub
    # print("All means", all_means)
    #print("Length of all means", np.shape(all_means))
    # print("All sems", all_sems)
    #print("Length of all sems", np.shape(all_sems))



if __name__ == "__main__":
    # Get file path
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
    all_means, all_sems, word_id_average_all_sub = all_parts_list_correlations(files_ltpFR2)
    print("Legnth of means:", np.shape(all_means))
    print("Legnth of sems:", np.shape(all_sems))

    print("Shape of word_id_corr", np.shape(word_id_average_all_sub))

    all_subjects = []
    for p in range(len(files_ltpFR2)):
        pt_sheet_ave = np.mean(word_id_average_all_sub[p])
        print("Average of each participant", pt_sheet_ave)
        all_subjects.append(pt_sheet_ave)
    print(all_subjects)
    print(len(all_subjects))
    print(np.mean(np.array(all_subjects)))
        #print("Length of the list", len(pt_sheet_ave))



"""Below is the output, very similar for each participant:
Average of each participant 0.125564104799
Average of each participant 0.125715211198
Average of each participant 0.12567775043
Average of each participant 0.125702501519
Average of each participant 0.126187796904
Average of each participant 0.126583646734
Average of each participant 0.126284666353
Average of each participant 0.126407434112
Average of each participant 0.126592933249
Average of each participant 0.126055629376
Average of each participant 0.126120894443
Average of each participant 0.126045525617
Average of each participant 0.126231152749
Average of each participant 0.125963093475
Average of each participant 0.126201449331
Average of each participant 0.126169054426
Average of each participant 0.126021077448
Average of each participant 0.126347927279
Average of each participant 0.126274857517
Average of each participant 0.126233516561
Average of each participant 0.125973830095
Average of each participant 0.126168449164
Average of each participant 0.125979744267
Average of each participant 0.126708211007
Average of each participant 0.126200881846
Average of each participant 0.125978338226
Average of each participant 0.126070217344
Average of each participant 0.126394312263
Average of each participant 0.126382372926
Average of each participant 0.12600916501
Average of each participant 0.126257349823
Average of each participant 0.125645003766
Average of each participant 0.126334211697
Average of each participant 0.126447227721
Average of each participant 0.126147865448
Average of each participant 0.125725172276
Average of each participant 0.126035142497
Average of each participant 0.12649062634
Average of each participant 0.125924545099
Average of each participant 0.126010850393
Average of each participant 0.126128259477
Average of each participant 0.126235606913
Average of each participant 0.125878593933
Average of each participant 0.125863381964
Average of each participant 0.1259773236
Average of each participant 0.125887918896
Average of each participant 0.126175280584
Average of each participant 0.126388987155
Average of each participant 0.12584788089
Average of each participant 0.126480972348
Average of each participant 0.126366539622
Average of each participant 0.125976517164
Average of each participant 0.12630194717
Average of each participant 0.126071888989
Average of each participant 0.126782108277
Average of each participant 0.126020150485
Average of each participant 0.125972750107
Average of each participant 0.126430182632
Average of each participant 0.126033566988
Average of each participant 0.126190842379
Average of each participant 0.126198855612
Average of each participant 0.126090416917
Average of each participant 0.126548521429
Average of each participant 0.125977045188
Average of each participant 0.126186155027
"""
