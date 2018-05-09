# Do for each participant

    #  Get a matrix that contains the sem sim values in each list, for each word to all the other words in the list
    # Average across each row and get 552 lists * 1 averaged value
    # Find the p_rec for each list
    # Correlate the two vectors (mean similarities vector to the p_rec vector)

## THIS IS THE FINAL CODE TO FIND SEMANTIC W2V MEANINGFULNESS MEASURE FOR EACH LIST OF EACH PARTICIPANT IN LTPFR2 TO OTHER WORDS IN ITS LIST


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
import json

w2v_path = "/Users/adaaka/Desktop/w2v.mat"
w2v = scipy.io.loadmat(w2v_path, squeeze_me=False, struct_as_record=False)['w2v']
w2v = np.array(w2v)
# print("Word to vector matrix", w2v)

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


def all_parts_list_correlations(files_ltpFR2):

    """Adding each participants' w2v similarity of presented items in each list to other words in the
    list into a gigantic, multidimentional matrix (participant # * 552 lists * 24 items in each list  """

    all_means = []
    all_sems = []
    all_parts_each_list_ave_sim = []

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
    #     if len(session_mat) < 576:
    #         # print('Skipping because participant did not finish...')
    #         continue
    #
    #     # Get the presented items matrix for the participant
    #     else:
    #         print(f)
    #         pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    #         # Get just the lists in sessions 1-23rd sessions
    for f in files_ltpFR2:
        if f not in ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP331.json']:
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
            session_mat = np.array(data['session'])
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

            all_means_participant, all_sems_participant = one_participants_list_means(w2v,pres_mat)


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

            all_means_participant = np.array(all_means_participant)
            #print("All means participant shape", np.shape(all_means_participant))

            this_part_each_list_ave_sim = np.mean(all_means_participant, axis= 1)
            # print("This part each list ave sim", this_part_each_list_ave_sim)
            # print("Length of this part each list ave sim", len(this_part_each_list_ave_sim))

            all_parts_each_list_ave_sim.append(this_part_each_list_ave_sim)

    return all_means, all_sems, all_parts_each_list_ave_sim
    # print("All means", all_means)
    #print("Length of all means", np.shape(all_means))
    # print("All sems", all_sems)
    #print("Length of all sems", np.shape(all_sems))


if __name__ == "__main__":
    # Get file path
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
    all_means, all_sems, all_parts_each_list_ave_sim = all_parts_list_correlations(files_ltpFR2)
    print("Shape of all_means:", np.shape(all_means))
    print("Shape of all_sems:", np.shape(all_sems))
    print("All parts each list ave sim", (all_parts_each_list_ave_sim))
    print("Shape of all parts each list ave sim", np.shape(all_parts_each_list_ave_sim))





