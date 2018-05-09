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

    for f in files_ltpFR2:
        # Read in data
        test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)

        # Skip if there is no data for a participant
        if isinstance(test_mat_file['data'], np.ndarray):
            #print('Skipping...')
            continue

        # Get the session matrix for the participant
        session_mat = test_mat_file['data'].session
        # print(len(session_mat))
        # print(np.bincount(session_mat))

        # Skip if the participant did not finish
        if len(session_mat) < 552:
            # print('Skipping because participant did not finish...')
            continue

        # Get the presented items matrix for the participant
        else:
            print(f)
            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
            # Get just the lists in sessions 1-23rd sessions

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

    # print("All means", all_means)
    #print("Length of all means", np.shape(all_means))
    # print("All sems", all_sems)
    #print("Length of all sems", np.shape(all_sems))
    return all_means, all_sems

def pres_item_similarities(files_ltpFR2, all_means):
    """Test function to try getting semantic similarities of each pres item"""
    word_id_corr = []
    #print(np.shape(word_id_corr))
    # to be able to track down which all means it is
    participant_number = 0
    # For all participants:
    for f in files_ltpFR2:

        # Grab pres_mat (552 lists x 24 words) for this participant

        test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)

        # Skip if there is no data for a participant
        if isinstance(test_mat_file['data'], np.ndarray):
            # print('Skipping...')
            continue

        # Get the session matrix for the participant
        session_mat = test_mat_file['data'].session
        # print(len(session_mat))
        # print(np.bincount(session_mat))

        # Skip if the participant did not finish
        if len(session_mat) < 576:
            #print('Skipping because participant did not finish...')
            continue

        # Get the presented items matrix for the participant
        else:
            participant_number += 1
            print("Participant:", f)
            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
            pres_mat = pres_mat[np.where(session_mat != 24)]
            # print("Pres_mat:", pres_mat)
            # print("Length of pres_mat", np.shape(pres_mat))

            # Get their all_means matrix, which has each word's sim to other words in the list across 552 lists
            # dimensions of all_means are (552 lists x 24 words)

            all_means_part = all_means[participant_number - 1]
            # print(all_means_part)
            # print(np.shape(all_means_part))
            # print("Length should be 552 to 24:", np.shape(all_means_part))

            # Use boolean indexing (or another way of saying it, use pres_mat as a map, to find which word each
            # similarity in all_means is matched up to
            # Use code: all_means_part[np.where((pres_mat) == word_id)

            # once we match them, we grab out the numbers (the means) from all_means that correspond to each word
            # in the list of words that were presented to thsi participant (whcch happens to be the same for all pt's b/c ltpFR2)
            # dimensions are:

            # double check how many unique words but probably 576
            # (Number of words, 576 x how many times it has been presented)

            # Take the average of each row



            all_means_part = np.array(all_means_part)
            word_id_corr_this_part = []

            for word_id in np.unique(pres_mat):
                # print("Word ID:", word_id)
                # pres_mat = np.array(pres_mat)
                # print(np.shape(pres_mat))
                # print(pres_mat.dtype)
                # print(np.where((pres_mat)==word_id))
                word_id_corr_this_part.append(all_means_part[np.where((pres_mat) == word_id)])
        #print("Shape Word id this part", np.shape(word_id_corr_this_part))
        word_id_corr.append(word_id_corr_this_part)
    # print("Shape of word_id_corr", np.shape(word_id_corr))
    return word_id_corr

    # CURRENTLY EVERY PARTICIPANT IS HAVING THEIR OWN MATRIX 552 * 23, BUT HOW TO COMBINE ROWS IN EACH OF THESE MATRICES?


if __name__ == "__main__":
    # Get file path
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    all_means, all_sems = all_parts_list_correlations(files_ltpFR2)
    print("Legnth of means:", np.shape(all_means))
    print("Legnth of sems:", np.shape(all_sems))
    word_id_corr = pres_item_similarities(files_ltpFR2, all_means)
    #print("Shape of word_id_corr", len(word_id_corr))
    print(word_id_corr[493])

    # each_word_average_sim = []
    # for i in range(4):
    #     corr = word_id_corr[i]
    #     print(i, np.shape(corr))
    #     for row in corr:
    #         each_word_average_sim.append(np.mean(row))
    #
    #
    #





