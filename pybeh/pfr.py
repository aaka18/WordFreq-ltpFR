from __future__ import division
import numpy as np

def pfr(recalls, subjects, listLength):
    """"
    PFR   Probability of first recall.

    Computes probability of recall by serial position for the
    first output position.
    [p_recalls] = pfr(recalls_matrix, subjects, list_length)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls_matrix
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls_matrix(find(subjects==S), :)

        listLength: a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.

    OUTPUTS:
        p_recalls:  a matrix of probablities.  Its columns are indexed by
                    serial position and its rows are indexed by subject.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    subject = np.unique(subjects)
    result = [[0] * listLength for count in range(len(subject))]
    for subject_index in range(len(subject)):
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                if recalls[subj][0] > 0 and recalls[subj][0] < 1 + listLength:
                    result[subject_index][recalls[subj][0] - 1] += 1
                count += 1
        for index in range(listLength):
            result[subject_index][index] /= count
    return result