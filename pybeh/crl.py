from __future__ import division
import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d


def crl(recalls=None, times=None, subjects=None, listLength=None, lag_num=None):
    """
    CRL  Inter-response time as a function of lag.

    Calculates the mean time it takes to move from one word position to another
    as a funciton of lag.  Returns lag-conditional response latency times.

    Please note: if i and j are consecutive words only real transitions are counted meaning
    1) neither i nor j are intrusions ( not == -1)
    2) words that have been recalled cannot be
      transitioned to nor cannot be transitioned from

    FUNCTION:
        crl = crl(recalls, times, subjects, listLength, lag_num);

    INPUT ARGS:
        recalls    - recall positions
        times      - time associated with each recall
        subjects   - subject number associated with each trial
        listLength - number of words in the list
        lag_num    - lag number to output


    OUTPUT ARGS:
        crl - a matrix of average crl times by lag position for each subject
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    if times is None:
        raise Exception('You must pass a times vector.')
    if subjects is None:
        raise Exception('You must pass a subject vector.')
    if listLength is None:
        raise Exception('You must pass a list length.')
    if lag_num is None:
        lag_num = listLength - 1
    if not (lag_num > 0):
        raise ValueError('lag number needs to be positive')
    if lag_num >= listLength:
        raise ValueError('Lag number too big')

    # Convert recalls and subjects to numpy arrays
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    trans_count = np.empty(num_lags)
    time_count = np.empty(num_lags)

    # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        trans_count.fill(0)
        time_count.fill(0)
        cur_recs = recalls[subjects == subj]
        cur_times = times[subjects == subj]
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(cur_recs))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(cur_recs):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition and timing counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k + 1]:
                    next_rec = trial_recs[k + 1]
                    trans = next_rec - rec
                    trans_time = cur_times[j, k+1] - cur_times[j, k]
                    # Record the transition that was made and its IRT
                    trans_count[trans + listLength - 1] += 1
                    time_count[trans + listLength - 1] += trans_time

        result[i, :] = time_count / trans_count
        result[i, trans_count == 0] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]
