import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats

from ltpFR2.Get_Bins import get_bins
from ltpFR2.WordFreq_Excel import M
from ltpFR_final.pools import get_bins_ltpFR

# REPLICATION OF LOHNAS PAPER WITH EXACT SAME PARTICIPANTS + UPDATED PARTICIPANTS

files_ltp = ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP063.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP064.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP065.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP066.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP067.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP069.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP070.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP073.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP074.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP075.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP076.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP077.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP079.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP081.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP082.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP084.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP085.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP086.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP087.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP088.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP089.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP090.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP091.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP092.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP093.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP094.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP095.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP096.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP098.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP099.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP100.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP101.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP102.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP103.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP104.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP105.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP106.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP107.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP108.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP110.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP111.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP112.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP113.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP114.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP115.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP117.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP118.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP119.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP120.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP122.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP123.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP124.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP125.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP127.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP128.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP130.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP131.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP132.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP133.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP134.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP135.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP136.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP137.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP138.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP139.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP140.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP141.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP142.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP143.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP144.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP145.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP146.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP147.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP148.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP149.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP150.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP151.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP153.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP155.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP159.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP166.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP168.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP174.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP184.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP185.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP186.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP187.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP188.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP190.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP191.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP192.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP193.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP194.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP195.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP196.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP197.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP198.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP199.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP200.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP201.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP202.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP207.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP209.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP210.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP211.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP212.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP215.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP227.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP228.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP229.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP230.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP231.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP232.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP233.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP234.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP235.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP236.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP237.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP238.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP239.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP240.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP241.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP242.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP243.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP244.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP245.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP246.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP247.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP249.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP250.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP251.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP252.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP253.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP254.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP256.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP258.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP259.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP260.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP261.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP263.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP264.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP265.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP267.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP268.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP269.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP270.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP271.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP272.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP273.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP274.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP275.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP276.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP277.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP278.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP279.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP280.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP281.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP282.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP283.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP284.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP285.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP286.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP287.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP288.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP289.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP290.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP291.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP292.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP293.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP294.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP331.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP332.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP333.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP334.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP335.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP336.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP337.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP338.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP339.mat']


files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')

# function to calculate probability of recall in each bin of ltpFR
def get_ltpFR(files_ltp):

    # counting the number of times a word is presented in each bin
    def count_presentations(a, pres_task):
        counts_with_task = []
        counts_no_task = []

        for k in range(0, 10):
            counts_with_task.append(len(np.where((a == k) & (pres_task > -1))[0]))
            counts_no_task.append(len(np.where((a == k) & (pres_task == -1))[0]))
        return np.array([counts_with_task, counts_no_task])

    # counting the number of times a word is recalled in each bin
    def count_recalls(recalled, a, pres_task):
        counts_with_task = []
        counts_no_task = []

        for k in range(0, 10):
            counts_with_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task > -1))[0]))
            counts_no_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task == -1))[0]))
        return np.array([counts_with_task, counts_no_task])

    # Get the recall probabilities of each bin
    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities


    word_dict = get_bins_ltpFR()

    all_prob = np.zeros((2,10))

    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"
    n = 0

    participant_probs_ltpFR = []

    # Do this for all participants
    for f in files_ltp:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


    # Get the array of items that were presented
    # Each row = 1 list of items

        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
        pres_task = test_mat_file['data'].pres.task.astype('int16')

        #print(test_mat_file['data'].pres_itemnos)

    # Testing out indexing of the presented item numbers file

    # print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    # Get the array of items that were recalled from each list presented
        rec_mat = test_mat_file['data'].pres.recalled
    #print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

        pres_bins = np.copy(pres_mat)
        rec_bins = np.copy(rec_mat)

    # get the bins of the word frequencies of the presented items

        for i in range(pres_bins.shape[0]):
            for j in range(pres_bins.shape[1]):
                if pres_bins[i][j] not in word_dict:
                    pres_bins[i][j] = -1
    # get the bins for the recalled items as well
        rec_bins = pres_bins

    #assign the new bin to the words (instead of the word freq)
        for id, bin in word_dict.items():
            pres_bins[pres_mat==id] = bin

    #print()
    #print(pres_bins)
    #print()
    #print(rec_bins)

        pres_counts = count_presentations(pres_bins, pres_task)
        rec_counts = count_recalls(rec_mat, rec_bins, pres_task)
        probi = (prob(pres_counts, rec_counts))
        participant_probs_ltpFR.append(probi)
        all_prob += probi
        n += 1


    print(" N of ltpFR IS:", n)
    all_prob /= n
    participant_probs_ltpFR = np.array(participant_probs_ltpFR)
    sem_ltpFR = scipy.stats.sem(participant_probs_ltpFR, axis=0)
    sd_ltpFR = np.std(participant_probs_ltpFR, axis=0)
    return sem_ltpFR, all_prob, sd_ltpFR


# function to calculate probability of recall in each bin of ltpFR2

def get_ltpFR2(files_ltpFR2):
    def count_presentations(a):
        counts_pres = []
        for k in range(0, 10):
            counts_pres.append(len(np.where((a == k))[0]))
        return np.array([counts_pres])

    def count_recalls(recalled, a):
        counts_rec = []
        for k in range(0, 10):
            counts_rec.append(len(np.where((a == k) & (recalled == 1))[0]))
        return np.array([counts_rec])

    # How to get items recalled once in each list & not get the repetitions or intrusions within session so the shapes would match?

    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities

    if __name__ == "__main__":
        word_dict = get_bins()

        all_prob = np.zeros((1,10))

        #files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
        #test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
        #word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"

        n = 0
        participant_probs_ltpFR2 = []
        for f in files_ltpFR2:
            print(f)
            test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
            if isinstance(test_mat_file['data'], np.ndarray):
                print('Skipping...')
                continue
          #  word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


        # Get the array of items that were presented
        # Each row = 1 list of items
            session_mat = test_mat_file['data'].session
            if len(session_mat) != 576:  # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
                print('Skipping because participant did not finish...')
                continue

            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')


            #print(test_mat_file['data'].pres_itemnos)

        # Testing out indexing of the presented item numbers file
        # print()
        # print("Testing indexing:")
        # print(pres_mat[0])
        # print()

        #    print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

        # Get the array of items that were recalled from each list presented
            rec_mat = test_mat_file['data'].pres.recalled
         #   print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

            pres_bins = np.copy(pres_mat)
            rec_bins = np.copy(rec_mat)

            for i in range(pres_bins.shape[0]):
                for j in range(pres_bins.shape[1]):
                    if pres_bins[i][j] not in word_dict:
                        pres_bins[i][j] = -1

            rec_bins = pres_bins

            for id, bin in word_dict.items():
                pres_bins[pres_mat==id] = bin

        #print()
        #print(pres_bins)
        #print()
        #print(rec_bins)

            pres_counts = count_presentations(pres_bins)
            rec_counts = count_recalls(rec_mat, rec_bins)
            probi = (prob(pres_counts, rec_counts))
            participant_probs_ltpFR2.append(probi)

            all_prob += probi
            n += 1

        print(" N of ltpFR2 IS:", n)
        all_prob /= n

        # matrix of individual participant probabilities across bins
        participant_probs_ltpFR2 = np.array(participant_probs_ltpFR2)

        # standard error of the mean, for each bin, across participants
        sem_ltpFR_2 = scipy.stats.sem(participant_probs_ltpFR2, axis=0)
        sd_ltpFR_2 = np.std(participant_probs_ltpFR2, axis = 0)
        return sem_ltpFR_2, all_prob, sd_ltpFR_2




# sem, mean probability rec within each bin
sem_ltpFR, all_prob_ltpFR, sd_ltpFR = get_ltpFR(files_ltp)
print(all_prob_ltpFR)
print(sem_ltpFR)
print(sd_ltpFR)

# sem, mean probability
sem_ltpFR2, all_prob_ltpFR2, sd_ltpFR_2 = get_ltpFR2(files_ltpFR2)
print(all_prob_ltpFR2)
print(sd_ltpFR_2)
print(sem_ltpFR2)

###


plt.figure(1)
plt.suptitle('Word Frequency', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=18)
f_no_task_ltpFR = plt.errorbar(M, all_prob_ltpFR[1], yerr = sem_ltpFR[1],  label='ltpFR No Task')
f_task_ltpFR = plt.errorbar(M, all_prob_ltpFR[0], yerr = sem_ltpFR[0],  label='ltpFR Task')
f_all_ltpFR2 = plt.errorbar(M, all_prob_ltpFR2[0], yerr = sem_ltpFR2[0], label='ltpFR2')
plt.xscale('log')
plt.legend(loc='upper right', prop={'size':12 }).draggable()
plt.show()
