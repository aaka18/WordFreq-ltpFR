import glob


def main():
    subj_data_dir = '/Volumes/rhino/data5/eeg/scalp/ltp/ltpFR/behavioral/data/'
    files = glob.glob('/Volumes/rhino/data5/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP[0-9][0-9][0-9].mat')
    #files = glob.glob('/Volumes/rhino/data5/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP062.mat')

    print(files)

if __name__ == "__main__":
    main()

