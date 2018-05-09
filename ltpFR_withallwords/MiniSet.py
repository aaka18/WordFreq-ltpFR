test_freq = [1, 1, 1, 3, 4, 4, 5, 5, 5, 6]
# bins = 2
# approx no items per bin = 5
bin_zero = []
bin_one = []


bins = []

for i in range(1,3):
    edge = test_freq[int((i*len(test_freq)/2)-1)]
    bins.append(edge)

print(bins)

for i, freq in enumerate(test_freq):
    if freq <= bins[0]:
        bin_zero.append([i, freq])

print(bin_zero)

# go through each frequency
    # assign first 1/10 of the frequencies to bin one

    # check the edge of that list and see if the next item is the same as the edge

    # if so, append it to the current bin

    # keep doing so until items are different

    # after doing so for the whole list, check # of items in each bin

    # and adjust accordingly

################

# go through each frequency

# determine edges of the bin based on no_words/10


# adjust edges of the bin to include words that should be moved over
    #### issue: how do we adjust the edges of the bin to the new values?


# then iterate through words and assign to the correct bins