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

""" This is the final code to find one W2V meaningfulness measure for each word in ltpFR2 to other words in its list"""


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
    # print(np.shape(np.array(mini_mat)))
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

    #print(np.shape(np.array(means_of_lists)))
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


word_id_all_participants = []

def all_parts_list_correlations(files_ltpFR2):

    """Adding each participants' w2v similarity of presented items in each list to other words in the
    list into a gigantic, multidimentional matrix (participant # * 552 lists * 24 items in each list  """

    all_means = []
    all_sems = []
    word_id_corr = []

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
        if len(session_mat) < 576:
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

        word_id_corr_this_part = []


        for word_id in np.unique(pres_mat):
            word_id_corr_this_part.append(all_means_participant[np.where((pres_mat) == word_id)])
        # print(word_id_corr_this_part)
        # print("Shape Word id this part", np.shape(word_id_corr_this_part))

        word_id_all_participants.append(word_id_corr_this_part)
        # print("word id all", word_id_all_participants)
        # print("Shape of word id all participants", np.shape(word_id_corr_this_part))
        each_word_aver_sim_part = np.mean(word_id_corr_this_part, axis = 1)
        # print(each_word_aver_sim_part)
        #print("Each word aver sim part shape:", np.shape(np.array((each_word_aver_sim_part))))
        #
        word_id_corr.append(each_word_aver_sim_part)



    average = np.mean(word_id_corr, axis= 0)
    # print("Average of each word", average)
    # print("Length of the list", len(average))

    return average
    # print("All means", all_means)
    #print("Length of all means", np.shape(all_means))
    # print("All sems", all_sems)
    #print("Length of all sems", np.shape(all_sems))

    # CURRENTLY EVERY PARTICIPANT IS HAVING THEIR OWN MATRIX 576 * 23, BUT HOW TO COMBINE ROWS IN EACH OF THESE MATRICES?


if __name__ == "__main__":
    # Get file path
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    #all_means, all_sems, word_id_corr = all_parts_list_correlations(files_ltpFR2)
    average = all_parts_list_correlations(files_ltpFR2)
    # print("Legnth of means:", np.shape(all_means))
    # print("Legnth of sems:", np.shape(all_sems))
    print("Average length", np.shape(average))
    print(average)

    for i in average:
        print(i)
    # word_id = [5, 6, 10, 12, 13, 24, 29, 33, 35, 38, 39, 46, 53, 55, 61, 63, 64, 66, 67, 68, 75, 82, 83, 84, 86, 88, 91,
    #            94, 100, 104, 108, 109, 113, 119, 125, 127, 131, 133, 134, 136, 137, 139, 142, 143, 145, 157, 159, 162,
    #            164, 166, 168, 169, 171, 172, 175, 177, 179, 183, 184, 187, 188, 194, 198, 199, 200, 203, 210, 211, 216,
    #            223, 226, 230, 235, 240, 248, 249, 253, 254, 255, 258, 261, 265, 267, 278, 280, 283, 292, 295, 299, 304,
    #            307, 308, 313, 315, 316, 323, 325, 327, 328, 336, 338, 339, 340, 341, 343, 346, 348, 353, 358, 360, 368,
    #            371, 372, 373, 378, 383, 384, 386, 388, 389, 390, 392, 395, 397, 400, 401, 403, 405, 407, 408, 410, 413,
    #            415, 417, 425, 427, 429, 431, 435, 437, 439, 441, 442, 446, 447, 466, 467, 476, 483, 484, 485, 488, 490,
    #            492, 494, 495, 496, 497, 498, 499, 501, 508, 509, 512, 519, 520, 530, 541, 543, 546, 553, 559, 561, 565,
    #            567, 569, 570, 573, 575, 576, 578, 579, 580, 584, 585, 590, 591, 592, 593, 598, 599, 600, 601, 602, 604,
    #            605, 606, 610, 614, 618, 620, 622, 623, 624, 627, 628, 633, 635, 637, 639, 641, 642, 645, 654, 658, 659,
    #            660, 661, 662, 666, 667, 670, 675, 682, 683, 684, 689, 691, 692, 696, 697, 699, 702, 706, 707, 708, 713,
    #            714, 715, 718, 719, 723, 724, 726, 728, 731, 732, 733, 734, 737, 738, 745, 746, 749, 752, 754, 755, 760,
    #            761, 763, 764, 766, 769, 771, 776, 778, 782, 783, 785, 790, 792, 793, 794, 795, 796, 798, 800, 801, 804,
    #            807, 809, 812, 814, 820, 821, 827, 829, 833, 834, 839, 840, 843, 846, 847, 848, 849, 852, 857, 858, 861,
    #            866, 869, 870, 871, 872, 873, 874, 876, 878, 879, 883, 885, 888, 893, 895, 900, 901, 908, 912, 916, 919,
    #            922, 929, 932, 935, 937, 940, 947, 953, 955, 959, 960, 963, 972, 974, 976, 977, 980, 982, 987, 988, 989,
    #            994, 995, 996, 997, 999, 1000, 1002, 1006, 1009, 1011, 1013, 1015, 1016, 1017, 1019, 1021, 1022, 1024,
    #            1027, 1031, 1032, 1035, 1036, 1037, 1044, 1047, 1048, 1049, 1051, 1057, 1060, 1061, 1065, 1066, 1070,
    #            1071, 1075, 1076, 1079, 1080, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1095, 1096, 1098, 1104, 1108,
    #            1111, 1112, 1113, 1117, 1119, 1123, 1124, 1125, 1132, 1139, 1140, 1143, 1144, 1147, 1148, 1149, 1152,
    #            1155, 1158, 1159, 1170, 1171, 1172, 1173, 1176, 1183, 1184, 1192, 1194, 1197, 1199, 1200, 1202, 1203,
    #            1209, 1215, 1219, 1221, 1224, 1227, 1229, 1232, 1238, 1242, 1243, 1245, 1246, 1250, 1251, 1256, 1258,
    #            1260, 1261, 1262, 1271, 1272, 1275, 1280, 1282, 1284, 1287, 1292, 1293, 1294, 1295, 1299, 1301, 1303,
    #            1307, 1312, 1315, 1317, 1319, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1341, 1343, 1345, 1347,
    #            1349, 1351, 1355, 1356, 1357, 1358, 1368, 1369, 1370, 1371, 1373, 1374, 1382, 1384, 1385, 1388, 1392,
    #            1395, 1400, 1403, 1405, 1407, 1408, 1411, 1414, 1415, 1416, 1420, 1422, 1423, 1426, 1427, 1436, 1438,
    #            1440, 1448, 1451, 1456, 1460, 1461, 1468, 1472, 1478, 1486, 1488, 1493, 1496, 1501, 1503, 1505, 1508,
    #            1511, 1512, 1515, 1519, 1520, 1521, 1522, 1524, 1526, 1527, 1530, 1532, 1533, 1535, 1536, 1544, 1545,
    #            1546, 1555, 1557, 1562, 1565, 1572, 1578, 1580, 1581, 1587, 1592, 1593, 1605, 1606, 1608, 1609, 1611,
    #            1618, 1620, 1621, 1622, 1625, 1626, 1628, 1629, 1631, 1633, 1635, 1636]
    #
    # top_ten = np.percentile(average, 90)
    # print("Top Ten", top_ten)
    #
    # # Finding the ltpFR2 word indices from the whole ltpFR word pool that are in the top 10 percentile
    # if_m_list_is_larger_90 = []
    # word_id_m_list_larger_90 = []
    # for index, word_meaningfulness in enumerate(average):
    #     # print(index)
    #     # print(word_meaningfulness)
    #     if word_meaningfulness >= top_ten:
    #         if_m_list_is_larger_90.append(1)
    #         word_id_m_list_larger_90.append(word_id[index])
    #
    #     else:
    #         if_m_list_is_larger_90.append(0)
    #
    # print("if m_list_is_larger", if_m_list_is_larger_90)
    # print(len(if_m_list_is_larger_90))
    # print("word if m list larger", word_id_m_list_larger_90)
    # print("len word id index larger", word_id_m_list_larger_90)



"""Output is:
Average length (576,)
[ 0.12654216  0.12452603  0.12209616  0.12802416  0.1249303   0.12578301
  0.12729648  0.13139068  0.13009769  0.12522033  0.12383469  0.12008217
  0.12239785  0.12336479  0.1199156   0.11891321  0.12867105  0.12139257
  0.12815503  0.12317369  0.1257358   0.12747482  0.12447597  0.12341179
  0.12540499  0.12626988  0.12457579  0.12636651  0.13174159  0.12590417
  0.12841048  0.13041011  0.12551723  0.12985977  0.12798512  0.12473746
  0.12743525  0.1250253   0.13031711  0.12736237  0.11967824  0.12794933
  0.12095697  0.12270354  0.12570069  0.1255815   0.12450363  0.12717228
  0.12438677  0.12177149  0.12194783  0.12856533  0.12597306  0.12538152
  0.12540379  0.12451674  0.12646462  0.12307893  0.12806046  0.12747547
  0.1250206   0.12788941  0.13006456  0.12162118  0.12667831  0.12740903
  0.13122176  0.12733422  0.12914658  0.12631768  0.12586773  0.12607789
  0.12517071  0.12032284  0.12710647  0.12792007  0.12659663  0.12503519
  0.1270522   0.12531926  0.12862906  0.12826814  0.1253342   0.12527044
  0.12659029  0.12541242  0.12192835  0.12858484  0.12871432  0.12914372
  0.12576327  0.12971191  0.12910203  0.12799898  0.12432605  0.12555244
  0.12240542  0.12692749  0.12222971  0.13171406  0.12551641  0.1246697
  0.1261811   0.12618309  0.12597054  0.12029687  0.12393103  0.12623553
  0.12572135  0.12271522  0.12084705  0.12310972  0.12881216  0.1290328
  0.12775244  0.12883284  0.1270959   0.12930962  0.12086329  0.12258362
  0.11987519  0.12705196  0.12595271  0.13125198  0.1249997   0.12898503
  0.13129103  0.12697463  0.13014334  0.12465323  0.12162762  0.12001673
  0.12385753  0.12594433  0.12780339  0.12434061  0.13004169  0.12084778
  0.12954885  0.12670184  0.12737932  0.12575061  0.12845035  0.12429133
  0.12597949  0.12471514  0.12827309  0.12493933  0.12811104  0.12830668
  0.1219727   0.12503628  0.12790799  0.12811979  0.11854612  0.12895779
  0.12869059  0.1233125   0.12838237  0.1228213   0.12019203  0.12693328
  0.12672539  0.12860215  0.12423496  0.12606879  0.12248456  0.12013415
  0.12090494  0.12404068  0.12702611  0.12599069  0.12332892  0.12525315
  0.12659278  0.12936787  0.1279276   0.12434265  0.12832777  0.12832434
  0.11903087  0.13034679  0.1276594   0.13201058  0.1276038   0.12434575
  0.12233818  0.13099694  0.12582038  0.12953246  0.12450302  0.12330207
  0.11850563  0.13021369  0.12886683  0.12583837  0.12700758  0.12925807
  0.12804707  0.12725599  0.12746103  0.12521667  0.12941204  0.13108502
  0.12405374  0.12679301  0.12499286  0.12517607  0.1297582   0.12858817
  0.11904697  0.12776128  0.12801951  0.12872841  0.12515922  0.12384326
  0.12629789  0.1291678   0.12587926  0.12436892  0.12369708  0.12796784
  0.12549509  0.12071269  0.1252496   0.1303736   0.126881    0.12439913
  0.12515401  0.1294582   0.12554884  0.12467491  0.1283463   0.12381864
  0.12217782  0.12533005  0.13065801  0.12818976  0.12526317  0.1287405
  0.12947795  0.12481756  0.12578077  0.12355852  0.12955656  0.12214422
  0.12405569  0.12492122  0.1270086   0.12963925  0.12422131  0.12494336
  0.12568182  0.12029666  0.12915866  0.13069456  0.13132913  0.12313166
  0.1240527   0.12687057  0.128       0.12348023  0.12489188  0.12504765
  0.12558567  0.13211399  0.12963967  0.12824873  0.13053955  0.12289426
  0.12720329  0.12323021  0.12868777  0.12843847  0.12655346  0.12815992
  0.12592084  0.12198231  0.12724704  0.12047125  0.12519992  0.1282012
  0.13001419  0.12390733  0.12762829  0.12624317  0.12782972  0.12804291
  0.13172431  0.12716787  0.12715159  0.11984099  0.1275181   0.12799919
  0.12470336  0.12366437  0.12550278  0.12527889  0.12722411  0.12836579
  0.12724553  0.12177255  0.12491209  0.11958894  0.12360862  0.12101728
  0.12515056  0.13034328  0.12175376  0.12876964  0.12880487  0.12252774
  0.12050614  0.12738973  0.12793857  0.12606357  0.12788889  0.12105031
  0.12949096  0.12366119  0.12723569  0.12483831  0.13397385  0.12983896
  0.12598025  0.12760505  0.12737161  0.12142145  0.13285513  0.12557372
  0.12307463  0.1210988   0.12635823  0.12326699  0.1274528   0.12993711
  0.13049799  0.12897717  0.12953138  0.12497815  0.12363274  0.12210199
  0.12708087  0.12971307  0.1222876   0.12147789  0.12489016  0.12731495
  0.12940105  0.12700577  0.12991438  0.12475453  0.12372804  0.12732915
  0.13160868  0.12033093  0.11772307  0.12927764  0.13107377  0.1227483
  0.1207266   0.13057385  0.12552263  0.1293979   0.12804535  0.12457096
  0.12949711  0.12676624  0.12612564  0.12673239  0.12225215  0.12712966
  0.12663428  0.12553033  0.12902279  0.12609163  0.12131309  0.13082469
  0.12766622  0.12502532  0.13174961  0.12804158  0.12707459  0.12697319
  0.1230011   0.12265961  0.12685395  0.11944838  0.1284179   0.12297729
  0.12809926  0.12215213  0.12982059  0.127797    0.12472034  0.13006016
  0.12131671  0.12697008  0.12440018  0.12673311  0.12630599  0.12776639
  0.1217553   0.13037482  0.13015288  0.12236539  0.13165695  0.13083816
  0.11801297  0.12745211  0.13293979  0.12482729  0.13161034  0.1226577
  0.12509759  0.12759475  0.12461896  0.12142551  0.1189102   0.1230551
  0.11891218  0.12737948  0.12597166  0.12743946  0.12875176  0.12861127
  0.1215468   0.12809727  0.13056339  0.12598028  0.12557098  0.12531174
  0.13153356  0.13018066  0.12673256  0.12943423  0.13133249  0.12837968
  0.1273246   0.12880482  0.13077738  0.12763032  0.12700629  0.12473891
  0.12903156  0.12690867  0.12858989  0.1228742   0.12452676  0.12556734
  0.12487243  0.12372429  0.1218493   0.12281779  0.12828241  0.13157002
  0.12512816  0.12466109  0.12644115  0.12946715  0.12579611  0.12520813
  0.12795104  0.12608737  0.12529472  0.12802828  0.12744353  0.12155538
  0.12667069  0.12629363  0.12666658  0.1236445   0.12716576  0.13100311
  0.13144449  0.12692861  0.12537495  0.12014889  0.12856606  0.12927578
  0.12919463  0.12923525  0.12699964  0.13255091  0.12720722  0.11936542
  0.12370597  0.12436042  0.12170054  0.12089817  0.12768835  0.12587822
  0.12724272  0.13101427  0.13058161  0.12215676  0.12406773  0.12402599
  0.12797622  0.12011488  0.124519    0.12546711  0.12735978  0.12027875
  0.12374605  0.12855337  0.12376568  0.12106191  0.13099126  0.12405507
  0.1242317   0.12822858  0.1239341   0.12484509  0.12669598  0.12639627
  0.12798173  0.12528201  0.12535389  0.12766207  0.13068328  0.12842454
  0.12824258  0.12516022  0.12879764  0.12390795  0.12453879  0.12642048
  0.12592308  0.12388233  0.12923485  0.12944326  0.12604513  0.12484039
  0.13198472  0.12907354  0.13224608  0.12994473  0.12695602  0.1294045
  0.12592259  0.12440061  0.12393322  0.12275847  0.12584471  0.12896903
  0.12332743  0.12536018  0.12985324  0.1231806   0.12582452  0.12646167
  0.12448393  0.12720203  0.12671204  0.12980651  0.1275493   0.12811179
  0.12984869  0.12991251  0.12598101  0.12862698  0.12496086  0.12420312
  0.12249999  0.13066563  0.12455465  0.12035803  0.12798124  0.12599916
  0.12539199  0.12216943  0.12525655  0.12840187  0.12722009  0.12787709]
if m_list_is_larger [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
576

Process finished with exit code 0

  When n = 76"""


