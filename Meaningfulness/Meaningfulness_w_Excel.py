import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table,Column
import math
import numpy.ma as ma
from w2vec import semanticism_each_word_ave as m_list
import scipy.io
import glob

# Array below is the word frequencies of all items in the word pool in an ascending order

# files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
# m_list = m_list.all_parts_list_correlations(files_ltpFR2)

meaningfulness = [0.12654216, 0.12452603, 0.12209616, 0.12802416, 0.1249303, 0.12578301,
                      0.12729648, 0.13139068, 0.13009769, 0.12522033, 0.12383469, 0.12008217,
                      0.12239785, 0.12336479, 0.1199156, 0.11891321, 0.12867105, 0.12139257,
                      0.12815503, 0.12317369, 0.1257358, 0.12747482, 0.12447597, 0.12341179,
                      0.12540499, 0.12626988, 0.12457579, 0.12636651, 0.13174159, 0.12590417,
                      0.12841048, 0.13041011, 0.12551723, 0.12985977, 0.12798512, 0.12473746,
                      0.12743525, 0.1250253, 0.13031711, 0.12736237, 0.11967824, 0.12794933,
                      0.12095697, 0.12270354, 0.12570069, 0.1255815, 0.12450363, 0.12717228,
                      0.12438677, 0.12177149, 0.12194783, 0.12856533, 0.12597306, 0.12538152,
                      0.12540379, 0.12451674, 0.12646462, 0.12307893, 0.12806046, 0.12747547,
                      0.1250206, 0.12788941, 0.13006456, 0.12162118, 0.12667831, 0.12740903,
                      0.13122176, 0.12733422, 0.12914658, 0.12631768, 0.12586773, 0.12607789,
                      0.12517071, 0.12032284, 0.12710647, 0.12792007, 0.12659663, 0.12503519,
                      0.1270522, 0.12531926, 0.12862906, 0.12826814, 0.1253342, 0.12527044,
                      0.12659029, 0.12541242, 0.12192835, 0.12858484, 0.12871432, 0.12914372,
                      0.12576327, 0.12971191, 0.12910203, 0.12799898, 0.12432605, 0.12555244,
                      0.12240542, 0.12692749, 0.12222971, 0.13171406, 0.12551641, 0.1246697,
                      0.1261811, 0.12618309, 0.12597054, 0.12029687, 0.12393103, 0.12623553,
                      0.12572135, 0.12271522, 0.12084705, 0.12310972, 0.12881216, 0.1290328,
                      0.12775244, 0.12883284, 0.1270959, 0.12930962, 0.12086329, 0.12258362,
                      0.11987519, 0.12705196, 0.12595271, 0.13125198, 0.1249997, 0.12898503,
                      0.13129103, 0.12697463, 0.13014334, 0.12465323, 0.12162762, 0.12001673,
                      0.12385753, 0.12594433, 0.12780339, 0.12434061, 0.13004169, 0.12084778,
                      0.12954885, 0.12670184, 0.12737932, 0.12575061, 0.12845035, 0.12429133,
                      0.12597949, 0.12471514, 0.12827309, 0.12493933, 0.12811104, 0.12830668,
                      0.1219727, 0.12503628, 0.12790799, 0.12811979, 0.11854612, 0.12895779,
                      0.12869059, 0.1233125, 0.12838237, 0.1228213, 0.12019203, 0.12693328,
                      0.12672539, 0.12860215, 0.12423496, 0.12606879, 0.12248456, 0.12013415,
                      0.12090494, 0.12404068, 0.12702611, 0.12599069, 0.12332892, 0.12525315,
                      0.12659278, 0.12936787, 0.1279276, 0.12434265, 0.12832777, 0.12832434,
                      0.11903087, 0.13034679, 0.1276594, 0.13201058, 0.1276038, 0.12434575,
                      0.12233818, 0.13099694, 0.12582038, 0.12953246, 0.12450302, 0.12330207,
                      0.11850563, 0.13021369, 0.12886683, 0.12583837, 0.12700758, 0.12925807,
                      0.12804707, 0.12725599, 0.12746103, 0.12521667, 0.12941204, 0.13108502,
                      0.12405374, 0.12679301, 0.12499286, 0.12517607, 0.1297582, 0.12858817,
                      0.11904697, 0.12776128, 0.12801951, 0.12872841, 0.12515922, 0.12384326,
                      0.12629789, 0.1291678, 0.12587926, 0.12436892, 0.12369708, 0.12796784,
                      0.12549509, 0.12071269, 0.1252496, 0.1303736, 0.126881, 0.12439913,
                      0.12515401, 0.1294582, 0.12554884, 0.12467491, 0.1283463, 0.12381864,
                      0.12217782, 0.12533005, 0.13065801, 0.12818976, 0.12526317, 0.1287405,
                      0.12947795, 0.1248175, 0.12578077, 0.12355852, 0.12955656, 0.12214422,
                      0.12405569, 0.12492122, 0.1270086, 0.12963925, 0.12422131, 0.12494336,
                      0.12568182, 0.12029666, 0.12915866, 0.13069456, 0.13132913, 0.12313166,
                      0.1240527, 0.12687057, 0.128, 0.12348023, 0.12489188, 0.12504765,
                      0.12558567, 0.13211399, 0.12963967, 0.12824873, 0.13053955, 0.12289426,
                      0.12720329, 0.12323021, 0.12868777, 0.12843847, 0.12655346, 0.12815992,
                      0.12592084, 0.12198231, 0.12724704, 0.12047125, 0.12519992, 0.1282012,
                      0.13001419, 0.12390733, 0.12762829, 0.12624317, 0.12782972, 0.12804291,
                      0.13172431, 0.12716787, 0.12715159, 0.11984099, 0.1275181, 0.12799919,
                      0.12470336, 0.12366437, 0.12550278, 0.12527889, 0.12722411, 0.12836579,
                      0.12724553, 0.12177255, 0.12491209, 0.11958894, 0.12360862, 0.12101728,
                      0.12515056, 0.13034328, 0.12175376, 0.12876964, 0.12880487, 0.12252774,
                      0.12050614, 0.12738973, 0.12793857, 0.12606357, 0.12788889, 0.12105031,
                      0.12949096, 0.12366119, 0.12723569, 0.12483831, 0.13397385, 0.12983896,
                      0.12598025, 0.12760505, 0.12737161, 0.12142145, 0.13285513, 0.12557372,
                      0.12307463, 0.1210988, 0.12635823, 0.12326699, 0.1274528, 0.12993711,
                      0.13049799, 0.12897717, 0.12953138, 0.12497815, 0.12363274, 0.12210199,
                      0.12708087, 0.12971307, 0.1222876, 0.12147789, 0.12489016, 0.12731495,
                      0.12940105, 0.12700577, 0.12991438, 0.12475453, 0.12372804, 0.12732915,
                      0.13160868, 0.12033093, 0.11772307, 0.12927764, 0.13107377, 0.1227483,
                      0.1207266, 0.13057385, 0.12552263, 0.1293979, 0.12804535, 0.12457096,
                      0.12949711, 0.12676624, 0.12612564, 0.12673239, 0.12225215, 0.12712966,
                      0.12663428, 0.12553033, 0.12902279, 0.12609163, 0.12131309, 0.13082469,
                      0.12766622, 0.12502532, 0.13174961, 0.12804158, 0.12707459, 0.12697319,
                      0.1230011, 0.12265961, 0.12685395, 0.11944838, 0.1284179, 0.12297729,
                      0.12809926, 0.12215213, 0.12982059, 0.127797, 0.12472034, 0.13006016,
                      0.12131671, 0.12697008, 0.12440018, 0.12673311, 0.12630599, 0.12776639,
                      0.1217553, 0.13037482, 0.13015288, 0.12236539, 0.13165695, 0.13083816,
                      0.11801297, 0.12745211, 0.13293979, 0.12482729, 0.13161034, 0.1226577,
                      0.12509759, 0.12759475, 0.12461896, 0.12142551, 0.1189102, 0.1230551,
                      0.11891218, 0.12737948, 0.12597166, 0.12743946, 0.12875176, 0.12861127,
                      0.1215468, 0.12809727, 0.13056339, 0.12598028, 0.12557098, 0.12531174,
                      0.13153356, 0.13018066, 0.12673256, 0.12943423, 0.13133249, 0.12837968,
                      0.1273246, 0.12880482, 0.13077738, 0.12763032, 0.12700629, 0.12473891,
                      0.12903156, 0.12690867, 0.12858989, 0.1228742, 0.12452676, 0.12556734,
                      0.12487243, 0.12372429, 0.1218493, 0.12281779, 0.12828241, 0.13157002,
                      0.12512816, 0.12466109, 0.12644115, 0.12946715, 0.12579611, 0.12520813,
                      0.12795104, 0.12608737, 0.12529472, 0.12802828, 0.12744353, 0.12155538,
                      0.12667069, 0.12629363, 0.12666658, 0.1236445, 0.12716576, 0.13100311,
                      0.13144449, 0.12692861, 0.12537495, 0.12014889, 0.12856606, 0.12927578,
                      0.12919463, 0.12923525, 0.12699964, 0.13255091, 0.12720722, 0.11936542,
                      0.12370597, 0.12436042, 0.12170054, 0.12089817, 0.12768835, 0.12587822,
                      0.12724272, 0.13101427, 0.13058161, 0.12215676, 0.12406773, 0.12402599,
                      0.12797622, 0.12011488, 0.124519, 0.12546711, 0.12735978, 0.12027875,
                      0.12374605, 0.12855337, 0.12376568, 0.12106191, 0.13099126, 0.12405507,
                      0.1242317, 0.12822858, 0.1239341, 0.12484509, 0.12669598, 0.12639627,
                      0.12798173, 0.12528201, 0.12535389, 0.12766207, 0.13068328, 0.12842454,
                      0.12824258, 0.12516022, 0.12879764, 0.12390795, 0.12453879, 0.12642048,
                      0.12592308, 0.12388233, 0.12923485, 0.12944326, 0.12604513, 0.12484039,
                      0.13198472, 0.12907354, 0.13224608, 0.12994473, 0.12695602, 0.1294045,
                      0.12592259, 0.12440061, 0.12393322, 0.12275847, 0.12584471, 0.12896903,
                      0.12332743, 0.12536018, 0.12985324, 0.1231806, 0.12582452, 0.12646167,
                      0.12448393, 0.12720203, 0.12671204, 0.12980651, 0.1275493, 0.12811179,
                      0.12984869, 0.12991251, 0.12598101, 0.12862698, 0.12496086, 0.12420312,
                      0.12249999, 0.13066563, 0.12455465, 0.12035803, 0.12798124, 0.12599916,
                      0.12539199, 0.12216943, 0.12525655, 0.12840187, 0.12722009, 0.12787709]
m_list = meaningfulness
print(m_list)
m_list_sorted = np.sort(m_list)
bins = []

# Finding bin edges by looking at the word pool in groups of 1/10 and identifying the max bin values
for i in range(1,11):
    edge = m_list_sorted[int(i*len(m_list_sorted)/10)-1]
    bins.append(edge)

print("Bins",bins)

bin_zero = []
bin_one = []
bin_two = []
bin_three = []
bin_four = []
bin_five = []
bin_six = []
bin_seven = []
bin_eight = []
bin_nine = []


# Add the word frequency values to their corresponding bins depending on the max bin values
for i, freq in enumerate(m_list):
    if freq >= 0 and freq <= bins[0]:
        bin_zero.append([i, freq])
    elif (bins[0] < freq) and (freq <= bins[1]):
        bin_one.append([i, freq])
    elif (bins[1] < freq) and (freq <= bins[2]):
        bin_two.append([i, freq])
    elif (bins[2] < freq) and (freq <= bins[3]):
        bin_three.append([i, freq])
    elif (bins[3] < freq) and (freq <= bins[4]):
        bin_four.append([i, freq])
    elif (bins[4] < freq) and (freq <= bins[5]):
        bin_five.append([i, freq])
    elif (bins[5] < freq) and (freq <= bins[6]):
        bin_six.append([i, freq])
    elif (bins[6] < freq) and (freq <= bins[7]):
        bin_seven.append([i, freq])
    elif (bins[7] < freq) and (freq <= bins[8]):
        bin_eight.append([i, freq])
    elif (bins[8] < freq) and (freq  <= bins[9]):
        bin_nine.append([i, freq])

# Print lengths of each bin
#
print(len(bin_zero))
print(len(bin_one))
print(len(bin_two))
print(len(bin_three))
print(len(bin_four))
print(len(bin_five))
print(len(bin_six))
print(len(bin_seven))
print(len(bin_eight))
print(len(bin_nine))

# Sum of the lengths of each bin should be equal to the total number of words in the word pool
print((len(bin_zero) + len(bin_one) + len(bin_two) + len(bin_three) + len(bin_four) + len(bin_five) + len(bin_six) + len(bin_seven) + len(bin_eight) + len(bin_nine)))

# Confirming that the max bin values match the initial max bin values
# print("Bin Max Values")
# print(array[58-1])
# print(array[(58+60-1)])
# print(array[(58+60+54-1)])
# print(array[(58+60+54+59-1)])
# print(array[(58+60+54+59+57-1)])
# print(array[(58+60+54+59+57+57-1)])
# print(array[(58+60+54+59+57+57+58-1)])
# print(array[(58+60+54+59+57+57+58+57-1)])
# print(array[(58+60+54+59+57+57+58+57+58-1)])
# print(array[(58+60+54+59+57+57+58+57+58+58-1)])

# Finding the mean word frequency of each word bin
mean_zero = (np.mean(np.asarray(bin_zero), axis=0)[1])
mean_one = (np.mean(np.asarray(bin_one), axis=0)[1])
mean_two = (np.mean(np.asarray(bin_two), axis=0) [1])
mean_three = (np.mean(np.asarray(bin_three), axis=0)[1])
mean_four = (np.mean(np.asarray(bin_four), axis=0) [1])
mean_five = (np.mean(np.asarray(bin_five), axis=0)[1])
mean_six = (np.mean(np.asarray(bin_six), axis=0)[1])
mean_seven = (np.mean(np.asarray(bin_seven), axis=0)[1])
mean_eight = (np.mean(np.asarray(bin_eight), axis=0)[1])
mean_nine = (np.mean(np.asarray(bin_nine), axis=0) [1])

# Creating a table: Frequency Information for Each Word Bin

Bin = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Range = [" < 0.12646   ", " 0.12646 - 0.12775   ", " 0.12776 - 0.12599  ", " 0.12600 - 0.129458   ", " 0.129459 - 0.12804   ", " 0.12805 -  0.122287   ", "   0.122288 -  0.121755   ", " 0.121756 - 0.12946   ", " 0.12947 - 0.12528  ", " > 0.12529 "]
M = [mean_zero, mean_one, mean_two, mean_three,mean_four, mean_five,mean_six, mean_seven, mean_eight, mean_nine]
print(M)

t = Table([Bin, Range, M], names = ("Bin", "Range", "M"), dtype=('int', 'str', 'float'))
print('\033[1m' + "Table 1")
print("Frequency Information for Each Word Bin")
print('\033[0m')
print(t)