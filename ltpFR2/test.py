import glob
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.io



from scipy.stats.stats import pearsonr
import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table,Column

# These numbers come from the Python file
corr_word = [[-0.04922568, np.nan, np.nan, np.nan],
             [-0.03093967, -0.12375809, np.nan, np.nan],
             [ 0.11591534, -0.0533177 ,  0.03806268, np.nan],
             [ 0.47027421, -0.14924575,  0.00829743,  0.27704116 ]]

corr_word_abs = [[0.04922568, np.nan, np.nan, np.nan],
             [0.03093967, 0.12375809, np.nan, np.nan],
             [ 0.11591534, 0.0533177 ,  0.03806268, np.nan],
             [ 0.47027421, 0.14924575,  0.00829743,  0.27704116 ]]
#print(corr_word)
fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(corr_word_abs, cmap='Greys')

for (i, j), z in np.ndenumerate(corr_word):
    if np.isnan(z):
        continue
    else:
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.imshow(corr_word_abs,interpolation='none', cmap='Greys')
print(plt.spines.values)
plt.colorbar()
ax.set_xticklabels([[], 'Conc', 'Freq', 'Length', 'MList', 'MPool'])
ax.set_yticklabels([[], 'Freq', 'Length', 'MList', 'MPool'])
plt.show()