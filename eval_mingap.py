import os

import numpy as np
import pandas as pd

from collections import defaultdict

p = 0.5
M = 5
formulation = 'proposed'


solutions = defaultdict(list)
keys = ['id', 'eigenval', 'mingap']
solutions.fromkeys(keys, [])

dir_path = os.path.dirname(os.path.abspath(__file__))
if formulation == 'proposed':
    directory = os.path.join(dir_path, "results/mingap/eigenvalues")
elif formulation == 'laserre':
    directory = os.path.join(dir_path, "results/mingap/eigenvalues_l")
sol_directory = os.path.join(directory, str(formulation) + "_"+ str(p) + "_" + str(M) + ".xlsx")
tolerance = 1 #GHz

for filename in os.listdir(directory):
    if filename.endswith("_M"+ str(M) + ".npy") and filename.startswith("erdos_"+ str(p)) and not filename.startswith("erdos_idx"):
        eigenvalues = np.load(os.path.join(directory, filename), allow_pickle=True)
        for i in range(1, eigenvalues.shape[1]):
            if min(abs(eigenvalues[:, i] - eigenvalues[:, 0])) < tolerance:
                pass
            else:
                break
        print('Minimup gap computed with ' + str(i) + 'th eigenvalue')
        gap = eigenvalues[:, i] - eigenvalues[:, 0]

        solutions['id'].append((filename.split(str(p) + "_")[1]).split("_")[0])

        solutions['eigenval'].append(i)

        mingap = min(gap)
        print(mingap)


        solutions['mingap'].append(mingap)

sol_total = pd.DataFrame.from_dict(solutions)
sol_total.to_excel(sol_directory)
