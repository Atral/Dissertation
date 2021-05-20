import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
df = pd.read_csv('SVM-kernels.csv')
print(df.columns)


overall = df[" Overall"].to_list()
changed = df[" Changed"].to_list()


labels = ["Wrong Correction", "Incorrectly Changed", "Incorrectly Unchanged"]
# nb = [0.0364,0.9636,0.01]
# svc = [0.0818,0.9182,0.0397]
# rf = [0.0929,0.9071,0.0473]
# dt = [0.0771,0.9229,0.0351]
# lr = [0.0897,0.9103,0.0513]
metrics = df[[' WrongCorrection', ' IncorrectlyChanged', ' IncorrectlyUnchanged']]
print(metrics)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,2)
f3_ax1 = fig.add_subplot(gs[0, 0])
f3_ax1.pie(metrics.loc[0], autopct='%1.1f%%', explode=(0.1, 0.1, 0.1), radius=1.2)
f3_ax1.set_title(labels[0])

f3_ax2 = fig.add_subplot(gs[1, 0])
f3_ax2.pie(metrics.loc[1], autopct='%1.1f%%', explode=(0.1, 0.1, 0.1), radius=1.2)
f3_ax2.set_title(labels[1])

f3_ax3 = fig.add_subplot(gs[1:])
f3_ax3.pie(metrics.loc[2], autopct='%1.1f%%', explode=(0.1, 0.1, 0.1), radius=1.2)
f3_ax3.set_title(labels[2])


plt.legend(labels, bbox_to_anchor=(1,0), loc="lower right",
                      bbox_transform=plt.gcf().transFigure)


plt.show()