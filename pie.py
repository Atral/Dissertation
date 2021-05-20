import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
# df = pd.read_csv('classifier-selection.csv')


labels = ["Wrong Correction", "Incorrectly Changed", "Incorrectly Unchanged"]
data = [0.0818,0.9182,0.0397]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.set_title("Breakdown of Linear SVM Errors")
ax.pie(data, labels = labels,autopct='%1.2f%%')

plt.show()