import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
baseline = "baseline.csv"
optimised = "optimised results.csv"
plt.style.use('seaborn')

df = pd.read_csv(baseline)
df2 = pd.read_csv(optimised)
print(df.columns)
overall_b = df[" Overall"].to_list()
changed_b = df[" Changed"].to_list()
labels = df2["Classifier"].to_list()
overall_o = df2[" Overall"].to_list()
changed_o= df2[" Changed"].to_list()
print(labels)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, overall_b, width, label='Overall (O)')
rects2 = ax.bar(x + width/2, changed_b, width, label='Changed (CH)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Overall Accuracy and Accuracy on Data Requiring Correction')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()