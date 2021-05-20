import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sampling = "sampling.csv"
plt.style.use('seaborn')

df = pd.read_csv(sampling)
labels = df["Classifier"]
overall = df[" Overall"]
changed = df[" Changed"]
df["Average"] = (overall + changed )/ 2
print(df["Average"])
wc = df[" WrongCorrection"].to_list()
ic = df[" IncorrectlyChanged"].to_list()
iu = df[' IncorrectlyUnchanged'].to_list()
legend = df.columns.to_list()
print(changed)


plt.plot(labels, df["Average"])
# plt.plot(labels, changed)
# plt.plot(labels, wc)
# plt.plot(labels, ic)
# plt.plot(labels, iu)

plt.title("Change in Accuracy with Increasing Non-error to Error Ratio")
plt.xlabel("Non-error to Error Data Ratio")
plt.ylabel("Accuracy (%)")

plt.legend(legend[1:3], bbox_to_anchor=(1,1), loc="upper right",
                      bbox_transform=plt.gcf().transFigure)

plt.show()