import pandas as pd
import numpy as np

df = pd.read_csv("top_num.csv", names=["Word", "Count"])
print(df)
df['cum_percent'] = 100*(df.Count.cumsum() / df.Count.sum())
df.to_csv("cum_top.csv")

