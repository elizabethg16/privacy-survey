import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import numpy as np
import plotly.express as px

# preparing csv
df = pd.read_csv("providerSurvey.csv", header=0, skiprows=[1])
df = df.iloc[:, 13:]
df1 = df.iloc[3:14] 
df2 = df.iloc[14:] 

# print some data to terminal
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    print(df1.iloc[:, :10].head())
    print(df2.iloc[:, :10].head())


# example of generating statistical data
scores = df1['Q12_1'].dropna().astype(float)
mean = np.mean(scores)
median = np.median(scores)
mode = stats.mode(scores, keepdims=True)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode.mode[0])

# locally hosted interactive histogram
fig = px.histogram(df, x="S_intro", title="Interactive Histogram")
fig.update_layout(
    updatemenus=[{
        "buttons": [
            {"method": "update", "label": col, 
             "args": [{"x": [df[col]]}, {"title": f"Distribution of {col}"}]}
            for col in df.columns
        ],
        "direction": "down",
        "showactive": True
    }]
)
#fig.show()

# random t test
df.loc[:, 'S_intro'] = pd.to_numeric(df['S_intro'], errors='coerce')

df = df.dropna(subset=['S_intro', 'S_role'])  

group_works = df[df['S_role'].str.lower() == 'Yes']['S_intro']
group_other = df[df['S_role'].str.lower() == 'No']['S_intro']

t_stat, p_val = stats.ttest_ind(group_works, group_other, nan_policy='omit')

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("significant difference between groups.")
else:
    print("no significant difference between groups.")

#for each question/big question
#how many people answered each question
#results for each question
df3 = pd.read_csv("providerSurvey.csv", header=0)
df3v2 =df3.iloc[:, 13:] 
df3v3 = df3.dropna(subset=['Q8']) # only entries which have sucessfully got to the survey

for col in df3v3.columns:
    print("="*80)
    print(f"QUESTION: {col}")
    print("-"*80)

    counts = (
        df3v3[col]
        .value_counts(dropna=False)
        .reset_index()
    )
    counts.columns = ["Answer", "Count"]

    print(counts.to_string(index=False))
    print("\n")