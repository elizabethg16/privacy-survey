# up until line 160 is messing with the data and visualizations
# after 160 is the actual data cleaning
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import numpy as np
import plotly.express as px
import streamlit as st

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

# locally hosted interactive histogram (uncomment line 44 to display)
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

# example if a t test
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
df3 = pd.read_csv("providerSurvey.csv")
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


# fancy data visualization
FILE_NAME = 'providerSurvey.csv'

st.set_page_config(page_title="survey responses", layout="wide")

st.title("Provider Survey Data Vis")

@st.cache_data
def load_and_prep_data(filepath):
    try:
        df_raw = pd.read_csv(filepath)

        question_map = {}
        for col in df_raw.columns:

            q_text = str(df_raw.iloc[0][col])
            
            q_text_clean = q_text.replace("\n", " ").strip()
            if len(q_text_clean) > 100:
                q_text_clean = q_text_clean[:100] + "..."
                
            question_map[col] = f"{col}: {q_text_clean}"

        df_clean = df_raw.iloc[1:].copy()
        
        return df_clean, question_map
        
    except FileNotFoundError:
        st.error(f"Could not find file: {filepath}")
        return None, None


df, question_dict = load_and_prep_data(FILE_NAME)

if df is not None:

    options = list(question_dict.keys())
    
    selected_col = st.selectbox(
        "Select a Question:",
        options=options,
        format_func=lambda x: question_dict[x] 
    )

    if selected_col:
        st.divider()
        
        st.subheader(f"Question: {selected_col}")
        st.caption(question_dict[selected_col]) 
        
        series = df[selected_col].copy()
        
        series = series.fillna("No Answer")
        
        plot_df = pd.DataFrame({selected_col: series})
        
        fig = px.histogram(
            plot_df, 
            x=selected_col,
            title=f"Response Distribution for {selected_col}",
            text_auto=True, 
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_layout(
            xaxis_title="Response",
            yaxis_title="Count",
            bargap=0.2,
            xaxis={'categoryorder':'total descending'} 
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Response Counts Table"):
            counts = plot_df[selected_col].value_counts().reset_index()
            counts.columns = ['Response', 'Count']
            st.dataframe(counts, use_container_width=True)

#DATA CLEANING BEGINS!
# data cleaning based on state abortion legality and risk assesment 
df_full = pd.read_csv('providerSurvey.csv')

df_data = df_full.iloc[1:].copy() #exlcude first row because its questions
df_data.reset_index(drop=True, inplace=True)

initial_count = len(df_data)

# exclude null
# Q8: Think about the primary setting where you provide reproductive care. What abortion services, if any, do they provide? Select all that apply.
n_before_q8 = len(df_data)

dropped_q8_df = df_data[df_data['Q8'].notnull()]

df_data = df_data.dropna(subset=['Q8']) #Q8 is the first question that participants need to answer when they are officially in the survey
#so, if we drop all the participants who didnt answer this question, we are left with only valid responses
n_after_q8 = len(df_data)
dropped_q8 = n_before_q8 - n_after_q8

print(dropped_q8_df)
dropped_q8_df.to_csv('providerSurvey_dropped_q8', index=False)

# Defines abortion legality terms
RISK_LEGAL = 'Abortion is fully legal with no major bans or restrictions'
RISK_18W = 'Abortion is banned or seriously restricted at 18 weeks or beyond (e.g., around viability)'
RISK_12W = 'Abortion is banned or seriously restricted at 12 weeks'
RISK_6W = 'Abortion is banned or seriously restricted at 6 weeks'
RISK_BAN = 'Abortion is nearly totally banned in all or most circumstances'

states_legal = [
    'Alaska', 'Colorado', 'District of Columbia', 'Maryland', 'Michigan', 
    'Minnesota', 'New Jersey', 'New Mexico', 'Oregon', 'Vermont'
]
states_18w = [
    'Arizona', 'California', 'Connecticut', 'Delaware', 'Hawaii', 'Illinois', 
    'Kansas', 'Maine', 'Massachusetts', 'Montana', 'Nevada', 'New Hampshire', 
    'New York', 'Ohio', 'Pennsylvania', 'Rhode Island', 'Utah', 'Virginia', 
    'Washington', 'Wisconsin', 'Wyoming'
]
states_12w = ['Nebraska', 'North Carolina']
states_6w = ['Florida', 'Georgia', 'Iowa', 'South Carolina']
states_ban = [
    'Alabama', 'Arkansas', 'Idaho', 'Indiana', 'Kentucky', 'Louisiana', 
    'Mississippi', 'North Dakota', 'Oklahoma', 'South Dakota', 'Tennessee', 
    'Texas', 'West Virginia'
]

likert_map = {
    'Not at all important': 1,
    'Slightly important': 2,
    'Somewhat important': 3,
    'Very important': 4,
    'Extremely important': 5
}

risk_pairs = [
    ('patient_risk_importa_1', 'patient_risk_importa_6'),
    ('patient_risk_importa_2', 'patient_risk_importa_7'),
    ('patient_risk_importa_3', 'patient_risk_importa_8'),
    ('patient_risk_importa_4', 'patient_risk_importa_9'),
    ('patient_risk_importa_5', 'patient_risk_importa_10')
]

def check_state_alignment(row):
    state = row['Q1']
    risk = row['state_risk']
    
    if pd.isna(state) or pd.isna(risk):
        return True 
    
 
    if state == 'Missouri':
        if risk == RISK_LEGAL:
            return False
        return True
        
    if state in states_legal:
        return risk == RISK_LEGAL
    if state in states_18w:
        return risk == RISK_18W
    if state in states_12w:
        return risk == RISK_12W
    if state in states_6w:
        return risk == RISK_6W
    if state in states_ban:
        return risk == RISK_BAN
        
    return True

def check_risk_importance(row):
    for col_non, col_res in risk_pairs:
        val_non_str = row[col_non]
        val_res_str = row[col_res]
        
        if pd.isna(val_non_str) or pd.isna(val_res_str):
            continue
            
        val_non = likert_map.get(val_non_str)
        val_res = likert_map.get(val_res_str)
        
        if val_non is None or val_res is None:
            continue
            
        if val_non > val_res:
            return False
            
    return True

# filter
n_before_align = len(df_data)

df_data['alignment_pass'] = df_data.apply(check_state_alignment, axis=1)

#csv for values dropped because of an incorect state-legality pairing
df_state_misaligned = df_data[~df_data['alignment_pass']].copy()
df_state_misaligned.to_csv(
    'providerSurvey_dropped_state_alignment.csv',
    index=False
)

df_aligned = df_data[df_data['alignment_pass']].copy()
n_after_align = len(df_aligned)
dropped_align = n_before_align - n_after_align

n_before_logic = len(df_aligned)

df_aligned['logic_pass'] = df_aligned.apply(check_risk_importance, axis=1)

df_final = df_aligned[df_aligned['logic_pass']].copy()
n_after_logic = len(df_final)
dropped_logic = n_before_logic - n_after_logic

df_final.drop(columns=['alignment_pass', 'logic_pass'], inplace=True)
# just calculating and displaying stats
total_dropped = initial_count - len(df_final)

print(f"Initial Count: {initial_count}")
print(f"Dropped due to incomplete 'Q8': {dropped_q8}")
print(f"Dropped due to State Alignment mismatch: {dropped_align}")
print(f"Dropped due to Risk Importance Logic mismatch: {dropped_logic}")
print(f"Total Dropped: {total_dropped}")
print(f"Final Count: {len(df_final)}")

df_final.to_csv('providerSurvey_cleaned_v2.csv', index=False)