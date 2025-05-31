from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import csv


df_IHC=  pd.read_excel('G:/Breast Cancer/Reza Ancology/IHC/IHC.xlsx')
df_IHC = df_IHC.dropna(axis = 1, how ='all')  
df_IHC = df_IHC.dropna(axis = 0, how ='all')  


 #Pivot the DataFrame to create a structure with DocumentCode as index and Test Name as columns
pivot_df = df_IHC.pivot_table(index='DocumentCode', columns='Test Name', values=['status', 'IHC Intensity'], aggfunc='first')

# Apply transformation for 'status' columns (HER2, ER, PR) based on conditions
for col in ['status', 'IHC Intensity']:  # Apply to both status and IHC Intensity
    for test in ['HER2', 'ER', 'PR','P53','Fish or Cish']:  # Loop over relevant tests only
        if (col, test) in pivot_df.columns:  # Check if the column exists in the DataFrame
            pivot_df[(col, test)] = pivot_df[(col, test)].apply(lambda x: 'Positive' if x == '+' else ('Negative' if x == '-' else x))


# Now, select only the columns 'HER2', 'ER', 'PR' from both 'status' and 'IHC Intensity'
result_df = pivot_df[['status', 'IHC Intensity']].loc[:, (slice(None), ['HER2', 'ER', 'PR','P53','Fish or Cish'])]

# Flatten the MultiIndex columns for easier readability
result_df.columns = [f'{col[1]}_{col[0]}' for col in result_df.columns]

# Reset index to make DocumentCode a column
result_df.reset_index(inplace=True)

df = result_df
df.columns.tolist()

df.loc[(df['Fish or Cish_status'] == 1),'Fish or Cish_status'] = 'Positive'

counts = df.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
df = df.drop(df.columns[to_del], axis=1)


for col in ['PR_IHC Intensity', 'ER_IHC Intensity', 'HER2_IHC Intensity','P53_IHC Intensity']:
    df[col] = df[col].replace(0, np.nan)

for col in ['PR_status', 'ER_status']:
    df[col] = df[col].replace({
        3: 'Positive',
        1: 'Positive'
        })
df['HER2_status']= df['HER2_status'].replace({
        0:'Negative',
        1:'Negative',
        2:'Equivocal',
        3:'Positive'
        })


df['HER2_status'].value_counts()
df[df['HER2_status'] == 'Equivocal']['HER2_IHC Intensity'].value_counts()

df.loc[((df['HER2_status'] == 'Equivocal') & (df['Fish or Cish_status'] == 'Positive')),'HER2_status'] = 'Positive'
df.loc[((df['HER2_status'] == 'Equivocal') & (df['Fish or Cish_status'] == 'Negative')),'HER2_status'] = 'Negative'
df.loc[((df['HER2_status'] == 'Equivocal') & (df['HER2_IHC Intensity'] == 'Weakly')),'HER2_status'] = 'Negative'


df['PR_status'] = df['PR_IHC Intensity'].apply(
    lambda x: 'Positive' if x in ['Strongly', 'Moderately', 'Weakly'] else 'Negative'
)

df['ER_status'] = df['ER_IHC Intensity'].apply(
    lambda x: 'Positive' if x in ['Strongly', 'Moderately', 'Weakly'] else 'Negative'
)

df.drop('HER2_IHC Intensity', axis=1, inplace=True)  


def determine_molecular_subtype(row):
    """
    Determines the molecular subtype based on HER2, ER, and PR status.
    Returns NaN if all values are missing.
    """
    her2 = row["HER2_status"]
    er = row["ER_status"]
    er_intensity = row["ER_IHC Intensity"]
    pr = row["PR_status"]
    pr_intensity = row["PR_IHC Intensity"]

    # If all values are NaN, return NaN
    if pd.isna(her2) and pd.isna(er) and pd.isna(er_intensity) and pd.isna(pr) and pd.isna(pr_intensity):
        return np.nan

    if er == "Positive" or pr == "Positive":
        if her2 == "Positive":
            return "Luminal B (HER2-positive)"
        elif her2 == "Equivocal":
            return "Luminal B (HER2-moderate)"
        else:  # HER2 Negative
            if er_intensity == "Strongly" or pr_intensity == "Strongly":
                return "Luminal A"
            else:
                return "Luminal B (HER2-negative)"
    else:
        if her2 == "Positive":
            return "HER2-enriched"
        else:
            return "Triple-negative"


# Apply the function to determine the molecular subtype
df["Molecular_Subtype"] = df.apply(determine_molecular_subtype, axis=1)

df["Molecular_Subtype"].value_counts(dropna=False)
#df[['DocumentCode', 'Molecular_Subtype','HER2_status', 'ER_status', 'ER_IHC Intensity', 'PR_status', 'PR_IHC Intensity']].to_excel('G:/Breast Cancer/Reza Ancology/Edited/Subtype.xlsx', index=False)


df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/IHC.xlsx', index=False)