from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
sns.set_style('darkgrid')
sns.set_color_codes('bright')
import plotly.express as px
from sklearn.impute import KNNImputer
from scipy.stats import mode, skew
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import csv

imputer = KNNImputer(n_neighbors=10, weights='uniform', metric='nan_euclidean')
df_Midwifery_Information1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Midwifery Information.xlsx')
df_Midwifery_Information2 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Midwifery Information2.xlsx')
combined_df_MI_ = pd.concat([df_Midwifery_Information1, df_Midwifery_Information2], ignore_index=True)  # Combine rows, ignoring original indices

df_Midwifery_Information3 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Midwifery Information1397.xlsx')
df_Midwifery_Information3[df_Midwifery_Information3['Topography']] == 'C50.9' 
combined_df_MI1 = pd.concat([combined_df_MI_, df_Midwifery_Information3], ignore_index=True)  # Combine rows, ignoring original indices

df_Midwifery_Information4 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Midwifery Information1401.xlsx')
df_Midwifery_Information4[df_Midwifery_Information4['Topography']] == 'C50.9'
combined_df_MI = pd.concat([combined_df_MI1, df_Midwifery_Information4], ignore_index=True)  # Combine rows, ignoring original indices

df = combined_df_MI
df.shape
#df = df.drop(columns=['PatientId','Morphology','TopographyName','Topography'], inplace=True)
df.drop(columns=['Morphology','TopographyName','OCPUseLastDate','PatientId','Column31'], inplace=True)
#Filter the DataFrame to keep only rows related to the specified value in column 'BiopsiOrgan'
# filtered_df = df[df['Topography'] == 'C50.9']
# df = filtered_df
df.shape
# Remove rows with all value missing    
df = df.dropna(axis = 0, how ='all') 
#Drop columns with all NaN values
df = df.dropna(axis=1, how='all')
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)

df.shape
# Define a mapping dictionary for Persian to English
persian_to_english = {
    'خیر': 'NO',
    'بلي': 'YES',
    'هیسترکتومی':'Hysterectomy',  
    'راديوتراپي': 'Radiotherapy',
    'هورمون تراپي': 'Hormone Therapy',
     'شيمي درماني': 'Chemotherapy',
    'IMRT': 'Intensity-Modulated Radiation Therapy'    
}
farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)

df.loc[(df['MenopauseAge'] == 0) , 'MenopauseAge'] =  pd.NA

df.shape
# Define the list of columns to consider
columns_ = df.columns.tolist()
# For each group, keep the row with the smallest number of missing values
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)
df.shape

# Words to filter by
words_to_filter = ['C50']

# Combine the words into a single regex pattern
pattern = '|'.join(words_to_filter)

# Use the str.contains() method to filter the DataFrame
filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=False)]

df = filtered_df
df.shape

df.columns.tolist()
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/MI.xlsx', index=False)




