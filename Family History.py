from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.impute import KNNImputer
import os
import csv

df_Family_History1 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Family History.xlsx') 
df_Family_History2 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Family History2.xlsx') 
combined_df_FH_ = pd.concat([df_Family_History1, df_Family_History2], ignore_index=True)  # Combine rows, ignoring original indices
df_Family_History3 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Family History1397.xlsx') 
df_Family_History3[df_Family_History3['Topography']] == 'C50.9' 
combined_df_FH1 = pd.concat([combined_df_FH_, df_Family_History3], ignore_index=True) 
df_Family_History4 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Family History1401.xlsx')
df_Family_History4[df_Family_History4['Topography']] == 'C50.9' 
combined_df_FH4 = pd.concat([combined_df_FH1, df_Family_History4], ignore_index=True) 
df = combined_df_FH4
df.shape
# Words to filter by
words_to_filter = ['C50'               
                   ]
# # Combine the words into a single regex pattern
pattern = '|'.join(words_to_filter)
# Use the query method to filter the DataFrame
filtered_combined_df_DI = df.query("Topography.str.contains(@pattern, case=False, na=False) or Topography.isna()")
df = filtered_combined_df_DI

df.drop(columns=['PatientId','FamilyHistoryTopography','Morphology','TopographyName','Topography'],inplace=True)#, 'Morphology', 'Topography', 'TopographyName']

df = df.dropna(axis=1, how='all')
df.columns.tolist()
# Specify the value to keep
# specified_value = 'C50.9'
# # Filter the DataFrame to keep only rows related to the specified value in column 'Organ'
# filtered_df = df[df['Topography'] == specified_value]
# df = filtered_df

# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)


# Define the percentage threshold for missing values
# percentage_threshold = 0.40 
# # Calculate the number of allowable missing values based on the percentage threshold
# allowable_missing_values = int(len(df) * percentage_threshold)
# # Drop columns with missing values exceeding the allowable threshold
# df_cleaned = df.dropna(axis=1, thresh=len(df) - allowable_missing_values)
# df = df_cleaned
# # Finding colums with missing data more than 40 percent
# for i in range(df.shape[1]):
#     num = df.iloc[:, i].isnull().sum() 
#     percentage = float(num) / df.shape[0] * 100
#     if percentage > percentage_threshold:
#        print('%d, %d, %.1f%%' % (i, num, percentage))


# Remove rows with all value missing    
df = df.dropna(axis = 0, how ='all') 
df = df.dropna(axis=1, how='all') 
# Define the list of columns to consider
columns_ = df.columns.tolist()
# # For each group, keep the row with the smallest number of missing values
# def keep_row_with_smallest_missing_values(group):
#     if len(group) == 1:
#         return group
#     else:
#         missing_values_counts = group[columns_].isnull().sum(axis=1)
#         min_missing_values_index = missing_values_counts.idxmin()
#         return group.loc[[min_missing_values_index]]

# df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

def merge_duplicate_rows(df, id_column):
    # Function to merge rows
    def merge_rows(group):
        return group.ffill().bfill().iloc[0]

    # Group by ID and apply the merge function
    merged_df = df.groupby(id_column).apply(merge_rows).reset_index(drop=True)
    
    return merged_df
merged_df = merge_duplicate_rows(df, 'DocumentCode')

df = merged_df

df.shape

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/FH.xlsx', index=False)
