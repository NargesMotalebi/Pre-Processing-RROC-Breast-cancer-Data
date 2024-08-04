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

from scipy.stats import mode, skew

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import csv
# Specify the directory containing your Excel


df_Treatment_Information1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Treatment Information.xlsx')
df_Treatment_Information2 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Treatment Information2.xlsx')
combined_df_TI = pd.concat([df_Treatment_Information1, df_Treatment_Information2], ignore_index=True)  # Combine rows, ignoring original indices
df = combined_df_TI 
df.columns.tolist()


df.drop(columns=['Morphology','MorphologyName','TopographyName','TreatmentProtocol','TreatmentDrugs','Cult'],inplace=True)#, 'Morphology', 'Topography', 'TopographyName']

# Define the list of columns to consider
columns_ = df.columns.tolist()
# Group by 'DocumentCode', sort=False to maintain original order
# For each group, keep the row with the smallest number of missing values
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]



# Filter the DataFrame to keep only rows related to the specified value in column 'BiopsiOrgan'
filtered_df = df[df['Topography'] == 'C50.9']
filtered_df = filtered_df[filtered_df['TreatmentOrgan'] ==  'BREAST C50']
df = filtered_df
#df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)
df.shape
df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')

# Define the percentage threshold for missing values
# percentage_threshold = 0.40 
# # Finding colums with missing data more than 40 percent
# for i in range(df.shape[1]):
#     num = df.iloc[:, i].isnull().sum() 
#     percentage = float(num) / df.shape[0] * 100
#     if percentage > percentage_threshold:
#        print('%d, %d, %.1f%%' % (i, num, percentage))


# # Calculate the number of allowable missing values based on the percentage threshold
# allowable_missing_values = int(len(df) * percentage_threshold)
# # Drop columns with missing values exceeding the allowable threshold
# df_cleaned = df.dropna(axis=1, thresh=len(df) - allowable_missing_values)
# df = df_cleaned
# get number of unique values for each column
counts = df.nunique()
columns_to_drop = ['BirthDate', 'MariageStatus', 'Education', 'JobStatus','Topography']
df = df.drop(columns = columns_to_drop)

persian_to_english = {
   
    'راديوتراپي': 'Radiotherapy',
    'هورمون تراپي': 'Hormone Therapy',
    'شيمي درماني': 'Chemotherapy',
    #'IMRT': 'Intensity-Modulated Radiation Therapy'
}

farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)


#decide if we want to delet TreatmentInformation or not


# Treatment_Dose = {
# 6000 : 60 ,
# 4256 : 4, 
# 1000 : 1,
# 5320 : 5,
# 5000 : 5,
# 5256 :9 ,
# 6400 : 6,
# 5050 : 7,
# 6600 : 8,
# 4050 :4,
# 4200 :3,
# 5400 :2,
# 5200 :3,
# 5250 :4,
# 5226 :3,
# 4160 :5,
# 5192 :4,
# 4250 :4,
# 3900 :4,
# 4620 :2,
# 6200 :5,
# 4000 :7,
# 4192 :3,
# 5350 :3,
# 600  :6,
# 7000 :7,
# 5236 :4,
# 4950 :2,
# 3990 :3,
# 4600 :4,
# 5580 :5,
# 5220 :4,

# }

# df['TotalDose'] = df['TotalDose'].replace(Treatment_Dose)









df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/TI.xlsx', index=False)
