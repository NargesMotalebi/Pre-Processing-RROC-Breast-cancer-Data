
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

df_Patient_Situation1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Patient Situation.xlsx')
df_Patient_Situation2 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Patient Situation2.xlsx')
combined_df_PS = pd.concat([df_Patient_Situation1, df_Patient_Situation2], ignore_index=True)  # Combine rows, ignoring original indices
df = combined_df_PS
df.shape
df.drop(columns=['Morphology', 'MorphologyName', 'TopographyName'], inplace=True)
# Filter the DataFrame to keep only rows related to the specified value in column 'BiopsiOrgan'
filtered_df = df[df['Topography'] == 'C50.9']
df = filtered_df
df = df.dropna(axis=1, how='all')
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)


# Define a mapping dictionary for Persian to English
persian_to_english = {
    'خیر': 'NO',
    'بلی': 'YES',
    'بلي': 'YES',
    'خير': 'NO',
    'مرد': 'Men',
    'زن': 'Female',
    'ديپلم': 'Educated',
    'ابتدايي': 'Illiterate',
    'كارشناسي': 'Educated',
    'كارداني': 'Educated',
    'نهضت': 'Illiterate',
    'راهنمايي': 'Illiterate',
    'بيسواد': 'Illiterate',
    'كارشناسي ارشد': 'Educated',
    'سواد قرآني': 'Illiterate',
    'زيرديپلم': 'Illiterate',
    'دكتري': 'Educated',
    'متاهل': 'Married',
    'مطلقه': 'Divorced (Female)',
    'همسرفوت شده': 'Widow/Widower',
    'مجرد': 'Single',
    'بيماري اوليه' :'Primary disease',
    'بيماري اوليه' :'Primary disease',
    'متاستاتيک': 'Metastatic',
    'متاستاتيک-عود شده': 'Metastatic - Recurrent',
    'بيماري اوليه-متاستاتيک-عود شده' : 'Primary disease -Metastatic - Recurrent',
    'بيماري اوليه-متاستاتيک': 'Primary disease -Metastatic',
    'عود شده-متاستاتيک': 'Recurrent -Metastatic',
    'عود شده': 'Recurrent',
    'عود شده-بيماري اوليه': 'Recurrent - Primary disease',
    'متاستاتيک-متاستاتيک': 'Metastatic -Metastatic',
    'بيماري اوليه-عود شده': 'Primary disease - Recurrent',
    'عود شده-عود شده': 'Recurrent -Recurrent',
    'بيماري اوليه-بيماري اوليه': 'Primary disease - Primary disease',
    }

farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)




# Split the data in the 'MetastasisOrgans' column into columns
df[['FirstMetastasisOrgan', 'Firstperim']] = df['MetastasisOrgans'].str.split('--', n=1, expand=True)
df[['FirstDate', 'Secondperim']] = df['Firstperim'].str.split(',', n=1, expand=True)
df[['SecondMetastasisOrgan', 'SecondDate']] = df['Secondperim'].str.split('--', n=1, expand=True)

columns_to_drop = ['MetastasisName','Firstperim','Secondperim']
df.drop(columns= columns_to_drop, inplace=True)

# # Define the percentage threshold for missing values
# percentage_threshold = 0.95  
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

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/PS.xlsx', index=False)