from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import csv


def convert_persian_to_gregorian(persian_dates):
    """
    Convert a list of Persian dates to Gregorian dates.

    Parameters:
    - persian_dates (list): A list of Persian dates in the format 'yyyy/mm/dd'.

    Returns:
    - gregorian_dates (list): A list of Gregorian dates in the format 'yyyy-mm-dd'.
    """
    gregorian_dates = []

    for persian_date in persian_dates:
        if persian_date is None or pd.isnull(persian_date):
           gregorian_dates.append(None)  # Append None if birth date is NaN
        else:
           year, month, day = map(int, persian_date.split('/'))
           jalali_date = JalaliDate(year, month, day)
           gregorian_dates.append(jalali_date.to_gregorian().strftime("%Y-%m-%d"))

    return gregorian_dates


df =  pd.read_excel('G:/Breast Cancer/Reza Ancology/PI/Patientsituation-allyears.xlsx',usecols = ['DocumentCode','Topography','IsBeginningDisease', 
                                                                                                  'IsMetastasis', 'IsReactive', 'ReActiveDate',
                                                                                                  'MetastasisOrgans'])

# Words to filter by
words_to_filter = ['C50']

# Combine the words into a single regex pattern
pattern = '|'.join(words_to_filter)
# # Use the str.contains() method to filter the DataFrame
# filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=False)]
# Filter the DataFrame to keep rows containing the pattern or rows where 'Topography' is NaN
filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=True) | df['Topography'].isna()]

# Update the original DataFrame with the filtered data
df = filtered_df

df['DocumentCode'] = df['DocumentCode'].replace({'مشکل شماره پرونده': np.nan})

df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)

# Define a mapping dictionary for Persian to English
persian_to_english = {
    'خیر': 'No',
    'بلی': 'Yes',
    'بلي': 'Yes',
    'خير': 'No',
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

df.columns.tolist()
# Split the data in the 'MetastasisOrgans' column into columns
df[['FirstMetastasisOrgan', 'Firstperim']] = df['MetastasisOrgans'].str.split('--', n=1, expand=True)
df[['FirstDate', 'Secondperim']] = df['Firstperim'].str.split(',', n=1, expand=True)
df[['SecondMetastasisOrgan', 'Thirdprime']] = df['Secondperim'].str.split('--', n=1, expand=True)

df[['SecondDate', 'Forthprime']] = df['Thirdprime'].str.split(',', n=1, expand=True)
df[['ThirdMetastasisOrgan','Fifthprime']] = df['Forthprime'].str.split('--', n=1, expand=True)

df[['ThirdDate', 'Sixprime']] = df['Fifthprime'].str.split(',', n=1, expand=True)
df[['ForthMetastasisOrgan','ForthDate']] = df['Sixprime'].str.split('--', n=1, expand=True)


columns_to_drop = ['MetastasisOrgans','Firstperim','Secondperim',
                   'Thirdprime','Forthprime','Fifthprime','Sixprime']
df.drop(columns= columns_to_drop, inplace=True)
#****************************************************************
columns_ = df.columns.tolist()


column_replace = {
    'Liver C22.0' : 'Liver',
    'Bone, NOS C41.9' : 'Bone',
    'Brain, NOS C71.9': 'Brain',
    'BRAIN C71': 'Brain',
    'Lung, NOS C34.9' :'Lung',
    'PANCREAS C25' : 'Panceras',
    'Vertebral column C41.2' :'Vertebral column',
    'Connective, Subcutaneous and other soft tissues of upper limb and shoulder C49.1' : 'Upper limb and shoulder',
    'Mediastinum, NOS C38.3' : 'Mediastinum',
    ' Lung, NOS C34.9' :'Lung',
    ' Liver C22.0' :'Liver',
    ' Bone, NOS C41.9': 'Bone',
    ' BRONCHUS AND LUNG C34': 'Lung',
    ' Bone marrow C42.1': 'Bone',
    ' BRAIN C71' :'Brain',
    ' Brain, NOS C71.9':'Brain',
    ' Peritoneum, NOS C48.2' : 'Peritoneum',
    ' Peritoneum, NOS C48.2':'Peritoneum',    
    ' Skin of scalp and neck C44.4':'Skin of scalp and neck',
    ' Head, face or neck, NOS C76.0':'Head, face or neck',
    ' Vertebral column C41.2': 'Vertebral column',
    'Pelvis, NOS C76.3': 'Pelvis',    
    ' THYROID GLAND C73': 'THYROID GLAND',
    ' Pleura, NOS C38.4': 'Pleura',
    ' Meninges, NOS C70.9': 'Meninges',
    ' Bone, NOS C41.9':'Bone',
    ' Brain, NOS C71.9':'Brain',
    ' Liver C22.0' :'Liver',
    ' Lung, NOS C34.9':'Lung',
    ' BRAIN C71':'Brain',
    ' Eye, NOS C69.9':'Eye',
    ' Spleen C42.2':'Spleen',
    ' Rib, sternum, clavicle and associated joints C41.3':'Rib',
    ' Vertebral column C41.2':'Vertebral column',
    ' Skin of upper limb and shoulder C44.6':'Skin of upper limb and shoulder',
    ' Adrenal gland, NOS C74.9' : 'Adrenal gland',
    ' Lung, NOS C34.9': 'Lung',
    ' Lung, NOS C34.9' :'Lung',
    ' Skin, NOS C44.9' :'Skin',
    ' Bone, NOS C41.9': 'Bone',
    ' BRAIN C71' : 'Brain',
    'Spinal cord C72.0' : 'Spinal cord',
    'LARYNX C32' :'LARYNX',
    'Cerebellum, NOS C71.6' : 'Cerebellum',
    'Bone marrow C42.1' :'Bone',
    'Lymph nodes of head, face and neck C77.0' : 'Lymph nodes of head, face and neck',
    'Rib, sternum, clavicle and associated joints C41.3' : 'Rib',
    'BONES, JOINTS AND ARTICULAR CARTILAGE OF LIMBS C40' :'Bone',
    'Skin of trunk C44.5' : 'Skin',
    'Orbit, NOS C69.6' : 'Orbit',
    'Thorax, NOS C76.1' : 'Thorax',
    'Intra-abdominal lymph nodes C77.2' : 'Intra-abdominal lymph nodes'

}


df['FirstMetastasisOrgan'] = df['FirstMetastasisOrgan'].replace(column_replace)
df['SecondMetastasisOrgan'] = df['SecondMetastasisOrgan'].replace(column_replace)
df['ThirdMetastasisOrgan'] = df['ThirdMetastasisOrgan'].replace(column_replace)
df['ForthMetastasisOrgan']  = df['ForthMetastasisOrgan'].replace(column_replace)



G_ReActiveDate = convert_persian_to_gregorian(df['ReActiveDate'])
G_FirstMetastasicDate = convert_persian_to_gregorian(df['FirstDate'])
G_SecondMetastasicDate = convert_persian_to_gregorian(df['SecondDate'])
G_ThirdMetastasicDate = convert_persian_to_gregorian(df['ThirdDate'])
G_ForthMetastasicDate = convert_persian_to_gregorian(df['ForthDate'])

df['G_ReActiveDate'] = pd.to_datetime(G_ReActiveDate)
df['G_FirstMetastasicDate'] = pd.to_datetime(G_FirstMetastasicDate)
df['G_SecondMetastasicDate'] = pd.to_datetime(G_SecondMetastasicDate)
df['G_ThirdMetastasicDate'] = pd.to_datetime(G_ThirdMetastasicDate)
df['G_ForthMetastasicDate'] = pd.to_datetime(G_ForthMetastasicDate)

# Convert date columns to datetime format
date_columns = ['G_FirstMetastasicDate', 'G_SecondMetastasicDate', 
                'G_ThirdMetastasicDate', 'G_ForthMetastasicDate']
value_columns = ['FirstMetastasisOrgan', 'SecondMetastasisOrgan', 'ThirdMetastasisOrgan', 'ForthMetastasisOrgan']

# Function to sort dates and corresponding values
def sort_dates_and_values(row):
    # Drop pairs with NaT dates, sort dates, and align values accordingly
    valid_pairs = [(row[date], row[value]) for date, value in zip(date_columns, value_columns) if pd.notna(row[date])]
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[0])  # Sort by date

    # Extract sorted dates and values separately, padding with NaT and None if needed
    sorted_dates = [date for date, value in sorted_pairs] + [pd.NaT] * (len(date_columns) - len(sorted_pairs))
    sorted_values = [value for date, value in sorted_pairs] + [None] * (len(value_columns) - len(sorted_pairs))
    
    return pd.Series(sorted_dates + sorted_values)

# Apply the function and split results into separate columns for sorted dates and values
df[['FirstMetastastasicDate', 'SecondMetastastasicDate', 'ThirdMetastastasicDate', 'ForthMetastastasicDate', 'FirstMetastastasicsite', 'SecondMetastastasicsite', 'ThirdMetastastasicsite', 'ForthMetastastasicsite']] = df.apply(sort_dates_and_values, axis=1)

df = df.drop(columns=date_columns)
df = df.drop(columns=value_columns)
df = df.drop(columns=['FirstDate', 'SecondDate', 'ThirdDate', 'ForthDate',])
df.columns.tolist()
# Add an incremental index within each group to pivot data later
df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['G_ReActiveDate'])

# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]

# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 

def get_three_earliest_dates(row):
    # Drop any NaT values (missing dates), then sort and get unique dates
    dates = row.dropna().sort_values().unique()  # Ensure dates are unique

    # Return up to three dates, filling with NaT if there are fewer than three dates
    if len(dates) >= 3:
        return dates[0], dates[1], dates[2]
    elif len(dates) == 2:
        return dates[0], dates[1], pd.NaT
    elif len(dates) == 1:
        return dates[0], pd.NaT, pd.NaT
    else:
        return pd.NaT, pd.NaT, pd.NaT

# Apply the function to each row and create three new columns
df[['FirstReActiveDate', 'SecondReActiveDate', 'ThirdReActiveDate']] = df[['G_ReActiveDate_1.0', 'G_ReActiveDate_2.0',
    'G_ReActiveDate_5.0', 'G_ReActiveDate_6.0', 'G_ReActiveDate_7.0', 
    'G_ReActiveDate_8.0']].apply(get_three_earliest_dates, axis=1, result_type='expand')

# Format the dates as strings if they are not NaT
df['FirstReActiveDate'] = df['FirstReActiveDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['SecondReActiveDate'] = df['SecondReActiveDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['ThirdReActiveDate'] = df['ThirdReActiveDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['FirstMetastastasicDate'] = df['FirstMetastastasicDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['SecondMetastastasicDate'] = df['SecondMetastastasicDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['ThirdMetastastasicDate'] = df['ThirdMetastastasicDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)
df['ForthMetastastasicDate'] = df['ForthMetastastasicDate'].dt.strftime('%d/%m/%Y').fillna(pd.NaT)

columns_ = df.columns.tolist()
#For each group, keep the row with the smallest number of missing values
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]
df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)



df = df.drop(columns=['G_ReActiveDate_nan','idx','G_ReActiveDate_4.0','G_ReActiveDate_1.0','ReActiveDate',
                      'G_ReActiveDate_5.0','G_ReActiveDate_6.0','G_ReActiveDate_7.0','G_ReActiveDate_2.0',
                      'G_ReActiveDate_8.0','G_ReActiveDate','G_ReActiveDate_3.0','Topography'
                      ])

def set_status(row):
    if pd.notna(row['FirstMetastastasicDate']) and pd.notna(row['FirstReActiveDate']):
        return 3  # Both are not null
    elif pd.notna(row['FirstMetastastasicDate']):
        return 1  # Only FirstDate is not null
    elif pd.notna(row['FirstReActiveDate']):
        return 2  # Only ReActiveDate is not null
    else:
        return 0  # Both are null

# Apply the function row-wise to create the 'Status' column
df['Status'] = df.apply(set_status, axis=1)

def set_event(row):
    if pd.notna(row['FirstMetastastasicDate']) and pd.notna(row['FirstReActiveDate']):
        return 1  # Both are not null
    elif pd.notna(row['FirstMetastastasicDate']):
        return 1  # Only FirstDate is not null
    elif pd.notna(row['FirstReActiveDate']):
        return 1  # Only ReActiveDate is not null
    else:
        return 0  # Both are null

df['Event'] = df.apply(set_event, axis=1)

df['Status'].value_counts()




df.rename(columns={
'IsBeginningDisease': 'Is Beginning Disease', 
'IsMetastasis':  'Metastasis Status',
'IsReactive': 'Recurrence Status',
'FirstMetastastasicDate': 'First Metastastasic Date',
'SecondMetastastasicDate' : 'Second Metastastasic Date',
'ThirdMetastastasicDate': 'Third Metastastasic Date',
'ForthMetastastasicDate': 'Forth Metastastasic Date',
'FirstMetastastasicsite': 'First Metastastasic site',
'SecondMetastastasicsite': 'Second Metastastasic site', 
'ThirdMetastastasicsite': 'Third Metastastasic site', 
'ForthMetastastasicsite': 'Forth Metastastasic site',
'FirstReActiveDate': 'First Recurrence Date',
'SecondReActiveDate' : 'Second Recurrence Date', 
'ThirdReActiveDate': 'Third Recurrence Date', 
}, inplace=True)




df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/PS.xlsx', index=False)





