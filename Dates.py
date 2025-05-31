from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import csv

#********************************************Determind Survival Time*************************
reference_date = pd.to_datetime('2024-09-21')

def calculate_survival_time(row):
    biopsy_date = row['First Biopsi Date']
    
    if row['Status'] == 1:
        # Event 1: First Metastasis Date - Biopsy Date
        if pd.notna(row['First Metastastasic Date']):
            return (row['First Metastastasic Date'] - biopsy_date).days
        else:
            return np.nan  # if First Metastasis Date is missing
        
    elif row['Status'] == 2:
        # Event 2: Recurrence Date - Biopsy Date
        if pd.notna(row['First Recurrence Date']):
            return (row['First Recurrence Date'] - biopsy_date).days
        else:
            return np.nan  # if Recurrence Date is missing
        
    elif row['Status'] == 0:
        # Event 0: Biopsy Date - Reference Date (01/01/2024)
        return (reference_date - biopsy_date).days
    
    elif row['Status'] == 3:
        # Event 3: Minimum of Recurrence Date and First Metastasis Date - Biopsy Date
        dates = [row['First Recurrence Date'], row['First Metastastasic Date']]
        # Filter out any NaT values
        valid_dates = [date for date in dates if pd.notna(date)]
        
        if valid_dates:
            nearest_date = min(valid_dates)
            return (nearest_date - biopsy_date).days
        else:
            return np.nan  # if both dates are missing


df=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DE.xlsx')
df_DIO =  pd.read_excel('G:/Breast Cancer/Reza Ancology//Edited/DI.xlsx')
df = pd.merge(df, df_DIO,  on=['DocumentCode'], how='outer')
df_SUR =  pd.read_excel('G:/Breast Cancer/Reza Ancology//Edited/SU.xlsx')
df = pd.merge(df, df_SUR,  on=['DocumentCode'], how='outer')
df_TI  =  pd.read_excel('G:/Breast Cancer/Reza Ancology//Edited/TI.xlsx')
df = pd.merge(df, df_TI,  on=['DocumentCode'], how='outer')

df_PS = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/PS.xlsx')
df = pd.merge(df, df_PS,  on=['DocumentCode'], how='outer')

df_Biobsy = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis.xlsx', usecols = ['DocumentCode','MetastasisName','Topography'])
df = pd.merge(df, df_Biobsy,  on=['DocumentCode'], how='outer', suffixes=('','_olddf'))

df_Biobsy_ = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis2.xlsx', usecols = ['DocumentCode','Column23','Topography'])

df_Biobsy_.rename(columns={
 'Column23' :  'MetastasisName', 
}, inplace=True)

df = pd.merge(df, df_Biobsy_, on=['DocumentCode'], how='outer', suffixes=('', '_df2'))


commonlistofcolumns = ['MetastasisName','Topography']

# Combine columns with '_df2' suffix into the original columns
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_df2'])

# Drop the '_df2' columns as they are now redundant
df = df.drop(columns=[f'{col}_df2' for col in commonlistofcolumns])

# df[['FirstMetastasis', 'SecondMetastasis']] = df['MetastasisName'].str.split('-', n=1, expand=True)
# df[['SecondMetastasis_', 'ThirdMetastasis']] = df['SecondMetastasis'].str.split('-', n=1, expand=True)
# df.drop(columns=['SecondMetastasis'], inplace=True)

df.columns.tolist()
df['First Biopsi Date'] = df['First Biopsi Date'].fillna(df['First Surgery Date'])
df['Type of First Treatment'].value_counts(dropna=False)
df['Type of First Treatment'] = df['Type of First Treatment'].fillna(df['Type of Second Treatment'])
df[df['Type of First Treatment'] == 'Chemotherapy']['Status'].value_counts()

df['Status'].value_counts(dropna=False)
start_date = pd.to_datetime('2019/03/21')
end_date = pd.to_datetime('2024/09/21')
df = df.dropna(subset=['First Biopsi Date', 'First Surgery Date'], how='all')
df.shape
df['First Biopsi Date'] = pd.to_datetime(df['First Biopsi Date'], dayfirst=True, errors='coerce')
df['First Metastastasic Date'] = pd.to_datetime(df['First Metastastasic Date'], dayfirst=True, errors='coerce')
df['First Recurrence Date'] = pd.to_datetime(df['First Recurrence Date'], dayfirst=True, errors='coerce')
df['First Surgery Date'] = pd.to_datetime(df['First Surgery Date'], dayfirst=True, errors='coerce')
df['First Treatment Date'] = pd.to_datetime(df['First Treatment Date'], dayfirst=True, errors='coerce')
df['Birth Day'] = pd.to_datetime(df['Birth Day'], dayfirst=True, errors='coerce')
df['Document Date'] = pd.to_datetime(df['Document Date'], dayfirst=True, errors='coerce')
df = df[(df['First Treatment Date'].isna()) | ((df['First Treatment Date'] >= start_date) & (df['First Treatment Date'] <= end_date))]
df = df[(df['First Biopsi Date'].isna()) | ((df['First Biopsi Date'] >= start_date) & (df['First Biopsi Date'] <= pd.to_datetime('2024/03/21')))]
df = df[(df['First Surgery Date'].isna()) | ((df['First Surgery Date'] >= start_date) & (df['First Surgery Date'] <= end_date))]
df = df[(df['First Recurrence Date'].isna()) | ((df['First Recurrence Date'] >= start_date) & (df['First Recurrence Date'] <= end_date))]
df = df[(df['First Metastastasic Date'].isna()) | ((df['First Metastastasic Date'] >= start_date) & (df['First Metastastasic Date'] <= end_date))]

df['Survival Time'] = df.apply(calculate_survival_time, axis=1)


# split_columns = df['MetastasisName'].str.split('-', expand=True)


# split_columns.columns = [f'Column_{i+1}' for i in range(split_columns.shape[1])]


# df = pd.concat([df, split_columns], axis=1)


# df = pd.merge(df, df,  on=['DocumentCode'], how='outer')

persian_to_english = {
    'خیر': 'No',
    'بلی': 'Yes',
    'بيماري اوليه' :'Primary disease',
    'بيماري اوليه' : 'Primary disease',
    'متاستاتيک': 'Metastatic',
    'متاستاتيک': 'Metastatic',
    'متاستاتيک-عود شده': 'Metastatic - Recurrent',
    'بيماري اوليه-متاستاتيک-عود شده' : 'Primary disease - Metastatic - Recurrent',
    'بيماري اوليه-متاستاتيک': 'Primary disease - Metastatic',
    'عود شده': 'Recurrent',
    'عود شده': 'Recurrent',
    'عود شده-بيماري اوليه': 'Recurrent - Primary disease',
    'متاستاتيک-متاستاتيک': 'Metastatic - Metastatic',
    'بيماري اوليه-عود شده': 'Primary disease - Recurrent',
    'عود شده-متاستاتيک': 'Recurrent -Metastatic',
    'متاستاتيک-متاستاتيک': 'Metastatic -Metastatic',
    'عود شده-عود شده': 'Recurrent -Recurrent',
    'بيماري اوليه-بيماري اوليه': 'Primary disease - Primary disease',
    'متاستاتيک-بيماري اوليه' : 'Primary disease - Metastatic',
}

columns = ['MetastasisName','FirstMetastasis', 'SecondMetastasis_','ThirdMetastasis']
df = df.replace(persian_to_english)

#df = df.dropna(subset=['MetastasisName'])
#******************************************************************************************************
columns_ = df.columns.tolist()  

def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group  # If only one row, keep it
    else:
        # Calculate missing values count for each row in the group (across all columns)
        missing_values_counts = group.isnull().sum(axis=1)
        # Find the index of the row with the minimum number of missing values
        min_missing_values_index = missing_values_counts.idxmin()
        # Return only the row with the least missing values
        return group.loc[[min_missing_values_index]]

# Apply the function within each group of 'DocumentCode'
df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)
#***********************************************************Determine Treatmentprotocol, Age at Diagnosis**********************************
df['Age at Diagnosis'] = df.apply(
    lambda row: (
        round(((row['First Biopsi Date'] if pd.notna(row['First Biopsi Date']) else row['First Surgery Date']) - row['Birth Day']).days / 365.25, 1)
        if pd.notna(row['Birth Day']) and (pd.notna(row['First Biopsi Date']) or pd.notna(row['First Surgery Date'])) else pd.NA
    ),
    axis=1
)
df.shape
#****************************************************************
df['Treatment Procedure'] = np.where(
    df['First Biopsi Date'].isna(),  
    np.nan,  
    np.where(
        
        (df['First Surgery Date'].isna()) & 
        ((df['First Biopsi Date'] - df['Document Date']).dt.days < 30) & 
        ((df['Type of First Treatment'] == 'Chemotherapy') | 
         (df['Type of Second Treatment'] == 'Chemotherapy') ),
        'Neo-Adjuvant',
        
        np.where(
            (df['First Surgery Date'] - df['First Biopsi Date']).dt.days >= 65,  
            'Neo-Adjuvant',
            'Adjuvant'  
        )
    )
)

#********************************************
df.loc[(df['TreatmentProtocol'] == 'Adjuvant') & (df['Treatment Procedure'] == 'Neo-Adjuvant'), 'TreatmentProtocol'] = 'Neo-Adjuvant'
df['TreatmentProtocol'] = df['TreatmentProtocol'].fillna(df['Treatment Procedure'])


df['First Surgery Type'].replace({
'mandibulectomy' : np.nan,
'Cecocolectomy'   : np.nan,
'bilobectomy' : np.nan,

}, inplace=True)

df['Grade'].replace({
'Undifferentiated' : 'Poorly Differentiated',
},inplace=True)

#df = df[df['TreatmentProtocol'] == 'Adjuvant']
df = df[(~df['Status'].isnull())]
df = df[~((df['Survival Time']> 1825))]
df=df[df['Event'] == 1]
df['Status'].value_counts(dropna=False)
df.shape
df = df[((df['Event'] == 1) & (df['Survival Time'] > 240))]
#df = df[~((df['Is Beginning Disease'] == 'No') & (df['Metastasis Status'] =='Yes'))]
df.columns.tolist()
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/stage4.xlsx', index=False)