from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import csv


def remove_outliers_iqr(df, column_name, thresh=1.5):

  Q1 = df[column_name].quantile(0.25)
  Q3 = df[column_name].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - (thresh * IQR)
  upper_bound = Q3 + (thresh * IQR)

  # Use boolean indexing to set outliers to NaN
  df.loc[(df[column_name] < lower_bound) | (df[column_name] > upper_bound), column_name] = pd.NA

  return df
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

df1 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DE.xlsx', usecols=['DocumentCode','Sex'])
df =  pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery-allyears.xlsx', usecols= ['FirstPeriodAge', 'GestationCount', 'AbortionCount', 
                                                                                           'StillbirthCount', 'FirstGestationAge', 'LastGestationAge', 
                                                                                           'LactationStatus', 'LoctationDurationTime', 'CesareanStatus', 
                                                                                           'CesareanCount', 'IsRegularPerion', 'MenopauseStatus',
                                                                                           'MenopauseAge', 'MenopauseCause', 'UsingOCP', 
                                                                                           'OCPDurationTime', 'OCPUseLastDate',
                                                                                           'DocumentCode'])

df = df.merge(df1, on ='DocumentCode', how = 'outer')

df2 = pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery Information.xlsx', usecols=['FirstPeriodAge', 
                                                                                             'GestationCount', 'AbortionCount', 'StillbirthCount', 
                                                                                             'FirstGestationAge','LastGestationAge', 'LactationStatus',
                                                                                             'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                                                                                              'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                                                                                              'MenopauseCause', 'UsingOCP','OCPDurationTime', 'OCPUseLastDate', 'DocumentCode']) 

commonlistofcolumns = ['FirstPeriodAge', 
                      'GestationCount', 'AbortionCount', 'StillbirthCount', 
                      'FirstGestationAge','LastGestationAge', 'LactationStatus',
                      'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                      'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                      'MenopauseCause', 'UsingOCP','OCPDurationTime', 'OCPUseLastDate']
df = pd.merge(df, df2,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])



df3 = pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery Information2.xlsx', usecols=['FirstPeriodAge', 
                                                                                             'GestationCount', 'AbortionCount', 'StillbirthCount', 
                                                                                             'FirstGestationAge','LastGestationAge', 'LactationStatus',
                                                                                             'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                                                                                              'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                                                                                              'MenopauseCause', 'UsingOCP','OCPDurationTime', 'OCPUseLastDate', 'DocumentCode']) 

commonlistofcolumns = ['FirstPeriodAge', 
                      'GestationCount', 'AbortionCount', 'StillbirthCount', 
                      'FirstGestationAge','LastGestationAge', 'LactationStatus',
                      'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                      'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                      'MenopauseCause', 'UsingOCP','OCPDurationTime', 'OCPUseLastDate']
df = pd.merge(df, df3,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])



df1397 = pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery Information1397.xlsx', usecols=['FirstPeriodAge', 
                                                                                             'GestationCount', 'AbortionCount', 'StillbirthCount', 
                                                                                             'FirstGestationAge','LastGestationAge', 'LactationStatus',
                                                                                             'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                                                                                              'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                                                                                              'MenopauseCause', 'UsingOCP','OCPDurationTime', 'DocumentCode']) 

commonlistofcolumns = ['FirstPeriodAge', 
                      'GestationCount', 'AbortionCount', 'StillbirthCount', 
                      'FirstGestationAge','LastGestationAge', 'LactationStatus',
                      'LoctationDurationTime', 'CesareanStatus', 'CesareanCount', 
                      'IsRegularPerion', 'MenopauseStatus',  'MenopauseAge', 
                      'MenopauseCause', 'UsingOCP','OCPDurationTime']
df = pd.merge(df, df1397,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df_mariage = pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/MaritalStatus.xlsx')
df = pd.merge(df, df_mariage,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )


df_mariage1 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis.xlsx', usecols=['DocumentCode','MariageStatus'])
df_mariage3 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis3.xlsx', usecols=['DocumentCode','MariageStatus'])
df_mariage1397 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1397.xlsx', usecols=['DocumentCode','MariageStatus'])
df_mariage1399 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1399.xlsx', usecols=['DocumentCode','MariageStatus'])
df_mariage1400 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1400.xlsx', usecols=['DocumentCode','MariageStatus'])
df_mariage1401 = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1401.xlsx', usecols=['DocumentCode','MariageStatus'])

df_mar = pd.concat([df_mariage1, df_mariage3, df_mariage1397, df_mariage1399, df_mariage1400, df_mariage1401])

commonlistofcolumn = ['MariageStatus']
df = pd.merge(df, df_mar,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumn:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumn])


persian_to_english = {
    'خیر': 'No',
    'بلی': 'Yes',
    'بلي': 'Yes',
    'هیسترکتومی': 'Hysterectomy',
    'متاهل': 'Married',
    'مطلقه': 'Divorced/widowed (DW)',
    'همسرفوت شده': 'Divorced/widowed (DW)',
    'مجرد': 'Single',
    'مسلمان' : np.nan
}

farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)

df.loc[(df['Sex'] == 'Men') & (df['FirstPeriodAge'].notnull()), 'Sex'] = 'Female'
#***************************************************************
df.loc[(df['FirstPeriodAge'] == 0), 'FirstPeriodAge'] =  np.nan
df.loc[(df['FirstGestationAge'] == 0), 'FirstGestationAge'] =  np.nan
df.loc[(df['LastGestationAge'] == 0), 'LastGestationAge'] =  np.nan
df.loc[(df['MenopauseAge'] == 0), 'MenopauseAge'] =  np.nan
#*****************************************************************
#******************************************************************
colum = ['FirstPeriodAge', 'GestationCount', 'AbortionCount', 
        'StillbirthCount', 'FirstGestationAge','LastGestationAge', 
        'CesareanCount', 'MenopauseAge', 'OCPDurationTime']

for col in colum:  
  df = remove_outliers_iqr(df.copy(), col) 
 
#*********************************************************
df.loc[(df['GestationCount'] != 0) & (df['GestationCount'].notnull()), 'MariageStatus'] = 'Married'
df.loc[(df['LoctationDurationTime'] == 0) & (df['GestationCount']== 0), 'LactationStatus'] = 'No'

# #****************************************************
df['StillbirthCount'] = df['StillbirthCount'].replace({26: pd.NA,
                               14 : pd.NA,
                               })

df['MenopauseCause'] = df['MenopauseCause'].replace({

'after chemotherapy':'Chemotherapy',
'surgery' : 'Hysterectomy'
})



df.loc[(df['GestationCount'] != 0) & (df['GestationCount'].notnull())& (df['MariageStatus'].isnull()), 'MariageStatus'] = 'Married'
df.loc[(df['AbortionCount'] != 0) & (df['AbortionCount'].notnull())& (df['MariageStatus'].isnull()), 'MariageStatus'] = 'Married'
df.loc[(df['CesareanCount'] != 0) & (df['CesareanCount'].notnull())& (df['MariageStatus'].isnull()), 'MariageStatus'] = 'Married'
df.loc[(df['LastGestationAge'].notnull()) | (df['FirstGestationAge'].notnull())& (df['MariageStatus'].isnull()), 'MariageStatus'] = 'Married'


columns_to_check = ['GestationCount', 'AbortionCount', 'StillbirthCount', 
                    'FirstGestationAge', 'LastGestationAge', 
                     'CesareanCount']

df.loc[(df[columns_to_check].isnull().all(axis=1)&df['MariageStatus'].isnull()) & (df['Sex'] != 'Men'), 'MariageStatus'] = 'Single'


df.loc[(df['LoctationDurationTime'] != 0) &  (df['MariageStatus'].isin(['Married', 'Divorced/widowed (DW)']))&(df['GestationCount'].notnull()), 'LactationStatus'] = 'Yes'


# Pre-processing steps for Mid-Wifery Information
#GestationAge = the age of a pregnancy
df.loc[(df['MariageStatus'] == 'Single'), 'FirstGestationAge'] =  np.nan
df.loc[(df['MariageStatus'] == 'Single'), 'LastGestationAge'] = np.nan
df.loc[(df['MariageStatus'] == 'Single'), 'LactationStatus'] =  'No'
df.loc[(df['MariageStatus'] == 'Single'), 'LoctationDurationTime'] =  0
df.loc[(df['MariageStatus'] == 'Single'), 'CesareanStatus']  =  'No'
df.loc[(df['MariageStatus'] == 'Single'), 'CesareanCount']   =   0
df.loc[(df['MariageStatus'] == 'Single'), 'StillbirthCount'] =   0
df.loc[(df['MariageStatus'] == 'Single'), 'GestationCount'] =   0
df.loc[(df['MariageStatus'] == 'Single'), 'AbortionCount'] =   0

def calculate_pregnancies(row):
    # Calculate the sum, setting nulls to 0 where necessary
    gestation = row['GestationCount'] if pd.notnull(row['GestationCount']) else 0
    abortion = row['AbortionCount'] if pd.notnull(row['AbortionCount']) else 0
    stillbirth = row['StillbirthCount'] if pd.notnull(row['StillbirthCount']) else 0

    # Check if all the relevant columns are null
    if row[['GestationCount', 'AbortionCount', 'StillbirthCount']].isnull().all():
        return np.nan
    else:
        return gestation + abortion + stillbirth

df['Number of Pregnancies'] = df.apply(calculate_pregnancies, axis=1)

df.loc[(df['MariageStatus'] == 'Single'), 'Parity']   =   'Nulliparous'
df.loc[(df['Number of Pregnancies'] == 0)&(df['MariageStatus'].isin(['Married', 'Divorced/widowed (DW)'])), 'Parity']   =   'Nulliparous'


columns_to_check = [
    'FirstGestationAge', 'LastGestationAge', 
    'LactationStatus', 'CesareanCount', 
    'Number of Pregnancies'
]

df.loc[(df[columns_to_check].notna().any(axis=1)) & 
       (df['MariageStatus'].isin(['Married', 'Divorced/widowed (DW)'])), 
       'Parity'] = 'parous'



G_OCPUseLastDate = convert_persian_to_gregorian(df['OCPUseLastDate'])
df['G_OCPUseLastDate'] = pd.to_datetime(G_OCPUseLastDate)

df = df.drop(columns=['OCPUseLastDate'])
columns_ = df.columns.tolist()

def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]


df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)


df.rename(columns={
'FirstPeriodAge': 'Age at Menarche', 
'GestationCount': 'Gestation Count', 
'AbortionCount': 'Abortion Count', 
'StillbirthCount': 'Stillbirth Count', 
'FirstGestationAge': 'Age at First Child', 
'LastGestationAge': 'Age at Last Child', 
'LactationStatus': 'Breast-feeding Status', 
'LoctationDurationTime': 'Breast-feeding Duration',
'CesareanStatus': 'Cesarean Status', 
'CesareanCount': 'Cesarean Count', 
'IsRegularPerion': 'Menstrual regularity', 
'MenopauseStatus': 'Menopause Status', 
'MenopauseAge': 'Age at Menopause',
'MenopauseCause': 'Menopause Cause', 
'UsingOCP': 'Using OCP', 
'OCPDurationTime': 'OCP Duration',
'G_OCPUseLastDate' : 'OCP Use Last Date',
'MariageStatus' : 'Marital Status',
}, inplace=True)



df_Birthday = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DE.xlsx', usecols=['DocumentCode','Birth Day'])
df_Dates = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/BiosurDates.xlsx', usecols=['DocumentCode','First Biopsi Date','First Surgery Date'])

df_ = pd.merge(df_Birthday, df_Dates,  on='DocumentCode', how='outer')
df = pd.merge(df, df_,  on='DocumentCode', how='outer')
df['First Biopsi Date'] = pd.to_datetime(df['First Biopsi Date'], dayfirst=True, errors='coerce')
df['First Surgery Date'] = pd.to_datetime(df['First Surgery Date'], dayfirst=True, errors='coerce')
df['Birth Day'] = pd.to_datetime(df['Birth Day'], dayfirst=True, errors='coerce')
df['Age at Diagnosis'] = df.apply(
    lambda row: (
        round(((row['First Biopsi Date'] if pd.notna(row['First Biopsi Date']) else row['First Surgery Date']) - row['Birth Day']).days / 365.25, 1)
        if pd.notna(row['Birth Day']) and (pd.notna(row['First Biopsi Date']) or pd.notna(row['First Surgery Date'])) else pd.NA
    ),
    axis=1
)

df.loc[(df['Sex'] == 'Men') & (df['Age at Menarche'].notnull()), 'Sex'] = 'Female'

df['Invalid Menopause Age'] = df['Age at Diagnosis']-df['Age at Menopause']

df.loc[(df['Invalid Menopause Age']<=0) & (df['Menopause Status'] == 'Post-Menopause'), 'Menopause Status'] = np.nan
df.loc[(df['Invalid Menopause Age']<=2) & (df['Menopause Cause'] == 'Chemotherapy'), 'Menopause Status'] = np.nan

df['Invalid Age at Last Child'] = df['Age at Diagnosis']-df['Age at Last Child']
df.loc[(df['Invalid Age at Last Child']<=0) , 'Age at Last Child'] = np.nan

df['Invalid Age at First Child'] = df['Age at Diagnosis']-df['Age at First Child']
df.loc[df['Invalid Age at First Child']<=0 , 'Age at First Child'] = np.nan

df['Invalid Days Since Last OCP Use'] = df['First Biopsi Date']-df['OCP Use Last Date']
df.loc[(df['Invalid Days Since Last OCP Use'].dt.days <= 0), 'OCP Use Last Date'] = np.nan
df.shape
df = df.drop(columns=['Invalid Menopause Age','Invalid Age at Last Child','Invalid Age at First Child',
                      'Invalid Days Since Last OCP Use','First Biopsi Date','Age at Diagnosis',
                      'First Surgery Date','Birth Day','Sex'])
df['Number of Pregnancies'].value_counts(dropna=False)
df['Marital Status'].value_counts()
df.columns.tolist()
df.loc[(df['Breast-feeding Duration'] == 0) & (df['Breast-feeding Status'].isnull())& (df['Gestation Count'].isnull()), 'Breast-feeding Duration'] = np.nan
df.loc[(df['Breast-feeding Duration'].notnull()&(df['Number of Pregnancies']==0)), 'Number of Pregnancies'] =  np.nan
df.loc[(df['Age at First Child'].notnull()&(df['Number of Pregnancies']==0)), 'Number of Pregnancies'] =  np.nan
df.loc[(df['Age at Last Child'].notnull()&(df['Number of Pregnancies']==0)), 'Number of Pregnancies'] =  np.nan

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/MI.xlsx', index=False)

df.shape

