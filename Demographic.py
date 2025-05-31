from persiantools.jdatetime import JalaliDate
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import os
import re

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

df =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis-allyears.xlsx', usecols=['DocumentCode', 'DocumentDate','Sex',
                                                                                          'Education','JobStatus','IsRural',
                                                                                          'BirthCountry',                                                   
                                                                                          'BirthDate'])

#,'MariageStatus'
df_Diagnosis=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis.xlsx',  usecols=['DocumentCode', 'Topography','Education', 'JobStatus','BirthDate', 'IsRural','BirthCountry'])
words_to_filter = ['C50']
pattern = '|'.join(words_to_filter)
filtered_df = df_Diagnosis[df_Diagnosis['Topography'].str.contains(pattern, case=False, na=True) | df_Diagnosis['Topography'].isna()]
df_Diagnosis = filtered_df
df_mariage2=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis2.xlsx', usecols=['DocumentCode','Topography','Education', 'JobStatus','Sex','DocumentDate','IsRural','BirthDate','BirthCountry'])
pattern = '|'.join(words_to_filter)
filtered_df = df_mariage2[df_mariage2['Topography'].str.contains(pattern, case=False, na=True) | df_mariage2['Topography'].isna()]
df_mariage2 = filtered_df

df_mariage3=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis3.xlsx', usecols=['DocumentCode','Education', 'JobStatus'])
df_mariage1397=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1397.xlsx',  usecols=['DocumentCode',  'Education', 'JobStatus','BirthDate', 'Sex', 'IsRural'])
df_mariage1399=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1399.xlsx',  usecols=['DocumentCode',  'Education', 'JobStatus','BirthDate'])
df_mariage1400=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1400.xlsx',  usecols=['DocumentCode', 'Education', 'JobStatus', 'BirthDate'])
df_mariage1401= pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1401.xlsx',  usecols=['DocumentCode',  'Education', 'JobStatus', 'BirthDate'])


df_mariage = pd.concat([df_Diagnosis,df_mariage2,df_mariage3, df_mariage1397,df_mariage1399,df_mariage1400,df_mariage1401])


commonlistofcolumns = ['Education', 'JobStatus','BirthCountry','BirthDate','Sex', 'IsRural','BirthCountry','DocumentDate']
df = pd.merge(df, df_mariage,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')


persian_to_english = {
    'خیر': 'No',
    'بلی': 'Yes',
    'مرد': 'Men',
    'زن': 'Female',
    'ديپلم': 'Diploma',
    'ابتدايي': 'Below-Diploma',
    'كارشناسي': 'Bachelor',
    'كارداني': 'Diploma',
    'نهضت': 'Below-Diploma',
    'راهنمايي': 'Below-Diploma',
    'بيسواد': 'Below-Diploma',
    'كارشناسي ارشد': 'Master',
    'سواد قرآني': 'Below-Diploma',
    'زيرديپلم': 'Below-Diploma',
    'دكتري': 'Doctorate',
    'متاهل': 'Married',
    'مطلقه': 'Divorced/widowed (DW)',
    'همسرفوت شده': 'Divorced/widowed (DW)',
    'مجرد': 'Single',
    'بيكار': 'Unemployed',
    'شاغل': 'Employed',
    'بازنشسته': 'Retired',
    'از كار افتاده': 'Laid off',
    'محصل': 'Student',
    'كودك': 'Child',
    'فارس': 'Persian',
    'ترك': 'Turkish',
    'افغان': 'Afghan',
    'بلوچ': 'Baloch',
    'عرب': 'Arab',
    'كرد': 'Kurd',
    'تركمن': 'Turkmen',
    'ايران': 'Iran',
    'افغانستان': 'Afghanistan',
    'عراق': 'Iraq',
    'تركمنستان': 'Turkmenistan',
    'پاکستان': 'Pakistan',
    'یمن' : 'Yemen',
    'سوريه' : 'Syria',
    'آذربايجان' : 'Azerbaijan',
    'عمان' : 'Oman',
    'ازبكستان': 'Uzbekistan',
    'تاجيكستان': 'Tajikistan',
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
    'بيماري اوليه-بيماري اوليه': 'Primary disease - Primary disease'
}

farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)

df['Education'] = df['Education'].replace(
   {    
    'مسلمان'  : np.nan,
    'يهودي'   : np.nan,
    'زرتشتي': np.nan,
    'مسلمان': np.nan
     })

df['BirthDate'] = df['BirthDate'].replace({
    '1345/00/00' : '1345/01/01',
    '0//' : np.nan,
    '1975/12/15' : '1354/01/01'
})

counts = df.nunique()

to_del = [i for i,v in enumerate(counts) if v == 1]

df = df.drop(df.columns[to_del], axis=1)

df['MariageStatus'].replace({
    'مسلمان': np.nan}, inplace=True)

df = df[df['JobStatus'] != 'Child']
    

Grego_Date_BirthDay = convert_persian_to_gregorian(df['BirthDate'])
df['Gregorian_BirthDay'] = pd.to_datetime(Grego_Date_BirthDay)
df['Gregorian_BirthDay'] = df['Gregorian_BirthDay'].dt.strftime('%d/%m/%Y').fillna('')
Grego_Date_DocumentDate = convert_persian_to_gregorian(df['DocumentDate'])
df['Grego_Date_DocumentDate'] = pd.to_datetime(Grego_Date_DocumentDate)
df['Grego_Date_DocumentDate'] = df['Grego_Date_DocumentDate'].dt.strftime('%d/%m/%Y').fillna('')

columns_ = df.columns.tolist()
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]



df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

df.columns.tolist()

df = df.drop(columns=['BirthDate','DocumentDate','Topography'])
df.rename(columns={'Gregorian_BirthDay' : 'Birth Day',
                   'BirthCountry': 'Birth Country',
                    'IsRural':'Residency(Rural)' , 
                    'JobStatus':'Job Status', 
                    'Grego_Date_DocumentDate':'Document Date'}, inplace=True)


# df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/DE.xlsx', index=False)



df_TI_Mor = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/TI_Mor.xlsx') 
df = pd.merge(df, df_TI_Mor,  on='DocumentCode', how='outer',suffixes=('','_df'))
df_TI_Mor = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Biosidate.xlsx', usecols=['DocumentCode','BiopsiDate']) 
df = pd.merge(df, df_TI_Mor,  on='DocumentCode', how='outer',suffixes=('','_df'))

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/DE_.xlsx', index=False)