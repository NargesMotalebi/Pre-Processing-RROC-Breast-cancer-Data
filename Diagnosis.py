from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import os
import re


# Read the Excel file into a DataFrame 
df_Diagnosis1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis.xlsx')
df_Diagnosis2 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis2.xlsx')
combined_df_DI = pd.concat([df_Diagnosis1, df_Diagnosis2], ignore_index=True) 
# Words to filter by
words_to_filter = ['C50'               
                   ]
# # Combine the words into a single regex pattern
pattern = '|'.join(words_to_filter)
# Use the query method to filter the DataFrame
filtered_combined_df_DI = combined_df_DI.query("Topography.str.contains(@pattern, case=False, na=False) or Topography.isna()")
# Use the str.contains() method to filter the DataFrame
# filtered_combined_df_DI = combined_df_DI[combined_df_DI['Topography'].str.contains(pattern, case=False, na=False)]
combined_df_DI = filtered_combined_df_DI
# Prefix to look for
prefix = 'Column'
# Get the list of columns to drop
columns_to_drop = filtered_combined_df_DI.filter(like=prefix, axis=1).columns
# Drop the columns
combined_df_DI.drop(columns=columns_to_drop, inplace=True)
df_Diagnosis3 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis3.xlsx')
df_Diagnosis3['Topography'] = 'C50.9'
combined_df_DI1 = pd.concat([df_Diagnosis3, combined_df_DI], ignore_index=True)
df_Diagnosis4 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis1399.xlsx')
df_Diagnosis4['TreatmentStartDate'] ='1399/01/01'
df_Diagnosis4['Topography'] = 'C50.9'
combined_df_DI2 = pd.concat([combined_df_DI1, df_Diagnosis4], ignore_index=True)
df_Diagnosis5 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis1400.xlsx')
df_Diagnosis5['TreatmentStartDate'] ='1400/01/01'
combined_df_DI3 = pd.concat([df_Diagnosis5, combined_df_DI2], ignore_index=True)
df_Diagnosis6 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis1397.xlsx')
df_Diagnosis6['Topography'] = 'C50.9'
df_Diagnosis6['TreatmentStartDate'] ='1397/01/01'
combined_df_DI4 = pd.concat([df_Diagnosis6, combined_df_DI3], ignore_index=True)
df_Diagnosis7 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Diagnosis1401.xlsx')
df_Diagnosis7['TreatmentStartDate'] ='1401/01/01'
df_Diagnosis7['Topography'] = 'C50.9'
combined_df_DI5 = pd.concat([df_Diagnosis7, combined_df_DI4], ignore_index=True)
df = combined_df_DI5
df.shape

#Remove theses column
columns_to_remove = ['Morphology','PatientId','TopographyName','SamplingMethod','BirthState','BirthCity','HabitationState', 'DocumentDate',
                     'Job', 'FatherName','HabitationCity','PassportNo','TumorSections','Gleason2','Gleason1','BiopsiOrgan','Race','MetastasisName']
df = df.drop(columns=columns_to_remove)

df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')
df.columns.tolist()
print(df.shape)
df['Sex'].isnull().sum()

persian_to_english = {
    'خیر': 'NO',
    'بلی': 'YES',
    'مرد': 'Men',
    'زن': 'Female',
    'ديپلم': 'Diploma',
    'ابتدايي': 'Illiterate',
    'كارشناسي': 'Bachelor',
    'كارداني': 'Diploma',
    'نهضت': 'Illiterate',
    'راهنمايي': 'Illiterate',
    'بيسواد': 'Illiterate',
    'كارشناسي ارشد': 'Master',
    'سواد قرآني': 'Illiterate',
    'زيرديپلم': 'Illiterate',
    'دكتري': 'Doctorate',
    'متاهل': 'Married',
    'مطلقه': 'Divorced (Female)',
    'همسرفوت شده': 'Widow/Widower',
    'مجرد': 'Single',
    'بيكار': 'Unemployed',
    'شاغل': 'Employed',
    'بازنشسته': 'Retired',
    'از كار افتاده': 'Laid off',
    'محصل': 'Student',
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
df.loc[:, 'BiopsiOrganDirection'].replace(
   {
    
    '--'  : np.nan,
    
    }, inplace=True)
df.loc[:, 'Education'].replace(
   {
    
    'مسلمان'  : np.nan,
    'يهودي'   : np.nan
    
    }, inplace=True)
# df.loc[:, 'Race'].replace(
#    {
    
#     'مسيحي'  : np.nan,
    
#     }, inplace=True)

#df['InsituPercentType'] = df['InsituPercent'].apply(lambda x: 'less' if x.startswith('<') else 'more' if x.startswith('>'))

# def replace_less_than(value):
#     match = re.match(r'<(\d+)', value)
#     return int(match.group(1)) if match else value

# df['InsituPercent'] = df['InsituPercent'].apply(replace_less_than)


df.loc[:, 'InsituPercent'].replace(
   {
    
    '<5'  : 5,
    '<10' : 10,
    '<15' : 15,
    '<25' : 25,
    
    }, inplace=True)


farsi_columns = df.columns.tolist()

# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)
df.shape
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)
# Define the list of columns to consider
columns_ = df.columns.tolist()
# Group by 'DocumentCode', sort=False to maintain original order
# For each group, keep the row with the smallest number of missing values
# def keep_row_with_smallest_missing_values(group):
#     if len(group) == 1:
#         return group
#     else:
#         missing_values_counts = group[columns_].isnull().sum(axis=1)
#         min_missing_values_index = missing_values_counts.idxmin()
#         return group.loc[[min_missing_values_index]]

# df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

# df.shape

# df_metastasicnames = pd.DataFrame()
#  #split the Name column into two columns using pd.Series.str.split()
# df[['FristMetastasisName', 'SecondMetastasisName','ThirdMetastasisName']] = df['MetastasisName'].str.split('-', expand=True)
# df.drop(columns=['MetastasisName'], inplace=True)

# df['SecondMetastasisName'].replace({" Metastatic ": " Metastatic"}, inplace=True)


#Define the mapping of terms to replacements
df['MorphologyName'].replace( {
    "Carcinoma, NOS": "ductal",
    'Carcinoma, anaplastic, NOS' : "ductal",#FFFFFFFFFFFFF
    "Adenocarcinoma, NOS": "ductal",
    "Paget disease, mammary (C50.)" : "ductal",
    "Paget disease, mammary (C50._)" : "ductal",
    "Inflammatory carcinoma (C50._)": "ductal",
    "Comedocarcinoma, NOS (C50._)": "ductal",
    "Sarcoma, NOS" : "ductal",
    "Infiltrating duct carcinoma (C50._)":"ductal",
    "Comedocarcinoma, noninfiltrating (C50._)" : "ductal",
    "Infiltrating duct and lobular carcinoma (C50._)":"ductal",
    "Intraductal papillary adenocarcinoma with invasion (C50._)" : "ductal",
    "Infiltrating duct mixed with other types of  carcinoma (C50._)" : "ductal",
    "Lobular carcinoma, NOS (C50._)": "lobular",
    "Lobular carcinoma in situ (C50._)" : "lobular",
    "Infiltrating lobular mixed with other types of  carcinoma (C50._)" : "lobular",
    "Noninfiltrating intraductal papillary adenocarcinoma (C50._)" :"Insitu",
    "Intraductal carcinoma, noninfiltrating, NOS": "Insitu",
    "Intraductal micropapillary carcinoma (C50._)": "Insitu",
    "Intraductal carcinoma and lobular carcinoma in situ (C50._)": "Insitu",
    "Neuroendocrine carcinoma, NOS" : "Insitu",#FFFFFFFFFFFFFF
    "Intraductal papilloma"   : "Insitu",  
    "Cribriform carcinoma in situ (C50._)" : "Insitu",#Ask it
    'Cribriform carcinoma' : "Insitu",#FFFFFFFFFFF
    "Medullary carcinoma, NOS": "Mix",
    "Mucinous adenocarcinoma" : "Mix",
    "Papillary carcinoma, NOS": "Mix",
    'Papillary carcinoma in situ': "Mix",#FFFFFFFFFFFFFF
    "Adenosquamous carcinoma" : "Mix",
    "Tubular adenocarcinoma"  : "Mix",
    "Atypical medullary carcinoma (C50._)": "Mix",
    "Adenocarcinoma with spindle cell metaplasia" : "Mix",
    "Hemangiosarcoma" : "Other",
    "Carcinomatosis"  : "Other",
    'Carcinosarcoma, NOS' : "Other", #Ask it
    "Metaplastic carcinoma, NOS"         : "Other",
    "Mucin-producing adenocarcinoma"     : "Other",
    "Phyllodes tumor, malignant (C50.)"  : "Other",
    "Phyllodes tumor, malignant (C50._)" : "Other",
    "Phyllodes tumor, malignant (C50._)" : "Other", 
    "Phyllodes tumor, borderline (C50._)" : "Other",
    "Squamous cell carcinoma, NOS"        :  "Other",
    'Apocrine adenocarcinoma': 'Other',#Ask It
    'Adenoid cystic carcinoma': 'Other',#Ask It
    'Adenocarcinoma with neuroendocrine differentiation': 'Mix',#Ask It
    'Neoplasm, malignant' : 'ductal',#Ask It
    'Liposarcoma, NOS' : 'Other',#Ask It
    'Stromal sarcoma, NOS' : 'Other'#Ask It
    
}, inplace=True)


df = df[~df['MorphologyName'].isin(['Osteosarcoma, NOS (C40._, C41._)',"Papillary cystadenocarcinoma, NOS (C56.9)",
                                    "Spindle cell sarcoma", "Adenoma, NOS","Papillary carcinoma, encapsulated (C73.9)",
                                    "Malignant lymphoma, non-Hodgkin, NOS",'Endometrioid adenocarcinoma, NOS','Lymphoid leukemia, NOS',
                                    'Plasmacytoma, NOS','Basaloid carcinoma','Renal cell carcinoma, NOS (C64.9)', 'Fibrous meningioma',
                                    'Alveolar rhabdomyosarcoma',"Chondrosarcoma, NOS (C40._, C41._), Basal cell carcinoma, NOS (C44._)",
                                    'Chondrosarcoma, NOS (C40._, C41._)','Basal cell carcinoma, NOS (C44._)'
                                    ])]


df.shape


df['MariageStatus'].replace({"مسلمان": np.nan}, inplace=True)
df['Education'].replace({"زرتشتي": np.nan}, inplace=True)

df.shape

# # Combine the words into a single regex pattern
pattern = '|'.join(words_to_filter)

# Use the query method to filter the DataFrame
filtered_df = df.query("Topography.str.contains(@pattern, case=False, na=False) or Topography.isna()")

df = filtered_df
# Prefix to look for
prefix = 'Column'
# Get the list of columns to drop
columns_to_drop = df.filter(like=prefix, axis=1).columns
# Drop the columns
df.drop(columns=columns_to_drop, inplace=True)
df.shape


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


df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/DI.xlsx', index=False)



