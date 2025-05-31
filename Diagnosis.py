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


df =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Biosidate.xlsx', usecols= ['DocumentCode','BiopsiOrgan', 'BiopsiOrganDirection', 'BiopsiDate',
                                                                                 'MorphologyName', 'Grade'])

df_Diagnosis=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis.xlsx',  usecols=['DocumentCode', 'Topography','BiopsiOrgan', 'MorphologyName','BiopsiOrganDirection','Grade','BiopsiDate'])
words_to_filter = ['C50']
pattern = '|'.join(words_to_filter)
filtered_df = df_Diagnosis[df_Diagnosis['Topography'].str.contains(pattern, case=False, na=True) | df_Diagnosis['Topography'].isna()]
df_Diagnosis = filtered_df
df_Diagnosis2=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis2.xlsx', usecols=['DocumentCode','Topography', 'MorphologyName','BiopsiOrganDirection','Grade','BiopsiDate'])
words_to_filter = ['C50']
pattern = '|'.join(words_to_filter)
filtered_df = df_Diagnosis2[df_Diagnosis['Topography'].str.contains(pattern, case=False, na=True) | df_Diagnosis2['Topography'].isna()]
df_Diagnosis2 = filtered_df
df_Diagnosis3=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis3.xlsx', usecols=['DocumentCode', 'MorphologyName', 'Grade'])
df_Diagnosis1397=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1397.xlsx',  usecols=['DocumentCode', 'MorphologyName','Grade'])
df_Diagnosis1399=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1399.xlsx',  usecols=['DocumentCode', 'MorphologyName', 'Grade'])
df_Diagnosis1400=  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1400.xlsx',  usecols=['DocumentCode', 'Grade'])
df_Diagnosis1401= pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1401.xlsx',  usecols=['DocumentCode',  'Grade'])



df_Diagnosis_all = pd.concat([df_Diagnosis,df_Diagnosis2, df_Diagnosis3,df_Diagnosis1397,df_Diagnosis1399,df_Diagnosis1400,df_Diagnosis1401])

df_Diagnosis_all.describe()

commonlistofcolumns = ['BiopsiOrgan', 'BiopsiOrganDirection','BiopsiDate','MorphologyName','Grade']
df = pd.merge(df, df_Diagnosis_all,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')
df.describe()

df_morphologyname =  pd.read_excel('G:/Breast Cancer/Reza Ancology/TI/Treatment-allyears.xlsx', usecols=['DocumentCode','MorphologyName'])


commonlistofcolumns = ['MorphologyName']
df = pd.merge(df, df_morphologyname,  on=['DocumentCode'], how='outer', suffixes=('','_morphologyname2') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_morphologyname2'])

df = df.drop(columns=[f'{col}_morphologyname2' for col in commonlistofcolumns])


df['BiopsiOrgan']= df['BiopsiOrgan'].replace({

'Breast, NOS' :                   'Breast',                    
'Skin of Breast' :                'Breast' , 
'Nipple'                  :       'Breast' ,
'Neck Lymph node'         :       'Breast' , 
# 'Axillary lymph node'     :   
'Lymph nodes of axilla or arm' :  'Axillary lymph node' ,            
'chest wall'              :       'Breast' ,            
'axillar'                 :       'Axillary lymph node' ,             
#'Lymph node of cervical'   :'Axillary lymph node',        
# Femural bone                       1
# Vertebral column                   1
'leg soft tissue' :'Leg soft tissue',
'Kidney, NOS' : 'Kidney',
'Vertebral column' : 'Vertebral column'   
})

df['MorphologyName']= df['MorphologyName'].replace({
    "Carcinoma, NOS": "Ductal",
    'Carcinoma, anaplastic, NOS' : 'Ductal',
    "Adenocarcinoma, NOS": 'Ductal',
    "Paget disease, mammary (C50.)" : 'Ductal',
    "Paget disease, mammary (C50._)" : 'Ductal',
    "Inflammatory carcinoma (C50._)": 'Ductal',
    "Comedocarcinoma, NOS (C50._)": 'Ductal',
    "Sarcoma, NOS" : 'Ductal',
    "Infiltrating duct carcinoma (C50._)":'Ductal',
    "Comedocarcinoma, noninfiltrating (C50._)" : 'Ductal',
    "Infiltrating duct and lobular carcinoma (C50._)":'Ductal',
    "Intraductal papillary adenocarcinoma with invasion (C50._)" : 'Ductal',
    "Infiltrating duct mixed with other types of  carcinoma (C50._)" : 'Ductal',
    'Lobular carcinoma, NOS (C50._)': "Lobular",
    "Lobular carcinoma, NOS (C50._)": "Lobular",
    "Lobular carcinoma in situ (C50._)" :  "Lobular",
    "Infiltrating lobular mixed with other types of  carcinoma (C50._)" :  "Lobular",
    "Noninfiltrating intraductal papillary adenocarcinoma (C50._)" :"Insitu",
    "Intraductal carcinoma, noninfiltrating, NOS": "Insitu",
    "Intraductal micropapillary carcinoma (C50._)": "Insitu",
    "Intraductal carcinoma and lobular carcinoma in situ (C50._)": "Insitu",
    "Neuroendocrine carcinoma, NOS" : "Insitu",
    "Intraductal papilloma"   : "Insitu",  
    "Cribriform carcinoma in situ (C50._)" : "Ductal",
    'Cribriform carcinoma' : "Insitu",
    "Medullary carcinoma, NOS": "Mix",
    "Mucinous adenocarcinoma" : "Mix",
    "Papillary carcinoma, NOS": "Mix",
    'Papillary carcinoma in situ': "Mix",
    "Adenosquamous carcinoma" : "Mix",
    "Tubular adenocarcinoma"  : "Mix",
    "Atypical medullary carcinoma (C50._)": "Mix",
    "Adenocarcinoma with spindle cell metaplasia" : "Mix",
    "Hemangiosarcoma" : "Other",
    "Carcinomatosis"  : "Other",
    'Carcinosarcoma, NOS' : "Other", 
    "Metaplastic carcinoma, NOS"         : "Other",
    "Mucin-producing adenocarcinoma"     : "Other",
    "Phyllodes tumor, malignant (C50.)"  : "Other",
    "Phyllodes tumor, malignant (C50._)" : "Other",
    "Phyllodes tumor, malignant (C50._)" : "Other", 
    "Phyllodes tumor, borderline (C50._)" : "Other",
    "Squamous cell carcinoma, NOS"        :  "Other",
    'Apocrine adenocarcinoma': 'Other',
    'Adenoid cystic carcinoma': 'Other',

    'Infiltrating duct carcinoma (C50._)-Infiltrating duct carcinoma (C50._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Adenocarcinoma, NOS' :'Ductal',
    'Infiltrating duct carcinoma (C50._)-': 'Ductal',
    'Infiltrating duct carcinoma (C50._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Hodgkin lymphoma, nodular lymphocyte predominance' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Lobular carcinoma, NOS (C50._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Papillary carcinoma, NOS' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)--' : 'Ductal',
    'Infiltrating duct and lobular carcinoma (C50._) -Infiltrating duct carcinoma (C50._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Adenoid cystic carcinoma' :' Ductal',
    'Infiltrating duct carcinoma (C50._)-Papillary carcinoma, follicular variant (C73.9)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Intraductal carcinoma, noninfiltrating, NOS' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Basal cell carcinoma, NOS (C44._)' :' Ductal',
    'Infiltrating duct carcinoma (C50._)-Osteosarcoma, NOS (C40._, C41._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Atypical meningioma' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Infiltrating duct carcinoma (C50._)-Infiltrating duct carcinoma (C50._)':'Ductal',
    'Infiltrating duct carcinoma (C50._)-Nodular melanoma (C44._)': 'Ductal',
    'Infiltrating duct mixed with other types of  carcinoma (C50._) -Infiltrating duct carcinoma (C50._)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Intraductal papilloma' :'Ductal', 
    'Infiltrating duct carcinoma (C50._)-Follicular adenoma (C73.9) -Renal cell carcinoma, NOS (C64.9)' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Mucinous adenocarcinoma' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Leiomyoma, NOS-Adenocarcinoma, NOS': 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Carcinoma, NOS' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Infiltrating duct carcinoma (C50._)-Intraductal carcinoma, noninfiltrating, NOS' :'Ductal',
    'Infiltrating duct carcinoma (C50._)-Gastrointestinal stromal tumor, NOS' : 'Ductal',
    'Infiltrating duct carcinoma (C50._)-Medullary carcinoma, NOS' : 'Ductal',
    'Medullary carcinoma, NOS  -Infiltrating duct carcinoma (C50._)': 'Ductal',
    '-Intraductal carcinoma, noninfiltrating, NOS':'Ductal',
    'Metaplastic carcinoma, NOS  -Infiltrating duct carcinoma (C50._)' :'Ductal',
    'Lobular carcinoma, NOS (C50._)  -Papillary serous cystadenocarcinoma (C56.9)' : "Lobular",
    'Lobular carcinoma, NOS (C50._)  -Tubular adenocarcinoma' : "Lobular",
    'Lobular carcinoma in situ (C50._)   -Infiltrating duct carcinoma (C50._)' : "Lobular",
    'Lobular carcinoma, NOS (C50._)  -Infiltrating duct carcinoma (C50._)' : "Lobular",
    'Lobular carcinoma, NOS (C50._)  -Comedocarcinoma, noninfiltrating (C50._)' : "Lobular",
    'Noninfiltrating intraductal papillary adenocarcinoma (C50._) -Cribriform carcinoma in situ (C50._)' : 'Mix',
    'Papillary carcinoma, NOS-Infiltrating duct carcinoma (C50._)' : 'Mix',
    'Spindle cell carcinoma, NOS' :'Other',
    '-Comedocarcinoma, noninfiltrating (C50._)':'Other',
    'Medullary carcinoma, NOS  -Liposarcoma, NOS':'Other',
    'Alveolar rhabdomyosarcoma-Alveolar rhabdomyosarcoma-Lymphoid leukemia, NOS' : 'Other',
    'Intraductal carcinoma, noninfiltrating, NOS-Papillary microcarcinoma (C73.9)':'Other',
    'Adenocarcinoma with neuroendocrine differentiation -Intraductal carcinoma, noninfiltrating, NOS' :'Other',
    '-Pseudosarcomatous carcinoma':'Other',
    'Pleomorphic liposarcoma':'Other',
    'Mucinous adenocarcinofibroma':'Other',
    'Carcinoma, NOS-Adenocarcinoma, NOS':'Other',
    'Myofibroblastoma':'Other',
    'Secretory carcinoma of the breast (C50._)':'Other',
    'Aggressive fibromatosis':'Other',
    'Tubular adenocarcinoma -Squamous cell carcinoma, NOS':'Other',
    'Adenocarcinoma with neuroendocrine differentiation'      : 'Other',
    '-'                                                       : np.nan,
    'Stromal sarcoma, NOS'                                : 'Other',
    'Adenoma, NOS'                                             : 'benign' 
})

df['BiopsiOrganDirection'].replace(
    {
    'bothSides' :'BothSides',
    'left' : 'Left',
    'right' : 'Right',
    '--': np.nan,
     }, inplace=True)

df = df[df['MorphologyName'].isin(['Ductal', 'Lobular', 'Mix', 'Insitu', 'Other'])]
df['BiopsiOrganDirection'].value_counts()
Grego_Date_Biopsi = convert_persian_to_gregorian(df['BiopsiDate'])
df['Gregorian_BiopsiDate'] = pd.to_datetime(Grego_Date_Biopsi)

df = df.drop(columns=['BiopsiDate'])

# Add an incremental index within each group to pivot data later
df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['Gregorian_BiopsiDate'])
# [,'NumberofTumors','MaxTumorSize','PickingNodeCount','InvolvedNodeCount']
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
df[['NearestBiopsiDate1', 'NearestBiopsiDate2', 'NearestBiopsiDate3']] = df[
    ['Gregorian_BiopsiDate_1','Gregorian_BiopsiDate_2','Gregorian_BiopsiDate_3',
     'Gregorian_BiopsiDate_4','Gregorian_BiopsiDate_5','Gregorian_BiopsiDate_6',
     'Gregorian_BiopsiDate_7','Gregorian_BiopsiDate_8','Gregorian_BiopsiDate_9',
     'Gregorian_BiopsiDate_10','Gregorian_BiopsiDate_11','Gregorian_BiopsiDate_12',
     'Gregorian_BiopsiDate_13','Gregorian_BiopsiDate_14']
].apply(get_three_earliest_dates, axis=1, result_type='expand')

df['NearestBiopsiDate1'] = pd.to_datetime(df['NearestBiopsiDate1'], dayfirst=True, errors='coerce')
df['NearestBiopsiDate2'] = pd.to_datetime(df['NearestBiopsiDate2'], dayfirst=True, errors='coerce')
df['NearestBiopsiDate3'] = pd.to_datetime(df['NearestBiopsiDate3'], dayfirst=True, errors='coerce')

# Format the dates as strings if they are not NaT
df['NearestBiopsiDate1'] = df['NearestBiopsiDate1'].dt.strftime('%d/%m/%Y').fillna(pd.NA)
df['NearestBiopsiDate2'] = df['NearestBiopsiDate2'].dt.strftime('%d/%m/%Y').fillna(pd.NA)
df['NearestBiopsiDate3'] = df['NearestBiopsiDate3'].dt.strftime('%d/%m/%Y').fillna(pd.NA)



columns_ = df.columns.tolist()

def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

df = df.drop(columns=['Gregorian_BiopsiDate','Gregorian_BiopsiDate_1','Gregorian_BiopsiDate_2','Gregorian_BiopsiDate_3','Gregorian_BiopsiDate_4',
                      'Gregorian_BiopsiDate_5','Gregorian_BiopsiDate_6','Gregorian_BiopsiDate_7','Gregorian_BiopsiDate_8',
                      'Gregorian_BiopsiDate_9', 'Gregorian_BiopsiDate_10','Gregorian_BiopsiDate_11','Gregorian_BiopsiDate_12',
                      'Gregorian_BiopsiDate_13','Gregorian_BiopsiDate_14','idx'])
df = df.drop(columns=['Gregorian_BiopsiDate_15', 'NearestBiopsiDate3',
'Gregorian_BiopsiDate_16', 'Gregorian_BiopsiDate_17', 'Gregorian_BiopsiDate_18', 
'Gregorian_BiopsiDate_19', 'Gregorian_BiopsiDate_20', 'Gregorian_BiopsiDate_21', 
'Gregorian_BiopsiDate_22', 'Gregorian_BiopsiDate_23', 'Gregorian_BiopsiDate_24', 
'Gregorian_BiopsiDate_25', 'Gregorian_BiopsiDate_26', 'Gregorian_BiopsiDate_27', 
'Gregorian_BiopsiDate_28', 'Gregorian_BiopsiDate_29', 'Gregorian_BiopsiDate_30', 
'Gregorian_BiopsiDate_31', 'Gregorian_BiopsiDate_32', 'Gregorian_BiopsiDate_33',
'Gregorian_BiopsiDate_34', 'Gregorian_BiopsiDate_35', 'Gregorian_BiopsiDate_36', 
'Gregorian_BiopsiDate_37', 'Gregorian_BiopsiDate_38', 'Gregorian_BiopsiDate_39', 
'Gregorian_BiopsiDate_40', 'Gregorian_BiopsiDate_41', 'Gregorian_BiopsiDate_42', 
'Gregorian_BiopsiDate_43', 'Gregorian_BiopsiDate_44', 'Gregorian_BiopsiDate_45', 
'Gregorian_BiopsiDate_46', 'Gregorian_BiopsiDate_47', 'Gregorian_BiopsiDate_48', 
'Gregorian_BiopsiDate_49', 'Gregorian_BiopsiDate_50', 'Gregorian_BiopsiDate_51', 
'Gregorian_BiopsiDate_52', 'Gregorian_BiopsiDate_53','Gregorian_BiopsiDate_54'])

#df['NearestBiopsiDate1'].isnull().sum()


df.rename(columns={'BiopsiOrgan' : 'Biopsi Organ' , 
'BiopsiOrganDirection':'Biopsi Organ Direction', 
'MorphologyName' : 'Morphology Name',
'NearestBiopsiDate1': 'First Biopsi Date',
'NearestBiopsiDate2': 'Second Biopsi Date',
}, inplace=True)

df.shape

pattern = '|'.join(words_to_filter)
filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=True) | df['Topography'].isna()]
df = filtered_df

df.drop(columns=['Topography'],inplace=True)

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/DI.xlsx', index=False)



# df.replace({
#     '-' : np.nan,
#     '-9731/3' : '9731/3',
#     '-8033/3' : '8033/3',
#     '-8500/3' : '8500/3',
#     '8500/3--': '8500/3',
#     '-8500/2' : '8500/2',
#     '1975/12/15' : '1358/09/17'
    

#     }, inplace=True)
# #df['InsituPercentType'] = df['InsituPercent'].apply(lambda x: 'less' if x.startswith('<') else 'more' if x.startswith('>'))

# # def replace_less_than(value):
# #     match = re.match(r'<(\d+)', value)
# #     return int(match.group(1)) if match else value

# # df['InsituPercent'] = df['InsituPercent'].apply(replace_less_than)

# # df['InsituPercent'].replace(
# #    {
    
# #     '<5'  : 5,
# #     '<10' : 10,
# #     '<15' : 15,
# #     '<25' : 25,
    
# #     }, inplace=True)
