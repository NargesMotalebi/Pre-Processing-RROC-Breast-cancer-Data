from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import csv


def remove_outliers_iqr(df, column_name, thresh=1.5):
  """
  Removes outliers from a DataFrame column based on IQR and replaces them with NaN.

  Args:
      df (pandas.DataFrame): The DataFrame containing the column.
      column_name (str): The name of the column to check for outliers.
      thresh (float, optional): The multiplier for IQR (default 1.5). Values exceeding
          thresh * IQR below Q1 or above Q3 are replaced with NaN.

  Returns:
      pandas.DataFrame: The DataFrame with outliers replaced by NaN.
  """

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



df =  pd.read_excel('G:/Breast Cancer/Reza Ancology/TI/Treatment-allyears.xlsx', usecols= ['DocumentCode','TreatmentProtocol','TotalDose','TreatmentOrgan',
                                                                                            'TreatmentStartDate', 'TreatmentType'])

df = df.replace({
    '-' : np.nan,
    'a' : np.nan
})
columns_ = df.columns.tolist()

# words_to_filter = ['C50']

# # Combine the words into a single regex pattern
# pattern = '|'.join(words_to_filter)

# # Use the str.contains() method to filter the DataFrame
# filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=False)]
# df = filtered_df

#df.drop(columns=['Topography'], inplace=True)

df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')


counts = df.nunique()


persian_to_english = {
   
    'راديوتراپي': 'Radiotherapy',
    'هورمون تراپي': 'Hormone Therapy',
    'شيمي درماني': 'Chemotherapy',
    'تارگت تراپي':'Targeted Therapy',
    'IMRT': 'Radiotherapy',#'Intensity-Modulated Radiation Therapy',
    'BREAST C50' :'Breast',
    'Breast, NOS C50.9':'Breast',
    'Brain, NOS C71.9' : 'Brain',
    'BRAIN C71' : 'Brain',
    'Bone, NOS C41.9' :'Bone',
    'Bones of skull and face and associated joints C41.0':'Bone',
    'Bone of limb, NOS C40.9':'Bone',
    'Pelvis, NOS C76.3' : 'Pelvis',
    'Vertebral column C41.2' : 'Vertebral column',
    'Pelvic bones, sacrum, coccyx and associated joints C41.4'      : 'Pelvis',
    'Thorax, NOS C76.1'  : 'Thorax',                                                         
    'Lung, NOS C34.9' :'Lung',                                                                      
    'Head, face or neck, NOS C76.0' :  'Head, face or neck'   , 
    'Orbit, NOS C69.6' : 'Orbit' ,                                                      
    'Rib, sternum, clavicle and associated joints C41.3' :'Rib, sternum, clavicle and associated joints',                                    
    'Mandible C41.1' : 'Mandible' ,                                                                        
    'Spinal cord C72.0'  : 'Spinal cord'  ,                                                                  
    'Connective, Subcutaneous and other soft tissues of upper limb and shoulder C49.1' : 'limb and shoulder' ,    
    'Axillary tail of breast C50.6' : 'Breast'   ,                                                      
    'Long bones of upper limb, scapula and associated joints C40.0'    : 'Bone'  ,                   
    'Lymph nodes of axilla or arm C77.3'    : 'Lymph nodes',                                                 
    'Long bones of lower limb and associated joints C40.2' : 'Bone' ,
    'Maxillary sinus C31.0'                 : 'Maxillary sinus',
    'Tongue, NOS C02.9'      : 'Tongue' ,
    'Lymph nodes of inguinal region or leg C77.4' : 'Lymph nodes of inguinal region',
    'Skin of scalp and neck C44.4' : 'Skin of scalp and neck',
    'Short bones of lower limb and associated joints C40.3' : 'Bone'

}


Grego_Date_Surgery = convert_persian_to_gregorian(df['TreatmentStartDate'])
df['Gregorian_TreatmentStartDate'] = pd.to_datetime(Grego_Date_Surgery)


farsi_columns = df.columns.tolist()
# Replace Persian names with English names
df[farsi_columns] = df[farsi_columns].replace(persian_to_english)

# Add an incremental index within each group to pivot data later
df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['Gregorian_TreatmentStartDate','TreatmentType','TreatmentOrgan','TotalDose'])

# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]

# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 


def get_Four_earliest_dates(row):
    # Drop any NaT values (missing dates), then sort and get unique dates
    dates = row.dropna().sort_values().unique()  # Ensure dates are unique

    # Return up to three dates, filling with NaT if there are fewer than three dates
    if len(dates) >= 4:
        return dates[0], dates[1], dates[2], dates[3]
    elif len(dates) == 3:
        return dates[0], dates[1], dates[2], pd.NaT
    elif len(dates) == 2:
        return dates[0], dates[1], pd.NaT, pd.NaT
    elif len(dates) == 1:
        return dates[0], pd.NaT, pd.NaT, pd.NaT
    
    else:
        return pd.NaT, pd.NaT, pd.NaT, pd.NaT

# Apply the function to each row and create two new columns
df[['NearestTreatmentStartDate1', 'NearestTreatmentStartDate2','NearestTreatmentStartDate3','NearestTreatmentStartDate4']] = df[['Gregorian_TreatmentStartDate_1','Gregorian_TreatmentStartDate_2',
                                                         'Gregorian_TreatmentStartDate_3','Gregorian_TreatmentStartDate_4','Gregorian_TreatmentStartDate_5']].apply(get_Four_earliest_dates, axis=1, result_type='expand')

# Format dates to 'dd/mm/yyyy' format
df['NearestTreatmentStartDate1'] = df['NearestTreatmentStartDate1'].dt.strftime('%d/%m/%Y')
df['NearestTreatmentStartDate2'] = df['NearestTreatmentStartDate2'].dt.strftime('%d/%m/%Y')
df['NearestTreatmentStartDate3'] = df['NearestTreatmentStartDate3'].dt.strftime('%d/%m/%Y')
df['NearestTreatmentStartDate4'] = df['NearestTreatmentStartDate4'].dt.strftime('%d/%m/%Y')


def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)



cols =['TreatmentType_1','TreatmentType_2', 'TreatmentType_3', 'TreatmentType_4', 'TreatmentType_5', 'TreatmentType_6', 'TreatmentType_7']

# Apply set to each row across the specified columns and expand the result into new columns
unique_values_df = df[cols].apply(lambda row: list(set(row)), axis=1)

# Now expand the lists in the 'unique_values_df' into separate columns
expanded_df = pd.DataFrame(unique_values_df.tolist(), index=df.index)

# Rename the columns to reflect the unique values
expanded_df.columns = [f'Unique_{i+1}' for i in range(expanded_df.shape[1])]

# Combine with original df if you want to keep other columns
df = pd.concat([df, expanded_df], axis=1)


df = df.drop(columns=['idx','TotalDose', 'TreatmentOrgan', 'TreatmentType','TreatmentOrgan_6', 'TreatmentOrgan_4',
                      'Gregorian_TreatmentStartDate_1','Gregorian_TreatmentStartDate','TreatmentStartDate','TreatmentOrgan_5',
                      'Gregorian_TreatmentStartDate_2', 'Gregorian_TreatmentStartDate_3', 'TotalDose_7', 'TreatmentOrgan_7',
                      'Gregorian_TreatmentStartDate_4', 'Gregorian_TreatmentStartDate_5', 'TotalDose_6', 'TreatmentType_5',
                      'Gregorian_TreatmentStartDate_6', 'Gregorian_TreatmentStartDate_7', 'TotalDose_5', 'TreatmentType_7',
                      'TreatmentType_6','TotalDose_3', 'TotalDose_4','TreatmentType_2', 'TreatmentType_1',
                      'TreatmentType_3', 'TreatmentType_4','TotalDose_1','TotalDose_2','Unique_4'
                      
                      ])
df.columns.tolist()



df.rename(columns={
'Unique_1': 'Type of First Treatment', 
'Unique_2': 'Type of Second Treatment',  
'Unique_3': 'Type of Third Treatment',  

'TreatmentOrgan_1': 'First Treatment Organ',  
'TreatmentOrgan_2': 'Second Treatment Organ', 
'TreatmentOrgan_3': 'Third Treatment Organ',  
# 'TotalDose_1': 'First Total Dose', 
# 'TotalDose_2': 'Second Total Dose', 
'NearestTreatmentStartDate1': 'First Treatment Date',  
'NearestTreatmentStartDate2': 'Second Treatment Date', 
'NearestTreatmentStartDate3': 'Third Treatment Date',  
'NearestTreatmentStartDate4': 'Forth Treatment Date', 
}, inplace=True)

df['Type of First Treatment'].value_counts(dropna=False )
df['Type of Second Treatment'].value_counts(dropna=False)

# df['Type of Second Treatment'].fillna(df['Type of Third Treatment'])
#df = df.drop(columns=['Type of Third Treatment'])
#df.columns.tolist()
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/TI.xlsx', index=False)

df['Second Treatment Organ'].value_counts()