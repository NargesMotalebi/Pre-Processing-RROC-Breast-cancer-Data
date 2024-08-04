
from persiantools.jdatetime import JalaliDate
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import csv

# Read the Excel file into a DataFrame 
df_Surgery1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Surgery.xlsx')
df_Surgery2 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Surgery2.xlsx')
combined_df_Surgery = pd.concat([df_Surgery1, df_Surgery2], ignore_index=True)  # Combine rows
df_Surgery3 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Surgery3.xlsx')
df_Surgery3['Organ'] = 'BREAST C50'
combined_df_Surgery1 = pd.concat([df_Surgery3, combined_df_Surgery], ignore_index=True)  # Combine rows
df_Surgery4 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Surgery1397.xlsx')
df_Surgery4['Organ'] = 'BREAST C50'
combined_df_Surgery2 = pd.concat([df_Surgery4, combined_df_Surgery1], ignore_index=True)  # Combine rows
df_Surgery5 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Surgery1401.xlsx')
df_Surgery5['Organ'] = 'BREAST C50'
combined_df_Surgery3 = pd.concat([df_Surgery5, combined_df_Surgery2], ignore_index=True)  # Combine rows

df = combined_df_Surgery3

df.shape   

df.drop(columns=['Morphology','TopographyName','BirthDate','PatientId','Staging'],axis=1, inplace=True)
df.columns.tolist()

# Specify the value to keep
specified_value = 'BREAST C50'
# Filter the DataFrame to keep only rows related to the specified value in column 'Organ'
filtered_df = df[df['Organ'] == specified_value]
df = filtered_df
  
df = df.dropna(axis = 0, how ='all')  
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df = df.drop(df.columns[to_del], axis=1)
# Define the list of columns to consider
columns_ = df.columns.tolist()
# Define a mapping dictionary for Persian to English
persian_to_english = {
    'خیر': 'NO',
    'بلی': 'YES',
    'مرد': 'Men',
    'زن': 'Female',
}
# Replace Persian names with English names
df[columns_] = df[columns_].replace(persian_to_english)

############################################Surgery Information Edition####################################
#set the values in the 'AxillyDissection' column to 'Yes' where the 'SurgeryType' 
#column equals 'Breast Conserving Surgery+Axillar Disection'
df.loc[df['SurgeryType'] == 'Breast Conserving Surgery+Axillar Disection', 'AxillyDissection'] = 'YES'
df.loc[((df['PickingNodeCount'] > 4)), 'AxillyDissection'] = 'YES'
df.loc[((df['AxillyDissection'] == 'YES' & df['SLNBX'] == np.nan)), 'SLNBX'] = 'No'

df.columns.tolist()
# For each group, keep the row with the smallest number of missing values
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)
df.shape

df.loc[:, 'SurgeryType'].replace(
    {
    "Breast Conserving Surgery+Axillar Disection": 'Lumpectomy',
    "Modified Radical Mastectomy+axillary dissection": 'Mastectomy',
    'Breast Conserving Surgery': 'Lumpectomy',
    'Tumor Resection': 'Lumpectomy'}, inplace=True)

df.loc[:, 'NValue'].replace(
   {
    '1a' : 1, 
    '1C' : 1,
    '2a' : 2,
    '3a' : 3,
    '0C' : 0,
    '2c' : 2,
    '2c' : 2,
    '1b' : 1,
    'x' : np.nan
    }, inplace=True)


df.loc[:, 'MValue'].replace({
    'x' : np.nan, 
   }, inplace=True)

df.loc[:, 'TValue'].replace(
   {
    '1a' : 1, 
    '1C' : 1,
    '2a' : 2,
    '2c' : 2,
    '4a' : 4,
    '4b' : 4,
    '4d' : 4,
    'x'  : np.nan,
    '1b' : 1
    }, inplace=True)

df.loc[:, 'OrganDirection'].replace(
   {
    
    '--'  : np.nan,
    
    }, inplace=True)
df.loc[:, 'ClosestDistance'].replace(
   {
    
    're3'  : np.nan,
    
    }, inplace=True)

df.loc[:, 'MValue'].replace(
   {
    
    'a'  : np.nan,
    
    }, inplace=True)


#Remove Surgerytype which are not Lumpectomy and Mastectomy
df = df[~df['SurgeryType'].isin(['lymphadenopathy', 'mandibulectomy','bilobectomy','Inoperable'])]
#df = df[~df['SurgeryType'].isna()]


df.shape
# df[['FirstTumorSizes', 'SecondTumorSizes','ThirdTumorSizes','ForthTumorSizes']]
df_TumorsizeInfo = df['TumorSizes'].str.split(',', expand=True)
# Assign column names to the new DataFrame
df_TumorsizeInfo.columns = ['FirstTumorSizes', 'SecondTumorSizes','ThirdTumorSizes','ForthTumorSizes','FifthTumorSizes']
df_TumorsizeInfo['TumorSizes'] = df['TumorSizes']
def split_and_convert(x):
  if not isinstance(x, str) or ':' not in x:
    return x  # Wrap the single value in a list
  else:
    return max([float(num) for num in x.split(':')])

# Apply the function using .loc[row_indexer, col_indexer] = value
df_TumorsizeInfo['FirstTumorSizes'] = df_TumorsizeInfo['FirstTumorSizes'].apply(split_and_convert)
df_TumorsizeInfo['SecondTumorSizes'] = df_TumorsizeInfo['SecondTumorSizes'].apply(split_and_convert)
df_TumorsizeInfo['ThirdTumorSizes'] = df_TumorsizeInfo['ThirdTumorSizes'].apply(split_and_convert)

def count_values(x):
    if pd.isna(x):  # Handle NaN values explicitly
      return np.nan
    elif not isinstance(x, str):  # Handle non-string values
      return np.nan
    else:
      return len(x.split(','))

    return series.apply(count_values)

df['NumberofTumors'] = df['TumorSizes'].apply(count_values)

# Define a function to calculate the sum value while handling NaN
def calculate_max_value(x):
    return x.max() if not pd.isnull(x).all() else np.nan

# Apply the function to each row and calculate the maximum non-NaN value, replacing nulls with NaN
df['MaxTumorSize'] = df_TumorsizeInfo[['FirstTumorSizes', 'SecondTumorSizes', 'ThirdTumorSizes']].apply(calculate_max_value, axis=1)


#df = df.drop(columns=['TumorSizes','InvadeOrgans','Inferior','Staging'])

df = df.dropna(axis=1, how='all') 


df['LNR'] = df['InvolvedNodeCount'] / df['PickingNodeCount']
#df['Column1'].div(df['Column2'])


#****************************TStaging acording to TreatmentProtocol****************
def get_t_value(tumor_size):
    """
    Determine the T value of breast cancer based on tumor size.
    
    Parameters:
    tumor_size (float): Size of the tumor in centimeters.
    
    Returns:
    str: T value category (T1, T2, T3, T4).
    """
    
    # if tumor_size is None:
    #    return None  
    if tumor_size  <= 2.0:
        return 1
    elif 2.0 < tumor_size <= 5.0:
        return 2
    elif 5.0 < tumor_size:
        return 3
    # elif 11.0 <= tumor_size:
    #     return 4

def get_n_value(num_positive_nodes):
    """
    Determine the N value of breast cancer based on the number of positive lymph nodes.
    
    Parameters:
    num_positive_nodes (int): Number of positive lymph nodes.
    
    Returns:
    str: N value category (N0, N1, N2, N3).
    """
    # if num_positive_nodes is None:
    #     return None  
    if num_positive_nodes == 0:
        return 0  # No positive lymph nodes
    elif 1 <= num_positive_nodes <= 3:
        return 1  # 1-3 positive lymph nodes
    elif 4 <= num_positive_nodes <= 9:
        return 2  # 4-9 positive lymph nodes
    elif 10 <= num_positive_nodes :
        return 3  # 10 or more positive lymph nodes


# def get_m_value(has_metastasis):
#     """
#     Determine the M value of breast cancer based on metastasis status.
    
#     Parameters:
#     has_metastasis (bool): True if distant metastasis is present, False otherwise.
    
#     Returns:
#     str: M value category (M0 or M1).
#     """
#     if has_metastasis == 'Yes':
#         return 1  # Distant metastasis present
#     else:
#         return 0  # No distant metastasis
def apply_operations(row):

    # if row['Treatmentprotocol'] == 'Adjuvant':
    row['TValue'] = get_t_value(row['MaxTumorSize'])
    row['NValue'] = get_n_value(row['InvolvedNodeCount'])

    return row
df = df.apply(apply_operations, axis=1)

df.loc[:, 'SurgeryDate'].replace(
   {
    
    '//'  : np.nan,
    
    }, inplace=True)
# selects rows where Organ is not missing 
#then specifies the column to modify ('Tvalue').
df.loc[~df['InvadeOrgans'].isna(), 'TValue'] = 4

#Deleted Columns
df.drop(columns=['PatologyNumber','Lab'], axis=1,inplace=True)

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/SU.xlsx', index=False)

