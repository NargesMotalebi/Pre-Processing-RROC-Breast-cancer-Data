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


df_Surgery=  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery.xlsx', usecols=['DocumentCode', 'SurgeryDate','Organ', 'OrganDirection',
                                                                                      'LymphaticEmboli', 'SLNBX','SurgeryType', 'AxillyDissection', 'InvolvedMargin',
                                                                                       'PickingNodeCount', 'InvolvedNodeCount','Skin', 'LVI', 'PNI','TValue', 'NValue', 
                                                                                       'MValue','InvadeOrgans', 'TumorSizes'])

df_Surgery2=  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery2.xlsx',  usecols=['DocumentCode', 'SurgeryDate','Organ',
                                                                                          'OrganDirection','SurgeryType','LymphaticEmboli', 
                                                                                          'SLNBX', 'AxillyDissection', 'InvolvedMargin','Skin','PickingNodeCount',
                                                                                           'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 'MValue',
                                                                                           'InvadeOrgans', 'TumorSizes'])

commonlistofcolumns = ['SurgeryDate','Organ', 'OrganDirection',
                        'LymphaticEmboli', 'SLNBX','SurgeryType', 'AxillyDissection','Skin', 'InvolvedMargin',
                        'PickingNodeCount', 'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 
                        'MValue','InvadeOrgans', 'TumorSizes']
df2 = pd.merge(df_Surgery2, df_Surgery,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df2[col] = df2[col].combine_first(df2[f'{col}_olddf'])

df2 = df2.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df_Surgery3=  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery3.xlsx',  usecols=['DocumentCode','TValue', 'NValue','TumorSizes',
                                                                                                            'MValue','PickingNodeCount', 'InvolvedNodeCount', 
                                                                                                            'LVI', 'PNI', 'Skin','SurgeryType', 'SurgeryDate',
                                                                                                            'SLNBX', 'AxillyDissection','InvolvedMargin'])



commonlistofcolumns = ['SurgeryDate','SLNBX','SurgeryType', 'AxillyDissection', 'InvolvedMargin',
                        'PickingNodeCount', 'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 
                        'MValue','Skin','TumorSizes']
df2 = pd.merge(df2, df_Surgery3,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df2[col] = df2[col].combine_first(df2[f'{col}_olddf'])

df2 = df2.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df_Surgery1397=  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery1397.xlsx',  usecols=['DocumentCode', 'SurgeryDate','LymphaticEmboli', 
                                                                                          'SLNBX', 'AxillyDissection', 'InvolvedMargin','Skin','PickingNodeCount',
                                                                                           'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 'MValue',
                                                                                           'InvadeOrgans', 'TumorSizes'])

commonlistofcolumns = ['SurgeryDate','LymphaticEmboli', 'SLNBX', 'AxillyDissection','Skin', 'InvolvedMargin',
                        'PickingNodeCount', 'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue',
                        'MValue','InvadeOrgans', 'TumorSizes']
df2 = pd.merge(df2, df_Surgery1397,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df2[col] = df2[col].combine_first(df2[f'{col}_olddf'])

df2 = df2.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df_Surgery1401= pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery1401.xlsx',  usecols=['DocumentCode', 'SurgeryDate','LymphaticEmboli', 
                                                                                          'SLNBX', 'AxillyDissection', 'InvolvedMargin','Skin','PickingNodeCount',
                                                                                           'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 'MValue',
                                                                                           'InvadeOrgans', 'TumorSizes'])

df2 = pd.merge(df2, df_Surgery1401,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df2[col] = df2[col].combine_first(df2[f'{col}_olddf'])

df2 = df2.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

commonlistofcolumns = ['SurgeryDate','Organ', 'OrganDirection',
                        'LymphaticEmboli', 'SLNBX','SurgeryType', 'AxillyDissection','Skin', 'InvolvedMargin',
                        'PickingNodeCount', 'InvolvedNodeCount', 'LVI', 'PNI','TValue', 'NValue', 
                        'MValue','InvadeOrgans', 'TumorSizes']

df = pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery-allyears.xlsx')

df.drop(columns=['PatientId','Staging','Lab', 'Lateral',  'IsMetastasisSurgery', 'MorphologyName',
                 'Anterior', 'Superior', 'Posterior', 'Deep', 'Medial', 'Inferior',
                 'PatologyNumber', 'Radial','Leiomyoma', 'Fibroma'],axis=1, inplace=True)#'BirthDate'


df = pd.merge(df, df2,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])



df['Organ'] = df['Organ'].replace({
'BREAST C50' : 'Breast',
'Breast, NOS C50.9' : 'Breast' ,
'Lymph nodes of axilla or arm C77.3'        :            'Lymph nodes',
'Axillary tail of breast C50.6'     :                    'Breast' ,
'Thyroid gland C73.9'               :                    'Thyroid gland',
'Endometrium C54.1'                 :                    'Endometrium',
'KIDNEY C64'                        :                    'Kidney',
'Lymph nodes of head, face and neck C77.0'     :         'Lymph nodes',
'Sigmoid colon C18.7'                          :         'Sigmoid colon',
'COLON C18'                                :             'Colon',
'Head of pancreas C25.0'                   :             'Pancreas',
'Tongue, NOS C02.9'                        :             'Tongue',
'Lung, NOS C34.9'                          :             'Lung',
'THYROID GLAND C73'                        :             'Thyroid gland',
'Thorax, NOS C76.1'                        :             'Thorax',
'PAROTID GLANID C07'                       :             'Parotid glanid',
'Skin of other and unspecified parts of face C44.3'  :   'Skin of face',
'OVARY C56'                                          :   'Ovary',
'Ovary C56.9'                                        :   'Ovary' ,  
'Corpus uteri C54.9'                                 :   'Corpus uteri'  
})

df['Topography'] = df['Topography'].replace({
'C50.9--' : 'C50.9',
'C50.9-' : 'C50.9'
})  
df['Morphology'] = df['Morphology'].replace({
'8500/3--' : '8500/3' ,
'8500/3-' : '8500/3' ,
'-' :np.nan
})  

df['TopographyName'] = df['TopographyName'].replace({
'Breast, NOS--' : 'Breast',
'Breast, NOS-Breast, NOS' :  'Breast' ,
'Upper-outer quadrant of breast'   :   'Breast'   ,                                                      
'Lower-inner quadrant of breast'  :  'Breast'  ,                                                          
'Axillary tail of breast' : 'Breast' ,
'Breast, NOS': 'Breast' ,
'Lower-outer quadrant of breast' : 'Breast' ,                                                            
'Central portion of breast'  : 'Breast' ,
'Nipple'               : 'Breast' ,                                                                         
'Upper-inner quadrant of breast'  : 'Breast' , 
'Lower-outer quadrant of breast' : 'Breast' ,                                                              
'Central portion of breast'    : 'Breast'  ,
'Lower-inner quadrant of breast-Breast, NOS' : 'Breast'
}) 


df = df.dropna(axis = 0, how ='all')  
counts = df.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
df = df.drop(df.columns[to_del], axis=1)

columns_ = df.columns.tolist()

# Define a mapping dictionary for Persian to English
persian_to_english = {
    'خیر': 'No',
    'بلی': 'Yes',
    'مرد': 'Men',
    'زن': 'Female',
    'right' : 'Right',
    'left' :'Left',
    'bothSides': 'BothSides',    
                           
}

df[columns_] = df[columns_].replace(persian_to_english)

############################################Surgery Information Edition####################################
#set the values in the 'AxillyDissection' column to 'Yes' where the 'SurgeryType' 
#column equals 'Breast Conserving Surgery+Axillar Disection'
df.loc[df['SurgeryType'] == 'Breast Conserving Surgery+Axillar Disection', 'AxillyDissection'] = 'Yes'
df.loc[df['SurgeryType'] == 'Modified Radical Mastectomy+axillary dissection', 'AxillyDissection'] = 'Yes'
df.loc[df['SurgeryType'] == 'mastectomy + axillary resection', 'AxillyDissection'] = 'Yes'
df.loc[df['SurgeryType'] == 'lumpectomy + axillary resection', 'AxillyDissection'] = 'Yes'
df.loc[df['SurgeryType'] == 'partial mastectomy + axillary disection', 'AxillyDissection'] = 'Yes'

df.loc[((df['PickingNodeCount'] > 4)), 'AxillyDissection'] = 'Yes'
df.loc[((df['AxillyDissection'] == 'YES') & (df['SLNBX'].isnull())), 'SLNBX'] = 'No'


df.loc[:, 'SurgeryType'].replace(
    {
    "Breast Conserving Surgery+Axillar Disection": 'Lumpectomy',
    "Modified Radical Mastectomy+axillary dissection": 'Mastectomy',
    'mastectomy + axillary resection' : 'Mastectomy',
    'partial mastectomy + axillary disection' : 'Lumpectomy',
    'Breast Conserving Surgery': 'Lumpectomy',
    'Tumor Resection': 'Lumpectomy',
    'lumpectomy + axillary resection': 'Lumpectomy',
    'Total Hysterectomy and Bilateral Salpingo-Oophorectomy': 'Hysterectomy'
    }, inplace=True)

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
    'a'  : np.nan,
    '1C '    : 1,
    '1b'      : 1,
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
    '1b' : 1,
    '2b' : 2,
    '3c' : 3,
    '3a' : 3,
    '4c' : 4

    }, inplace=True)

df.loc[:, 'OrganDirection'].replace(
   {
    
    '--'  : np.nan,
    
    }, inplace=True)

df.loc[:, 'SurgeryDate'].replace(
   {
    
    '//'  : np.nan,
    '1396/22/4' : '1396/12/4'
    
    }, inplace=True)

df.loc[:, 'ClosestDistance'].replace(
   {
    
    're3'  : np.nan,
    '<0.1' : 1
    
    }, inplace=True)


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

#df.drop(columns=['Topography'], inplace=True)

df = df.dropna(axis=1, how='all') 


df['LNR'] = df['InvolvedNodeCount'] / df['PickingNodeCount']

colum = ['MaxTumorSize', 'LNR']

for col in colum:  
  df = remove_outliers_iqr(df.copy(), col) 
 

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

df['TValue'] = df['MaxTumorSize'].apply(get_t_value)

df.loc[(df['Skin'] == 'Yes'), 'TValue'] = 4

# Set 'Tvalue' to 4 where 'InvadeOrgans' is not null
df.loc[(df['InvadeOrgans'].notnull()), 'TValue'] = 4

df['InvadeOrgans'].replace({
    'near to superior': np.nan,
    'Duodenal wall, Ampula of Vater, peripancreatic fat': np.nan,
    'more than 50% Myometrium': np.nan,
}, inplace=True)

def get_n_value(num_positive_nodes):
    """
    Determine the N value of breast cancer based on the number of positive lymph nodes.
    
    Parameters:
    num_positive_nodes (int): Number of positive lymph nodes.
    
    Returns:
    str: N value category (N0, N1, N2, N3).
    """
    if num_positive_nodes is None:
        return None  
    if num_positive_nodes == 0:
        return 0  # No positive lymph nodes
    elif 1 <= num_positive_nodes <= 3:
        return 1  # 1-3 positive lymph nodes
    elif 4 <= num_positive_nodes <= 9:
        return 2  # 4-9 positive lymph nodes
    elif 10 <= num_positive_nodes :
        return 3  # 10 or more positive lymph nodes

df['NValue'] = df['InvolvedNodeCount'].apply(get_n_value)

df['NValue'].value_counts(dropna=False)

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


Grego_Date_Surgery = convert_persian_to_gregorian(df['SurgeryDate'])
df['Gregorian_SurgeryDate'] = pd.to_datetime(Grego_Date_Surgery)


# Add an incremental index within each group to pivot data later
df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['Gregorian_SurgeryDate','Organ','SurgeryType','OrganDirection'])
# [,'NumberofTumors','MaxTumorSize','PickingNodeCount','InvolvedNodeCount']
# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]
# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 


# Function to get the two earliest dates, handling cases with fewer than 2 dates
def get_two_earliest_dates(row):
    # Drop any NaT values (missing dates), then sort the remaining valid dates
    dates = row.dropna().sort_values()

    # If there's only one valid date, return it and NaT for the second
    if len(dates) == 1:
        return dates.iloc[0], pd.NaT
    
    # If there are two or more valid dates
    elif len(dates) >= 2:
        # If the first two dates are the same, return the first and the third if it exists
        if dates.iloc[0] == dates.iloc[1]:
            if len(dates) > 2:  # Ensure there is a third date
                return dates.iloc[0], dates.iloc[2]
            else:
                return dates.iloc[0], pd.NaT  # Not enough dates to provide a second
        else:
            return dates.iloc[0], dates.iloc[1]  # Return the two earliest dates

    # # If there are no valid dates, return NaT for both earliest dates
    # return pd.NaT, pd.NaT

# Apply the function to each row and create two new columns
df[['NearestsurgeryDate1', 'NearestsurgeryDate2']] = df[['Gregorian_SurgeryDate_1','Gregorian_SurgeryDate_2','Gregorian_SurgeryDate_3','Gregorian_SurgeryDate_4',
                                                         'Gregorian_SurgeryDate_5', 'Gregorian_SurgeryDate_6', 'Gregorian_SurgeryDate_7', 'Gregorian_SurgeryDate_8',
                                                        'Gregorian_SurgeryDate_9', 'Gregorian_SurgeryDate_10', 'Gregorian_SurgeryDate_11', 'Gregorian_SurgeryDate_12',
                                                        'Gregorian_SurgeryDate_13', 'Gregorian_SurgeryDate_14','Gregorian_SurgeryDate_15', 'Gregorian_SurgeryDate_16', 
                                                        'Gregorian_SurgeryDate_17', 'Gregorian_SurgeryDate_18',
                                                         ]].apply(get_two_earliest_dates, axis=1, result_type='expand')

# Format dates to 'dd/mm/yyyy' format
df['NearestsurgeryDate1'] = df['NearestsurgeryDate1'].dt.strftime('%d/%m/%Y')
df['NearestsurgeryDate2'] = df['NearestsurgeryDate2'].dt.strftime('%d/%m/%Y')


def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)


df['Organ_1'] = df['Organ_1'].fillna(df['Organ_2']).fillna(df['Organ_3'])
df['Organ_2'] = df['Organ_2'].fillna(df['Organ_3']).fillna(df['Organ_4'])
df['SurgeryType_1'] = df['SurgeryType_1'].fillna(df['SurgeryType_2']).fillna(df['SurgeryType_3'])
df = df.drop(columns=['SurgeryDate','Gregorian_SurgeryDate','Organ','idx','SurgeryType','OrganDirection',
                      'Gregorian_SurgeryDate_1','Gregorian_SurgeryDate_2','Gregorian_SurgeryDate_3',
                      'Gregorian_SurgeryDate_4','SurgeryType_3','SurgeryType_4','Organ_4','Organ_3',
                      'Gregorian_SurgeryDate_1','Gregorian_SurgeryDate_2','Gregorian_SurgeryDate_3','Gregorian_SurgeryDate_4',
                      'Gregorian_SurgeryDate_5', 'Gregorian_SurgeryDate_6', 'Gregorian_SurgeryDate_7', 'Gregorian_SurgeryDate_8',
                      'Gregorian_SurgeryDate_9', 'Gregorian_SurgeryDate_10', 'Gregorian_SurgeryDate_11', 'Gregorian_SurgeryDate_12',
                      'Gregorian_SurgeryDate_13', 'Gregorian_SurgeryDate_14','Gregorian_SurgeryDate_15', 'Gregorian_SurgeryDate_16',
                      'Gregorian_SurgeryDate_17', 'Gregorian_SurgeryDate_18', 'OrganDirection_3','OrganDirection_4',
                      'InvadeOrgans','Skin'])

df = df.drop(columns = ['Organ_5', 'Organ_6', 'Organ_7', 'Organ_8', 'Organ_9', 'Organ_10', 'Organ_11', 'Organ_12', 'Organ_13', 'Organ_14', 'Organ_15', 
                        'Organ_16', 'Organ_17', 'Organ_18', 'SurgeryType_5', 'SurgeryType_6', 'SurgeryType_7', 
                        'SurgeryType_8', 'SurgeryType_9', 'SurgeryType_10', 'SurgeryType_11', 'SurgeryType_12', 'SurgeryType_13', 'SurgeryType_14',
                        'SurgeryType_15', 'SurgeryType_16', 'SurgeryType_17', 'SurgeryType_18', 'OrganDirection_1', 'OrganDirection_2',
                        'OrganDirection_5', 'OrganDirection_6', 'OrganDirection_7', 'OrganDirection_8', 'OrganDirection_9', 'OrganDirection_10',
                        'OrganDirection_11', 'OrganDirection_12', 'OrganDirection_13','OrganDirection_14', 'OrganDirection_15', 'OrganDirection_16',
                        'OrganDirection_17', 'OrganDirection_18'])



df = df.drop(columns= ['Gregorian_SurgeryDate_19', 'Gregorian_SurgeryDate_20', 'Gregorian_SurgeryDate_21', 'Gregorian_SurgeryDate_22', 
                       'Gregorian_SurgeryDate_23', 'Gregorian_SurgeryDate_24', 'Gregorian_SurgeryDate_25', 'Gregorian_SurgeryDate_26', 
                       'Gregorian_SurgeryDate_27', 'Organ_19', 'Organ_20', 'Organ_21', 'Organ_22', 'Organ_23', 'Organ_24', 'Organ_25', 'Organ_26', 'Organ_27',
                       'SurgeryType_19', 'SurgeryType_20', 'SurgeryType_21', 'SurgeryType_22', 'SurgeryType_23', 'SurgeryType_24', 
                       'SurgeryType_25', 'SurgeryType_26', 'SurgeryType_27', 'OrganDirection_19', 'OrganDirection_20', 'OrganDirection_21', 
                       'OrganDirection_22', 'OrganDirection_23', 'OrganDirection_24', 'OrganDirection_25', 'OrganDirection_26', 'OrganDirection_27',
                  ])


#df = df[~df['SurgeryType_2'].isin(['Hysterectomy', 'Glossectomy', 'Sigmoidectomy'])]

df.rename(columns={
'OrganDirection_1' : 'First Organ Direction' , 
'OrganDirection_2' : 'Second Organ Direction' , 
'Organ_1' : 'First Surgery Organ', 
'Organ_2' : 'Second Surgery Organ', 
'SurgeryType_1' : 'First Surgery Type', 
'SurgeryType_2' : 'Second Surgery Type',
'MorphologyName':'Morphology Name', 
'LymphaticEmboli': 'Lymphatic Emboli',
'SLNBX': 'SLNBX',
'ClosestDistance' : 'Closest Distance',
'ClosestDistanceType' : 'Closest Distance Type',
'AxillyDissection': 'Axillary Dissection',
'InvolvedMargin': 'Involved Margin', 
'ClosestDistance': 'Closest Distance', 
'ClosestDistanceType': 'Closest Distance Type', 
'PickingNodeCount': 'Picking Node Count',
'InvolvedNodeCount': 'Positive Axillary Nodes',
'TValue': 'T Value',
'NValue': 'N Value',
'MValue': 'M Value',
'NumberofTumors': 'Number of Tumors', 
'MaxTumorSize': 'Max Tumor Size',
'NearestsurgeryDate1' : 'First Surgery Date',
'NearestsurgeryDate2': 'Second Surgery Date',
'TumorSizes' : 'Tumor Sizes'
}, inplace=True)

df = df[df['First Surgery Organ'].isin(['Breast', None])]

df['First Surgery Type'] = df['First Surgery Type'].fillna(df['Second Surgery Type'])
df['First Surgery Type'].value_counts(dropna=False)

df.drop(columns=['Morphology','Topography','TopographyName'],inplace=True)
df.columns.tolist()
df['First Surgery Type'].replace({
'mandibulectomy' : np.nan,
'Cecocolectomy'   : np.nan,
'bilobectomy' : np.nan,
'lymphadenopathy': np.nan,

}, inplace=True)

df['M Value'].value_counts()
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/SU.xlsx', index=False)

