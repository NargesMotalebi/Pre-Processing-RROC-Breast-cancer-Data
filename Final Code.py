from persiantools.jdatetime import JalaliDate
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
from os import listdir

def compare_common_columns(df1, df2):
    """
    Compare common columns between two pandas DataFrames.
    
    Parameters:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
    
    Returns:
        str: A message indicating whether the common columns are the same or different.
    """
    # Get the common columns between the two DataFrames
    common_columns = set(df1.columns).intersection(set(df2.columns))
    
    if not common_columns:

        return "No common columns found between the two DataFrames."
        
    
    return common_columns

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


#The Order of reading files is Important here
#***********************Read Surgery and delete all columns with Null***********************
df_Surgery =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/SU.xlsx')
#***********************Read Family_History**********************
df_Family_History = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/FH.xlsx') 
df = pd.merge(df_Surgery, df_Family_History,  on=['DocumentCode'], how='outer')
#*************************Treatmentn_Information*****************************
df_Treatmentn_Information=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/TI.xlsx')
df = pd.merge(df, df_Treatmentn_Information,  on='DocumentCode', how='outer')
#************************Midwifery_Information************************
df_Midwifery_Information = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/MI.xlsx') 
df = pd.merge(df, df_Midwifery_Information,  on='DocumentCode', how='outer')
#*************************Patient_Situation************************
df_Patient_Situation = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/PS.xlsx') 
df = pd.merge(df, df_Patient_Situation,  on='DocumentCode', how='outer')
#*****************************************
df_Indication = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/Indication.xlsx')
df = pd.merge(df_Indication, df,  on='DocumentCode', how='outer')
#*************************Diagnosis*******************
df_Diagnosis=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DI.xlsx')
common_columns =  compare_common_columns(df_Diagnosis, df)
columns_to_remove = [col for col in common_columns if col not in 'DocumentCode']
df = pd.merge(df_Diagnosis, df,  on='DocumentCode', how='outer')
# Filter the DataFrame to keep only rows related to the specified value in column 'specified_value'
# filtered_df = df[df['Topography'] == specified_value ]
# df = filtered_df
df.columns.tolist()
df.shape
#***************************Integrate Sex columnes*************************
df.loc[df['Sex_x'].isna(),'Sex_x'] = df['Sex_y']
# Rename the column
df.rename(columns={'Sex_x': 'Sex'}, inplace=True)

df.loc[df['TreatmentStartDate_y'].isna(),'TreatmentStartDate_y'] = df['TreatmentStartDate_x']
df.rename(columns={'TreatmentStartDate_y': 'TreatmentStartDate'}, inplace=True)
df.drop(columns=['TreatmentStartDate_x'], inplace=True)


# Define the list of columns to consider
columns_ = df.columns.tolist()

#********************************************Status column based on conditions******************
df['Status'] = np.where(df['ReActiveDate'].notnull() | df['FirstDate'].notnull(), 1, 0)
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
#***********************************************************Determine Treatmentprotocol, Age at Diagnosis**********************************
# we convert the time of surgery, BiopsiDate, TreatmentStartDate and Birth day to Gregorian date
Grego_Date_Surgery = convert_persian_to_gregorian(df['SurgeryDate'])
Grego_Date_Biopsi = convert_persian_to_gregorian(df['BiopsiDate'])
Grego_Date_BirthDay = convert_persian_to_gregorian(df['BirthDate'])
Grego_Date_Treatment = convert_persian_to_gregorian(df['TreatmentStartDate'])
G_ReActiveDate = convert_persian_to_gregorian(df['ReActiveDate'])
G_FirstMetastasicDate = convert_persian_to_gregorian(df['FirstDate'])

df['Gregorian_SurgeryDate'] = pd.to_datetime(Grego_Date_Surgery)
df['Gregorian_BiopsiDate'] = pd.to_datetime(Grego_Date_Biopsi)
df['Gregorian_BirthDay'] = pd.to_datetime(Grego_Date_BirthDay)
df['Gregorian_TreatmentStartDate'] = pd.to_datetime(Grego_Date_Treatment)
df['G_ReActiveDate'] = pd.to_datetime(G_ReActiveDate)
df['G_FirstMetastasicDate'] = pd.to_datetime(G_FirstMetastasicDate)
df['Gregorian_BirthDay'].isnull().sum()

# Specific value to compare biopsy date against
# specific_date = pd.to_datetime('2017/12/10')#10 azar 1396
# # #2022/12/01 is azar 1401
# # # # Filter rows where biopsy date is greater than or equal to the specific value
# filtered_df = df[df['Gregorian_BiopsiDate'] >= specific_date]
# df = filtered_df 
# # specific_date_ = pd.to_datetime('2022/12/15')#10 azar 1401
# # filtered_df_ = df[df['Gregorian_BiopsiDate'] <= specific_date_]
# # df = filtered_df_
# df.shape

Status_mask = df['Status'] == 0
# Replace null values in Gregorian_BiopsiDate with SurgeryDate if the status is '0'
df.loc[Status_mask, 'Gregorian_BiopsiDate'] = df.loc[Status_mask, 'Gregorian_BiopsiDate'].fillna(df.loc[Status_mask, 'Gregorian_SurgeryDate'])
# Replace null values in Gregorian_BiopsiDate with Gregorian_TreatmentDate - 7 months if the status is '0'
df.loc[Status_mask, 'Gregorian_BiopsiDate'] = df.apply(lambda row: row['Gregorian_TreatmentStartDate']  if pd.isnull(row['Gregorian_BiopsiDate']) else row['Gregorian_BiopsiDate'], axis=1)
# Replace null values in Gregorian_BiopsiDate with Gregorian_SurgeryDate - 25 days if the status is '1'
# Status_mask_ = df['Status'] == 1
# df.loc[Status_mask_, 'Gregorian_BiopsiDate'] = df.apply(lambda row: row['Gregorian_SurgeryDate'] - pd.DateOffset(day=25) if pd.isnull(row['Gregorian_BiopsiDate']) else row['Gregorian_BiopsiDate'], axis=1)

df.shape

#***********************************************************Determine Type of Surgery(NeoAdjuvant/Adjuvant)*********************************
df['Duration'] = (df['Gregorian_SurgeryDate']-df['Gregorian_BiopsiDate']).dt.days
df['Treatmentprotocol'] = df.apply(lambda row: 
    'NeoAdjuvant' if row['Duration'] >= 65 
    else ('Adjuvant' if row['Duration'] < 65           
        else ''), axis=1)


#********************************************Determind Survival Time*************************
#*****End of study is '2024/12/10'
# Convert '2022/12/10' to a datetime object
farthest_time =  '2024/12/10'    #df['Gregorian_BiopsiDate'].max()
target_date = pd.to_datetime(farthest_time)

# Calculate 'SurvivalTime' based on 'Status'
def calculate_survival_time(row):
    if pd.notnull(row['G_FirstMetastasicDate']):
        return (row['G_FirstMetastasicDate'] - row['Gregorian_BiopsiDate']).days
    elif pd.notnull(row['G_ReActiveDate']):
        return (row['G_ReActiveDate'] - row['Gregorian_BiopsiDate']).days
    else:
        return (target_date - row['Gregorian_BiopsiDate']).days
    
# Apply the function using .apply along with .loc
df.loc[:, 'Survival_Time'] = round(df.apply(calculate_survival_time, axis=1))

df.shape
df.loc[df['BiopsiOrganDirection'].isna(), 'BiopsiOrganDirection'] = df['OrganDirection']
# Filter rows where 'Status' is not equal to 1 or 'Survival_Time' is greater than or equal to 30
#df = df[df['Survival_Time'] > 0]
#df = df[(df['Status'] == 1) & (df['Survival_Time'] > 30)]

df.shape
# Calculate 'Age' based on 'Status'
def calculate_Age(row):
    if pd.notnull(row['Gregorian_BiopsiDate']):
        return (row['Gregorian_BiopsiDate'] - row['Gregorian_BirthDay']).days/365
    elif pd.notnull(row['Gregorian_SurgeryDate']):
        return (row['Gregorian_SurgeryDate'] - row['Gregorian_BirthDay']).days/365
    else:
        return (np.nan)
    
df['Age at Diagnosis'] = round(df.apply(calculate_Age, axis=1))#.astype(int)

# List of columns to format
columns_to_format = ['Gregorian_SurgeryDate', 'Gregorian_BirthDay', 'Gregorian_BiopsiDate', 'Gregorian_TreatmentStartDate', 'G_ReActiveDate', 'G_FirstMetastasicDate']

# Apply strftime to each column
for column in columns_to_format:
    df[column] = pd.to_datetime(df[column]).dt.strftime('%Y/%m/%d')

df = remove_outliers_iqr(df.copy(), 'MaxTumorSize')
# Filter rows where treatment protocol is 'adjuvant' and tumor size is 0, then replace 'TumorSize' with null
adjuvant_and_zero_size_mask = (df['Treatmentprotocol'] == 'adjuvant') & (df['MaxTumorSize'] == 0)
df.loc[adjuvant_and_zero_size_mask, 'MaxTumorSize'] = np.nan
df.columns.tolist()
#*************************Drop columns that we do not need any more
df.drop(columns=['MetastasisOrgans','TreatmentOrgan','OrganDirection','InvadeOrgans'], inplace=True)
df.drop(columns=['BiopsiDate','SurgeryDate','TreatmentStartDate','ReActiveDate','FirstDate','BirthDate'], inplace=True)
df.drop(columns=['IsReactive','IsBeginningDisease','IsMetastasis'], inplace=True)
#df.drop(columns=['FirstMetastasisOrgan','FristMetastasisName'], inplace=True)
#df.drop(columns=['SecondMetastasisOrgan','SecondMetastasisName','SecondDate','ThirdMetastasisName'], inplace=True)
df.drop(columns=['Duration','Sex_y','Staging','Fraction'], inplace=True)
df.drop(columns=['Topography','Inferior','TreatmentType','TumorSizes','MValue'], inplace=True)
#df.drop(columns=['G_ReActiveDate','G_FirstMetastasicDate'], inplace=True)
#df.drop(columns=['Gregorian_SurgeryDate','Gregorian_BiopsiDate','Gregorian_TreatmentStartDate','Gregorian_BirthDay'], inplace=True)

df.rename(columns={'BiopsiOrganDirection' : 'Organ Direction' , 
'MorphologyName':'Morphology Name', 
'BirthCountry': 'Birth Country',
'IsRural':'Residency(Rural)' , 
'JobStatus':'Job Status', 
'SurgeryType' : 'Surgery Type',
'LymphaticEmboli': 'Lymphatic Emboli',
'MariageStatus':'Marital Status', 
'SLNBX': 'SLNBX', 
'AxillyDissection': 'Axillary Dissection',
'InvolvedMargin': 'Involved Margin', 
'ClosestDistance': 'Closest Distance', 
'ClosestDistanceType': 'Closest Distance Type', 
'PickingNodeCount': 'Picking Node Count',
'InvolvedNodeCount': 'Positive Axillary Nodes',
'TValue': 'T Value',
'NValue': 'N Value',
'NumberofTumors': 'Number of Tumors', 
'MaxTumorSize': 'Tumor Size', 
'FamilyHistoryTopographyName': 'Family History Topography Name', 
'FamilyRelation': 'Family Relation',
'FamilyAge': 'Family Age', 
'TotalDose': 'Total Dose', 
'FirstPeriodAge': 'First Period Age', 
'GestationCount': 'Gestation Count', 
'AbortionCount': 'Abortion Count', 
'StillbirthCount': 'Stillbirth Count', 
'FirstGestationAge': 'First Gestation Age', 
'LastGestationAge': 'Last Gestation Age', 
'LactationStatus': 'Lactation Status', 
'LoctationDurationTime': 'Loctation Duration Time',
'CesareanStatus': 'Cesarean Status', 
'CesareanCount': 'Cesarean Count', 
'IsRegularPerion': 'Menstrual regularity', 
'MenopauseStatus': 'Menopause Status', 
'MenopauseAge': 'Menopause Age',
'MenopauseCause': 'Menopause Cause', 
'UsingOCP': 'Using OCP', 
'OCPDurationTime': 'OCP Duration', 
'Treatmentprotocol': 'Treatment protocol',
'Gregorian_SurgeryDate' : 'Surgery Date',
'Gregorian_BiopsiDate'  : 'Biopsi Date',
'Gregorian_BirthDay'    :  'Birth Day',
'InsituPercent' : 'Insitu Percent',
'InsituPercentType' : 'Insitu Percent Type',
'Gregorian_TreatmentStartDate':'Treatment Start Date',
'G_ReActiveDate' : 'ReActive Date',
'G_FirstMetastasicDate' : 'First Metastasic Date',
'SymptomsName' : 'Symptoms Name',
'MonthDuration' : 'Month Duration',
'Survival_Time' : 'Survival Time',
'MonthDuration' : 'Duration (Month)'
}, inplace=True)

colum = [ 'Closest Distance', 
          'Max Tumor Size', 'LNR', 'Family Age', 'Total Dose', 
          'First Period Age', 'Gestation Count', 'Abortion Count', 'Stillbirth Count', 
          'First Gestation Age', 'Last Gestation Age', 'Loctation Duration Time',
          'Cesarean Count', 'Menopause Age', 'OCP Duration']

for col in colum:  
  df = remove_outliers_iqr(df.copy(), col) 
 
df['Number of Tumors'].value_counts()

df.shape
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/Final_without_Replacemnt_all.xlsx', index=False)


#Filter rows where both ReActiveDate and FirstDate are null
# null_values = df[df['ReActive Date'].notnull() | df['First Metastasic Date'].notnull()]

# null_values.to_excel('G:/Breast Cancer/Reza Ancology/Edited/null_values.xlsx', index=False)

# not_null_values = df[df['ReActive Date'].isnull() & df['First Metastasic Date'].isnull()]

# not_null_values.to_excel('G:/Breast Cancer/Reza Ancology/Edited/not_null_values.xlsx', index=False)

filtered_df_Male = df[df['Sex'] == 'Men']
filtered_df_Female = df[df['Sex'] == 'Female']

# filtered_df_Female.to_excel('G:/Breast Cancer/Reza Ancology/Edited/Final_without_Replacemnt_Female_all.xlsx', index=False)
# filtered_df_Male.to_excel('G:/Breast Cancer/Reza Ancology/Edited/Final_without_Replacemnt_Male_all.xlsx', index=False)

df[['T Value', 'N Value', 'Total Dose','M Value', 'Gestation Count', 'Abortion Count', 'Stillbirth Count']].astype('object')

def calculate_missing_values(df):
    numeric_features = df.select_dtypes(include=np.number).columns
    categorical_features = df.select_dtypes(include='object').columns

    # Calculate the precent of missing values for each feature
    missing_values = df.isnull().mean()*100
    # Calculate mean and interquartile range for continuous variables
    continuous_stats = df[numeric_features].describe()

    # Calculate proportion for categorical and binary variables
    categorical_stats = df[categorical_features].describe()

    return missing_values, continuous_stats, categorical_stats


# # Calculate statistics
# missing_values_, continuous_stats_, categorical_stats_ = calculate_missing_values(filtered_df_Female)
# missing_values__, continuous_stats__, categorical_stats__ = calculate_missing_values(filtered_df_Male)
# missing_values, continuous_stats, categorical_stats = calculate_missing_values(df)

# # import xlsxwriter
# # # Create a Pandas Excel writer using XlsxWriter as the engine
# writer = pd.ExcelWriter('G:/Breast Cancer/Reza Ancology/Edited/data_statistics.xlsx', engine='xlsxwriter')

# # Write each DataFrame to a separate worksheet
# missing_values_.to_excel(writer, sheet_name='Missing_Values1')
# continuous_stats_.to_excel(writer, sheet_name='Continuous_Stats1')
# categorical_stats_.to_excel(writer, sheet_name='Categorical_Stats1')
# missing_values.to_excel(writer, sheet_name='Missing_Values')
# continuous_stats.to_excel(writer, sheet_name='Continuous_Stats')
# categorical_stats.to_excel(writer, sheet_name='Categorical_Stats')
# missing_values__.to_excel(writer, sheet_name='Missing_Values2')
# continuous_stats__.to_excel(writer, sheet_name='Continuous_Stats2')
# categorical_stats__.to_excel(writer, sheet_name='Categorical_Stats2')
#  # Close the Pandas Excel writer and output the Excel file
# writer.close()

# import xlsxwriter
# Create a Pandas Excel writer using XlsxWriter as the engine
# writer = pd.ExcelWriter('G:/Breast Cancer/Reza Ancology/Edited/data_statistics.xlsx', engine='xlsxwriter')
# Write each DataFrame to a separate worksheet
# missing_values.to_excel(writer, sheet_name='Missing_Values')
# continuous_stats.to_excel(writer, sheet_name='Continuous_Stats')
# categorical_stats.to_excel(writer, sheet_name='Categorical_Stats')

# # Close the Pandas Excel writer and output the Excel file
# writer.close()

