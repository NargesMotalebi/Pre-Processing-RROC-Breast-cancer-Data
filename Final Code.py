from persiantools.jdatetime import JalaliDate
import matplotlib.pyplot as plt
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
#*************************Diagnosis*******************
df_Diagnosis=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DI.xlsx')
#*************************Demographic*******************
df_Demographic=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/DE.xlsx')
df_ = pd.merge(df_Diagnosis, df_Demographic,  on='DocumentCode', how='outer')
#************************Midwifery_Information************************
df_Midwifery_Information = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/MI.xlsx')
df = pd.merge(df_, df_Midwifery_Information,  on='DocumentCode', how='outer')
#***********************Read Family_History**********************
df_Family_History = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/FH.xlsx') 
df = pd.merge(df, df_Family_History,  on=['DocumentCode'], how='outer')
#*****************************************
df_Symtoms = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/SY.xlsx')
df = pd.merge(df, df_Symtoms , on='DocumentCode', how='outer')
#***********************Read Surgery and delete all columns with Null***********************
df_Surgery =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/SU.xlsx')
df = pd.merge(df, df_Surgery , on='DocumentCode', how='outer')
#*************************Treatmentn_Information*****************************
df_Treatmentn_Information=  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/TI.xlsx')
df = pd.merge(df, df_Treatmentn_Information , on='DocumentCode', how='outer')
#*************************Patient_Situation************************
df_Patient_Situation = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/PS.xlsx') 
df = pd.merge(df, df_Patient_Situation,  on='DocumentCode', how='outer',suffixes=('','_df'))
#*************************Morphology************************
df_TI_Mor = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/TI_Mor.xlsx') 
df = pd.merge(df, df_TI_Mor,  on='DocumentCode', how='outer',suffixes=('','_df'))
#******************************************IHC
df_IHC = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/IHC.xlsx') 
df = pd.merge(df, df_IHC,  on='DocumentCode', how='outer',suffixes=('','_df'))
#******************************************Dates
df_BiosurDates = pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/BiosurDates.xlsx') 


commonlistofcolumns = ['First Biopsi Date', 'Second Biopsi Date', 'First Surgery Date', 'Second Surgery Date']
df = pd.merge(df, df_BiosurDates,  on='DocumentCode', how='outer',suffixes=('','_df'))

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_df'])

df = df.drop(columns=[f'{col}_df' for col in commonlistofcolumns])


df = df.dropna(axis = 0, how ='all') 
df = df.dropna(axis=1, how='all')

start_date_ = pd.to_datetime('2019/03/21')
end_date_ = pd.to_datetime('2024/09/21')
df['Document Date'] = pd.to_datetime(df['Document Date'], dayfirst=True, errors='coerce')
df = df[(df['Document Date'].isna()) | ((df['Document Date'] >= start_date_) & (df['Document Date'] <= end_date_))]
#********************************************************
df = df.dropna(subset=['First Biopsi Date', 'First Surgery Date','First Treatment Date'], how='all')
#******************************************************************************************************
columns_ = df.columns.tolist()  
df = df[df['First Treatment Date'].notnull()]
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group  # If only one row, keep it
    else:
        # Calculate missing values count for each row in the group (across all columns)
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        # Find the index of the row with the minimum number of missing values
        min_missing_values_index = missing_values_counts.idxmin()
        # Return only the row with the least missing values
        return group.loc[[min_missing_values_index]]

# Apply the function within each group of 'DocumentCode'
df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)
#************************************************************
#df['Document Date'] = df['Document Date'].fillna(df['First Treatment Date'])
df['Document Date'].isnull().sum()
#*******************************************************
def custom_grouping(date):
    year = date.year
    month = date.month
  
    if month <= 3:
        return f"{year-1}"
    else:
        return f"{year}"

df['period'] = df['Document Date'].apply(custom_grouping)
grouped = df.groupby(df['Document Date'].dt.to_period('M')).size().reset_index(name='Count')
# # #**********************************************
# patients_per_year = df.groupby('period').size()
# patients_per_year_ = patients_per_year.iloc[1:6]
# patients_per_year__ = grouped.iloc[1:6]

# plt.figure(figsize=(12, 10))
# plt.gca().set_axisbelow(True)
# plt.plot(patients_per_year_.index, patients_per_year_.values, marker='o', color='darkblue', markersize=9, linestyle='-',lw=2.2)

# # Annotate each data point with the number of patients referred in each year
# for year, patients in patients_per_year_.items():
#     plt.text(year, patients, str(patients), ha='center', va='bottom',fontsize = 35)

# plt.title('')
# plt.xlabel('Year',fontsize=34)
# plt.ylabel('Number of Patients',fontsize=34)
# plt.xticks(patients_per_year_.index)  # Set x-axis ticks to integer years
# plt.grid(axis ='y', linestyle='-', alpha=0.8, linewidth=1.5)
# plt.ylim(0, patients_per_year.max() * 1.75)  # Adjust the multiplier as needed for your data
# plt.xticks(rotation=0,fontsize=32)
# plt.yticks(fontsize=32)  
# plt.tight_layout(pad=2.0)
# plt.show() 
#**********

df = df.dropna(subset=['First Biopsi Date', 'First Surgery Date'], how='all')
df['First Biopsi Date'] = pd.to_datetime(df['First Biopsi Date'], dayfirst=True, errors='coerce')
df['First Surgery Date'] = pd.to_datetime(df['First Surgery Date'], dayfirst=True, errors='coerce')
df['Birth Day'] = pd.to_datetime(df['Birth Day'], dayfirst=True, errors='coerce')
df['First Metastastasic Date'] = pd.to_datetime(df['First Metastastasic Date'], dayfirst=True, errors='coerce')
df['First Recurrence Date'] = pd.to_datetime(df['First Recurrence Date'], dayfirst=True, errors='coerce')

start_date = pd.to_datetime('2019/03/21')
end_date = pd.to_datetime('2024/03/21')
df['First Treatment Date'] = pd.to_datetime(df['First Treatment Date'], dayfirst=True, errors='coerce')
df = df[(df['First Treatment Date'].isna()) | ((df['First Treatment Date'] >= start_date) & (df['First Treatment Date'] <= end_date_))]
df = df[(df['First Recurrence Date'].isna()) | ((df['First Recurrence Date'] >= start_date) & (df['First Recurrence Date'] <= end_date_))]
df = df[(df['First Metastastasic Date'].isna()) | ((df['First Metastastasic Date'] >= start_date) & (df['First Metastastasic Date'] <= end_date_))]
df = df[(df['First Biopsi Date'].isna()) | ((df['First Biopsi Date'] >= start_date) & (df['First Biopsi Date'] <= end_date))]
df = df[(df['First Surgery Date'].isna()) | ((df['First Surgery Date'] >= start_date) & (df['First Surgery Date'] <= end_date_))]
#************************************************************************
df['First Biopsi Date'] = df['First Biopsi Date'].fillna(df['First Surgery Date'])

#***********************************************************Determine Treatmentprotocol, Age at Diagnosis**********************************
df['Age at Diagnosis'] = df.apply(
    lambda row: (
        round(((row['First Biopsi Date'] if pd.notna(row['First Biopsi Date']) else row['First Surgery Date']) - row['Birth Day']).days / 365.25, 1)
        if pd.notna(row['Birth Day']) and (pd.notna(row['First Biopsi Date']) or pd.notna(row['First Surgery Date'])) else pd.NA
    ),
    axis=1
)
df['Age at Diagnosis'].isnull().sum()
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
df['Sex'].value_counts(dropna=False)
#********************************************Determind Survival Time*************************
reference_date_1 = pd.Timestamp('2024-03-21')
reference_date_2 = pd.Timestamp('2024-09-21')

def calculate_survival_time(row):
    biopsy_date = row['First Biopsi Date']
    treatment_date = row['First Treatment Date']
    
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
        # Event 0: Treatment Date - Reference Date
        if pd.notna(biopsy_date):
            if biopsy_date < reference_date_1:
                return (reference_date_1 - biopsy_date).days
            else:
                return (reference_date_2 - biopsy_date).days
        else:
            return np.nan  # if Treatment Date is missing
        
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


df.loc[(df['TreatmentProtocol'] == 'Adjuvant') & (df['Treatment Procedure'] == 'Neo-Adjuvant'), 'TreatmentProtocol'] = 'Neo-Adjuvant'
df.loc[df['TreatmentProtocol'] == 'Palliative', 'TreatmentProtocol'] = df.loc[df['TreatmentProtocol'] == 'Palliative', 'Treatment Procedure']
df.loc[df['TreatmentProtocol'] == 'Definitive', 'TreatmentProtocol'] = df.loc[df['TreatmentProtocol'] == 'Definitive', 'Treatment Procedure']
df['TreatmentProtocol'] = df['TreatmentProtocol'].fillna(df['Treatment Procedure'])

df['Survival Time'] = df.apply(calculate_survival_time, axis=1)

df['Status'].isnull().sum()
df['Status'].value_counts(dropna=False)
df['Survival Time'] = df['Survival Time'].apply(lambda x: max(x, 0))
df = df[(~df['Status'].isnull())]


# df1 = df[(df['Status'] == 1) & (df['Survival Time']< 240)] 
# df1.shape     
#df1.to_excel('G:/Breast Cancer/Reza Ancology/Edited/stage4.xlsx', index=False)
df.shape
df = df[~((df['Survival Time']> 1827))]
df = df[
    (((df['Status'] == 1) & (df['Survival Time'] > 240)) |
    ((df['Status'] == 3) & (df['Survival Time'] > 240))) |
    (df['Event'] == 0) |
    (df['Status'] == 2)
]
df.loc[(df['Status'] == 2) & (df['Survival Time'] < 240), 'Survival Time'] = np.nan
# df['Status'].replace({
#     3:1
# },inplace=True)
df[df['Status']==1]['Survival Time'].describe()
df['Survival Time'].describe()
df['Event'].value_counts()
df.shape[0]
df = df[(~df['Age at Diagnosis'].isnull())]
#((df['Status'] == 2) & (df['Survival Time'] > 210)) |
#df = df[~((df['Is Beginning Disease'] == 'No') & (df['Metastasis Status'] =='Yes'))]
#df = df[~((df['Is Beginning Disease'] == 'No') & (df['Recurrence Status'] =='Yes'))]
# df['Survival Time'].isnull().sum()
# df[df['TreatmentProtocol'] == 'Palliative']['Status'].value_counts()
#*******************************************
df.loc[df['Morphology Name'] == 'Insitu', 'T Value'] = 0

condition = (
    (df['TreatmentProtocol'] == 'adjuvant') & 
    (df['Morphology Name'].isin(['Ductal', 'Lobular', 'Mix', 'Other'])) & 
    (df['Max Tumor Size'] == 0)
)

df.loc[condition, 'Max Tumor Size'] = np.nan
df.loc[condition, 'Tumor Sizes'] = np.nan
df.loc[condition, 'T Value'] = np.nan

df['First Treatment Date'].isnull().sum()
# df = df.dropna(subset=['Status'], axis=0)

df['Type of First Treatment'].value_counts(dropna=False)
df = df.drop(columns=['Treatment Procedure','period','Is Beginning Disease','Metastasis Status','Recurrence Status'])
df.loc[(df['Age at Menarche'].notnull())&(df['Sex'].isnull()), 'Sex']   =   'Female'
#df = df[(df['Type of First Treatment'].isna()) | ((df['Type of First Treatment'] == 'Radiotherapy'))]

df = df[df['Status'].notnull()]

df[df['Type of First Treatment'] == 'Hormone Therapy']['Type of Second Treatment'].value_counts(dropna=False)

df['First Surgery Type'].replace({
'mandibulectomy' : np.nan,
'Cecocolectomy'   : np.nan,
'bilobectomy' : np.nan,
'lymphadenopathy': np.nan,

}, inplace=True)

df[df['Type of First Treatment'] == 'Radiotherapy']['Status'].value_counts()

df['TopographyName'].replace({
'Breast, NOS-Lower-inner quadrant of breast' : 'Breast',
},inplace=True)


df['Grade'].replace({
'Undifferentiated' : 'Poorly differentiate',
},inplace=True)
df['Event'].value_counts(dropna=False)
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/Final_without_Replacemnt_all3.xlsx', index=False)
df2['LVI'].value_counts()
df2 = df[df['Event']==1]
# df3 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/Edited/Final_without_Replacemnt_all.xlsx')
# df4 =df3[df3['Event']==0]

# df5 = pd.concat([df4, df2], join='inner', ignore_index=True)
# df['Event'].value_counts(dropna=False)
# df['Sex'].value_counts(dropna=False)
#df2.to_excel('G:/Breast Cancer/Reza Ancology/Edited/stage4.xlsx', index=False)