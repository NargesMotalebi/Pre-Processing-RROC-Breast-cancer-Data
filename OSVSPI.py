import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import os
import csv


df1 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/TI/Treatment Information.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])
df2 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/TI/Treatment Information2.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])
df3=  pd.read_excel('G:/Breast Cancer/Reza Ancology/TI/Treatment-allyears.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])

df4 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/PI/Patient Situation.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])
df5 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/PI/Patient Situation2.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])
df6 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/PI/Patientsituation-allyears.xlsx',usecols=['DocumentCode','Morphology', 'Topography', 'TopographyName'])

df7 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df8 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery2.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df9 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/Surgery-allyears.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )

df10 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Biosidate.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df11 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df12 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis2.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df13 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis1400.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df14 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/Diagnosis-allyears.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )

df15 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )

df16 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History2.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df17 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/FamilyHistory-allyears.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )

df18 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery Information.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df19 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery Information2.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )
df20 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/MI/Midwifery-allyears.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )


df21 =  pd.read_excel('G:/Breast Cancer/Reza Ancology/SY/Indication-allyears.xlsx',usecols=['DocumentCode', 'Morphology', 'Topography', 'TopographyName'] )


# Define the function to merge columns in a DataFrame based on DocumentCode
def combine_columns(df, commonlistofcolumns, key_column='DocumentCode'):
    for col in commonlistofcolumns:
        if f'{col}_df2' in df.columns:
            # Merge based on the key column, adding any unmatched rows
            merged_df = df[[key_column, col]].merge(
                df[[key_column, f'{col}_df2']],
                on=key_column,
                how='outer',
                suffixes=('', '_df2')
            )
            # Combine values from the '_df2' column into the original column
            df[col] = merged_df[col].combine_first(merged_df[f'{col}_df2'])
            # Drop the '_df2' column as it is now redundant
            df = df.drop(columns=f'{col}_df2')
    return df

# Define the list of common columns to combine
commonlistofcolumns = ['Morphology', 'Topography', 'TopographyName']

# List of DataFrames to process
dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15,df16,df17,df18,df19,df20,df21]

# Apply the function to each DataFrame
for i, df in enumerate(dataframes):
    dataframes[i] = combine_columns(df, commonlistofcolumns, key_column='DocumentCode')

df['Topography'] = df['Topography'].replace({
'C50.9--' : 'C50.9',
'C50.9-' : 'C50.9'
})  
df['Morphology'] = df['Morphology'].replace({
'8500/3--' : '8500/3',
'8500/3-' : '8500/3',
'-' :np.nan
})  
df['TopographyName'] = df['TopographyName'].replace({
'Breast, NOS--' : 'Breast',
'Breast, NOS-'  : 'Breast',
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
# Words to filter by
words_to_filter = ['C50']

pattern = '|'.join(words_to_filter)
filtered_df = df[df['Topography'].str.contains(pattern, case=False, na=True) | df['Topography'].isna()]
df = filtered_df
df['Topography'].value_counts(dropna=False)
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
df.shape
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/TI_Mor.xlsx', index=False)

