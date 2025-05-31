from persiantools.jdatetime import JalaliDate
from datetime import date
import pandas as pd
import numpy as np
import csv


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


df_DateSurgery=  pd.read_excel('G:/Breast Cancer/Reza Ancology/SU/surgerydate.xlsx')
df_Datebiosidate  = pd.read_excel('G:/Breast Cancer/Reza Ancology/DI/biopsydate.xlsx')


df = pd.merge(df_DateSurgery, df_Datebiosidate,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

df = df.dropna(axis = 0, how ='all')  
df = df.dropna(axis = 1, how ='all')  
df.columns.tolist()
Grego_Date_Surgery = convert_persian_to_gregorian(df['SurgeryDate'])
df['Gregorian_SurgeryDate'] = pd.to_datetime(Grego_Date_Surgery)

Grego_Date_Surgery = convert_persian_to_gregorian(df['BiopsiDate'])
df['Gregorian_BiopsiDate'] = pd.to_datetime(Grego_Date_Surgery)


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

df[['NearestBiopsiDate1', 'NearestBiopsiDate2']] = df[
    ['Gregorian_BiopsiDate_1','Gregorian_BiopsiDate_2','Gregorian_BiopsiDate_3',
     'Gregorian_BiopsiDate_4','Gregorian_BiopsiDate_5','Gregorian_BiopsiDate_6']
].apply(get_two_earliest_dates, axis=1, result_type='expand')

df.columns.tolist()


# Add an incremental index within each group to pivot data later
df['idx_'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx_', values=['Gregorian_SurgeryDate'])
# [,'NumberofTumors','MaxTumorSize','PickingNodeCount','InvolvedNodeCount']
# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]
# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 


# Apply the function to each row and create two new columns
df[['NearestsurgeryDate1', 'NearestsurgeryDate2']] = df[['Gregorian_SurgeryDate_1','Gregorian_SurgeryDate_2','Gregorian_SurgeryDate_3','Gregorian_SurgeryDate_4',
                                                         'Gregorian_SurgeryDate_5', 'Gregorian_SurgeryDate_6']].apply(get_two_earliest_dates, axis=1, result_type='expand')

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


df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

df = df.drop(columns=['Gregorian_BiopsiDate','Gregorian_BiopsiDate_1','Gregorian_BiopsiDate_2','Gregorian_BiopsiDate_3','Gregorian_BiopsiDate_4',
                      'Gregorian_BiopsiDate_5','Gregorian_BiopsiDate_6','idx','BiopsiDate','SurgeryDate','Gregorian_SurgeryDate'])

df = df.drop(columns = ['Gregorian_SurgeryDate_1','Gregorian_SurgeryDate_2','Gregorian_SurgeryDate_3','Gregorian_SurgeryDate_4',
                                                         'Gregorian_SurgeryDate_5', 'Gregorian_SurgeryDate_6','idx_'])
df.isnull().sum()

df.rename(columns={

'NearestsurgeryDate1' : 'First Surgery Date',
'NearestsurgeryDate2': 'Second Surgery Date',
'NearestBiopsiDate1': 'First Biopsi Date',
'NearestBiopsiDate2': 'Second Biopsi Date',
}, inplace=True)

df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/BiosurDates.xlsx', index=False)