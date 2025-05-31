import pandas as pd
import numpy as np
import os
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



df = pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/FamilyHistory-allyears.xlsx', usecols=['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode'])

df_History = pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History.xlsx', usecols=['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode'])


commonlistofcolumns = ['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge']

df = pd.merge(df, df_History,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df_History2 = pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History2.xlsx', usecols=['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode'])

df = pd.merge(df, df_History2,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df_History1397 = pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History1397.xlsx', usecols=['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode'])

df = pd.merge(df, df_History1397,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df_History1401 = pd.read_excel('G:/Breast Cancer/Reza Ancology/FH/Family History1401.xlsx', usecols=['FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode'])

df = pd.merge(df, df_History1397,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df = df.dropna(axis = 0, how ='all') 
df = df.dropna(axis=1, how='all') 
columns_ = df.columns.tolist()
counts = df.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
df = df.drop(df.columns[to_del], axis=1)


df['FamilyHistoryTopographyName'] = df['FamilyHistoryTopographyName'].replace({
'breast' : 'Breast',
'Breast, NOS': 'Breast',
'BREAST'  :  'Breast',
"BRAIN": "Brain",
"UTERUS, NOS": "Uterus",
'Uterus, NOS': "Uterus",
'Corpus uteri': "Uterus",
"Breast": "Breast",
"Lung, NOS": "Lung",
"STOMACH": "Stomach",
"Intestinal tract, NOS": "Intestinal tract",
"Blood": "Blood",
"UNKNOWN PRIMARY SITE": "Unknown primary site",
"COLON": "Colon",
"ESOPHAGUS": "Esophagus",
'Abdominal esophagus': "Esophagus",
"BLADDER": "Bladder",
"Bone, NOS": "Bone",
'Bone of limb, NOS' : 'Bone',
"LARYNX": "Larynx",
"OVARY": "Ovary",
"LYMPH NODES": "Lymph nodes",
"PROSTATE GLAND": "Prostate gland",
"KIDNEY": "Kidney",
"GALLBLADDER": "Gallbladder",
"Liver": "Liver",
"Head of pancreas": "Head of pancreas",
"NASOPHARYNX": "Nasopharynx",
"THYROID GLAND": "Thyroid gland",
"Spleen": "Spleen",
"Skin, NOS": "Skin",
"Head, face or neck, NOS": "Head, face or neck",
"TESTIS": "Testis",
"PANCREAS": "Pancreas",
"SKIN": "Skin",
"Stomach, NOS": "Stomach",
"Cervix uteri": "Uterus",
'CERVIX UTERI' : 'Uterus',
"Bone marrow": "Bone",
"Pancreas, NOS": "Pancreas",
"CONNECTIVE, SUBCUTANEOUS AND OTHER SOFT TISSUES": "Connective, subcutaneous and other soft tissues",
"Esophagus, NOS": "Esophagus",
"Skin of lower limb and hip": "Skin of lower limb and hip",
"Abdomen, NOS": "Abdomen",
"VAGINA": "Vagina",
"Tongue, NOS": "Tongue",
"Lymph nodes of axilla or arm": "Lymph node",
"Spinal cord": "Spinal cord",
"Brain, NOS": "Brain",
"Eye, NOS": "Eye",
"Tonsil, NOS": "Tonsil",
"LIP": "Lip",
"External ear": "Ear",
"Rib, sternum, clavicle and associated joints": "Rib, sternum, clavicle and associated joints",
"Colon, NOS": "Colon",
'Sigmoid colon': "Colon",
"Mandible": "Mandible",
"RECTUM": "Rectum",
"TRACHEA": "Trachea",
"GUM": "Gum",
"Upper lobe, lung": "Lung",
"PALATE": "Palate",
"Abdominal esophagus": "Abdominal esophagus",
"OROPHARYNX": "Oropharynx",
"VULVA": "Vulva",
"Bladder, NOS": "Bladder",
"Lower lobe, lung": "Lung",
"CORPUS UTERI": "Uterus",
"PAROTID GLAND": "Parotid gland",
'PAROTID GLANID' : 'Parotid Glanid',
"Middle lobe, lung": "Lung",
"Mouth, NOS": "Mouth",
"BRONCHUS AND LUNG": "Lung",
"Lymph node, NOS": "Lymph node",
'Lymph nodes' : "Lymph node"
})


df['FamilyRelation'] = df['FamilyRelation'].replace({
'sister' : 'Sister',
'mother': 'Mother',
'father': "Father",
'brother': 'Brother',
'nephew': 'Nephew',
'neice': 'Neice',
"father's sister": "Aunt (father)",
"father's brother":"Uncle (father)",
"mother's sister":"Aunt (mother)",
"mother's brother":"Uncle (mother)",
'grandmother': 'Grandmother',
'grandfather':'Grandfather',
'GrandChild':'Grandchild',
'cousin (father)': 'Cousin (father)',
'cousin (mother)' :'Cousin (mother)',
"father's relatives":"Relative (father)",
"mother's relatives":"Relative (mother)",
"partner's relative" :"Relative",
'unknown':'Unknown',
})

df.loc[df['FamilyRelation'].isnull(), 'FamilyHistoryTopographyName'] = np.nan
df['FamilyRelation'].value_counts()

colum = ['FamilyAge']

for col in colum:  
  df = remove_outliers_iqr(df.copy(), col) 

df['FamilyAge'].describe()


df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['FamilyRelation'])

# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]

# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df.columns.tolist()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 
cols = list(set(df.columns)-{'FamilyHistoryTopographyName', 'FamilyRelation', 'FamilyAge', 'DocumentCode', 'idx'})


# Step 2: Add a counter for each ID's unique topography entries
df['TopographyNum'] = df.groupby('DocumentCode').cumcount() + 1

# Step 3: Pivot to create separate columns
pivoted_df = df.pivot(index='DocumentCode', columns='TopographyNum', 
                           values='FamilyHistoryTopographyName')

# Step 4: Clean up column names
pivoted_df.columns = [f'FamilyHistoryTopographyName_{i}' for i in pivoted_df.columns]
pivoted_df = pivoted_df.reset_index()

df = pd.merge(df, pivoted_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 



df['FamilyageNum'] = df.groupby('DocumentCode').cumcount() + 1

# Step 3: Pivot to create separate columns
pivoted_df = df.pivot(index='DocumentCode', columns='FamilyageNum', 
                           values='FamilyAge')

# Step 4: Clean up column names
pivoted_df.columns = [f'FamilyAge_{i}' for i in pivoted_df.columns]
pivoted_df = pivoted_df.reset_index()

df = pd.merge(df, pivoted_df,  on=['DocumentCode'], how ='left', suffixes =('','_df2')) 



df['Number of family'] = df[cols].apply(lambda row: row.dropna().nunique(), axis=1)
df['Number of family'].value_counts(dropna=False)
df.shape

df.drop(columns=['idx','TopographyNum','FamilyageNum'],inplace=True)

def process_family_relations_FD(df, relation_cols):
    """
    Process family relations and get corresponding topography and age data
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    relation_cols (list): List of columns containing family relations
    
    Returns:
    pd.DataFrame: Modified dataframe with new columns
    """
        
    special_values = {'Mother', 'Sister', 'Daughter'}
    special_values1 = {'Mother','Sister','Daughter','Father','Brother','Son'}        
                      
        
    # Initialize new columns
    df['PrimaryRelation'] = None
    df['SecondaryRelation'] = None
    df['ThirdRelation'] = None
    df['PrimaryTopography'] = None
    df['PrimaryAge'] = None    
    df['SecondaryTopography'] = None
    df['SecondaryAge'] = None
    df['ThirdTopography'] = None
    df['ThirdAge'] = None
    
    for idx, row in df.iterrows():
        # Process primary relations (special_values)
        for col in relation_cols:
            if row[col] in special_values1:
                df.at[idx, 'PrimaryRelation'] = row[col]
                # Find corresponding topography and age
                # Assuming topography and age columns follow the same numbering
                col_num = col.split('_')[-1]  # Get the number from column name
                topo_col = f'FamilyHistoryTopographyName_{col_num}'
                age_col = f'FamilyAge_{col_num}'
                
                if topo_col in df.columns:
                    df.at[idx, 'PrimaryTopography'] = row[topo_col]
                if age_col in df.columns:
                    df.at[idx, 'PrimaryAge'] = row[age_col]
                break
       
        # Process secondary relations (special_values2)
        for col in relation_cols:
            if row[col] in special_values1:
                if pd.isna(df.at[idx, 'PrimaryRelation']) or row[col] != df.at[idx, 'PrimaryRelation']:                    
                    df.at[idx, 'SecondaryRelation'] = row[col]
                    # Find corresponding topography and age
                    col_num = col.split('_')[-1]  # Get the number from column name
                    topo_col = f'FamilyHistoryTopographyName_{col_num}'
                    age_col = f'FamilyAge_{col_num}'
                    
                    if topo_col in df.columns:
                        df.at[idx, 'SecondaryTopography'] = row[topo_col]
                    if age_col in df.columns:
                        df.at[idx, 'SecondaryAge'] = row[age_col]
                    break
        
        # Process Third relations (special_values2)
        for col in relation_cols:
            if row[col] in special_values1:
                
                if pd.isna(df.at[idx, 'PrimaryRelation']) or row[col] != df.at[idx, 'PrimaryRelation']:
                    if row[col] != df.at[idx, 'SecondaryRelation']:
                        
                        df.at[idx, 'ThirdRelation'] = row[col]
                        # Find corresponding topography and age
                        col_num = col.split('_')[-1]  # Get the number from column name
                        topo_col = f'FamilyHistoryTopographyName_{col_num}'
                        age_col = f'FamilyAge_{col_num}'
                        
                        if topo_col in df.columns:
                            df.at[idx, 'ThirdTopography'] = row[topo_col]
                        if age_col in df.columns:
                            df.at[idx, 'ThirdAge'] = row[age_col]
                        break
                         

   
    return df

def process_family_relations_SD(df, relation_cols):
    """
    Process family relations and get corresponding topography and age data
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    relation_cols (list): List of columns containing family relations
    
    Returns:
    pd.DataFrame: Modified dataframe with new columns
    """
    special_values = {
        'Aunt (mother)', 'Uncle (mother)','Cousin (mother)', 'Cousin (father)', 
        'Aunt (father)', 'Uncle (father)', 'Neice', 'Grandmother', 'Grandfather',
        'Nephew', 'Grandchild'
    }
    
    # Define the relation levels we want to extract
    relation_levels = [
        'Fourth', 'Fifth',
        'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth',
        'Eleventh', 'Twelfth'
    ]
    
    # Initialize all relation columns
    for level in relation_levels:
        df[f'{level}Relation'] = None
        df[f'{level}Topography'] = None
        df[f'{level}Age'] = None

    for idx, row in df.iterrows():
        found_relations = set()
        
        for col in relation_cols:
            if row[col] in special_values and row[col] not in found_relations:
                # Find the next available relation level
                for level in relation_levels:
                    if pd.isna(df.at[idx, f'{level}Relation']):
                        # Store the relation
                        df.at[idx, f'{level}Relation'] = row[col]
                        found_relations.add(row[col])
                        
                        # Find corresponding topography and age
                        col_num = col.split('_')[-1]
                        topo_col = f'FamilyHistoryTopographyName_{col_num}'
                        age_col = f'FamilyAge_{col_num}'
                        
                        if topo_col in df.columns:
                            df.at[idx, f'{level}Topography'] = row[topo_col]
                        if age_col in df.columns:
                            df.at[idx, f'{level}Age'] = row[age_col]
                        break

    return df
   
# Get all relation columns
relation_cols = [col for col in df.columns if col.startswith('FamilyRelation_')]

def process_family_relations_TD(df, relation_cols):
    """
    Process family relations and get corresponding topography and age data
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    relation_cols (list): List of columns containing family relations
    
    Returns:
    pd.DataFrame: Modified dataframe with new columns
    """
    special_values = {
             
        'Relative (father)', 'Relative (mother)',
        'Relative', 'Unknown'
    }
    
    # Define the relation levels we want to extract
    relation_levels = [
        'Fourth', 'Fifth',
        'Sixth', 'Seventh', 
    ]
    
    # Initialize all relation columns
    for level in relation_levels:
        df[f'{level}ThirdRelation'] = None
        df[f'{level}ThirdTopography'] = None
        df[f'{level}ThirdAge'] = None

    for idx, row in df.iterrows():
        found_relations = set()
        
        for col in relation_cols:
            if row[col] in special_values and row[col] not in found_relations:
                # Find the next available relation level
                for level in relation_levels:
                    if pd.isna(df.at[idx, f'{level}ThirdRelation']):
                        # Store the relation
                        df.at[idx, f'{level}ThirdRelation'] = row[col]
                        found_relations.add(row[col])
                        
                        # Find corresponding topography and age
                        col_num = col.split('_')[-1]
                        topo_col = f'FamilyHistoryTopographyName_{col_num}'
                        age_col = f'FamilyAge_{col_num}'
                        
                        if topo_col in df.columns:
                            df.at[idx, f'{level}ThirdTopography'] = row[topo_col]
                        if age_col in df.columns:
                            df.at[idx, f'{level}ThirdAge'] = row[age_col]
                        break

    return df
   

df = process_family_relations_FD(df, relation_cols)
df = process_family_relations_SD(df, relation_cols)
df = process_family_relations_TD(df, relation_cols)



cols_ = list(set(df.columns) - {'DocumentCode','Number of family','PrimaryRelation', 'PrimaryTopography', 'PrimaryAge',
                                  'SecondaryRelation', 'SecondaryTopography', 'SecondaryAge',
                                  'ThirdRelation', 'ThirdTopography', 'ThirdAge',
                                  'FourthRelation', 'FourthTopography', 'FourthAge',
                                  'FifthRelation', 'FifthTopography', 'FifthAge',
                                  'SixthRelation', 'SixthTopography', 'SixthAge',
                                  'SeventhRelation', 'SeventhTopography', 'SeventhAge',
                                  'EighthRelation', 'EighthTopography', 'EighthAge', 
                                  'NinthRelation', 'NinthTopography', 'NinthAge',
                                  'TenthRelation', 'TenthTopography', 'TenthAge',
                                  'EleventhRelation', 'EleventhTopography', 'EleventhAge',
                                  'TwelfthRelation', 'TwelfthTopography', 'TwelfthAge',
                                  'FourthThirdRelation', 'FourthThirdTopography', 'FourthThirdAge',
                                  'FifthThirdRelation', 'FifthThirdTopography', 'FifthThirdAge',
                                  'SixthThirdRelation', 'SixthThirdTopography', 'SixthThirdAge',
                                  'SeventhThirdRelation', 'SeventhThirdTopography', 'SeventhThirdAge', 
                                 })


df.drop(columns=cols_,inplace=True)

def cascade_relations(df):
   
    original_df = df.copy()
    
    # First replacement: PrimaryRelation with SecondaryRelation
    mask_primary_na = df['PrimaryRelation'].isna() & df['SecondaryRelation'].isna()
    df.loc[mask_primary_na, 'PrimaryRelation'] = df.loc[mask_primary_na, 'SecondaryRelation']
    df.loc[mask_primary_na, 'PrimaryTopography'] = df.loc[mask_primary_na, 'SecondaryTopography']
    df.loc[mask_primary_na, 'PrimaryAge'] = df.loc[mask_primary_na, 'SecondaryAge']

    # Set Secondary to NA where we've used it
    df.loc[mask_primary_na, 'SecondaryRelation'] = np.nan
    df.loc[mask_primary_na, 'SecondaryTopography'] = np.nan
    df.loc[mask_primary_na, 'SecondaryAge'] = np.nan


    mask_secondary_needed =  df['PrimaryRelation'].isna() &  df['SecondaryRelation'].isna() & df['ThirdRelation'].notna()
    df.loc[mask_secondary_needed, 'PrimaryRelation'] = df.loc[mask_secondary_needed, 'ThirdRelation']
    df.loc[mask_secondary_needed, 'PrimaryTopography'] = df.loc[mask_secondary_needed, 'ThirdTopography']
    df.loc[mask_secondary_needed, 'PrimaryAge'] = df.loc[mask_secondary_needed, 'ThirdAge']
    
    # Set ThirdRelation to NA where we've used it
    df.loc[mask_secondary_needed, 'ThirdRelation'] = np.nan
    df.loc[mask_secondary_needed, 'ThirdTopography'] = np.nan
    df.loc[mask_secondary_needed, 'ThirdAge'] = np.nan
    

    # For rows where SecondaryRelation was also NA, use ThirdRelation directly
    mask_use_third = df['PrimaryRelation'].notna() & df['SecondaryRelation'].isna() & df['ThirdRelation'].notna()
    df.loc[mask_use_third, 'SecondaryRelation'] = df.loc[mask_use_third, 'ThirdRelation']
    df.loc[mask_secondary_needed, 'SecondaryTopography'] = df.loc[mask_secondary_needed, 'ThirdTopography']
    df.loc[mask_secondary_needed, 'SecondaryAge'] = df.loc[mask_secondary_needed, 'ThirdAge']
    
    # Clear ThirdRelation where we've used it
    df.loc[mask_use_third, 'ThirdRelation'] = np.nan
    df.loc[mask_use_third, 'ThirdTopography'] = np.nan
    df.loc[mask_use_third, 'ThirdAge'] = np.nan
    
    return df

df = cascade_relations(df)

 
df.rename(columns={

    'PrimaryRelation':'Primary Relation1', 
    'PrimaryTopography':'Primary Topography1', 
    'PrimaryAge' : 'Primary Age1',

    'SecondaryRelation': 'Primary Relation2', 
    'SecondaryTopography':'Primary Topography2', 
    'SecondaryAge': 'Primary Age2',

    'ThirdRelation': 'Primary Relation3', 
    'ThirdTopography':'Primary Topography3', 
    'ThirdAge': 'Primary Age3',

    'FourthRelation':'Secondary Relation1',
    'FourthTopography': 'Secondary Topography1', 
    'FourthAge' : 'Secondary Age1', 

    'FifthRelation':'Secondary Relation2',   
    'FifthTopography': 'Secondary Topography2',
    'FifthAge': 'Secondary Age2', 

    'SixthRelation':'Secondary Relation3',
    'SixthTopography': 'Secondary Topography3',
    'SixthAge': 'Secondary Age3', 

    'SeventhRelation':'Secondary Relation4', 
    'SeventhTopography': 'Secondary Topography4',
    'SeventhAge': 'Secondary Age4',

    'EighthRelation':'Secondary Relation5', 
    'EighthTopography': 'Secondary Topography5',
    'EighthAge': 'Secondary Age5',

    'NinthRelation':'Secondary Relation6', 
    'NinthTopography': 'Secondary Topography6',
    'NinthAge': 'Secondary Age6',

    'TenthRelation':'Secondary Relation7', 
    'TenthTopography': 'Secondary Topography7',
    'TenthAge': 'Secondary Age7',

    'EleventhRelation':'Secondary Relation8', 
    'EleventhTopography': 'Secondary Topography8',
    'EleventhAge': 'Secondary Age8',

    'TwelfthRelation':'Secondary Relation9', 
    'TwelfthTopography': 'Secondary Topography9',
    'TwelfthAge': 'Secondary Age9',
    
    'FourthThirdRelation' : 'Third Relation1', 
    'FourthThirdTopography': 'Third Topography1', 
    'FourthThirdAge': 'Third Age1', 
    'FifthThirdRelation': 'Third Relation2', 
    'FifthThirdTopography': 'Third Topography2',
    'FifthThirdAge': 'Third Age2', 
    'SixthThirdRelation': 'Third Relation3', 
    'SixthThirdTopography': 'Third Topography3', 
    'SixthThirdAge': 'Third Age3', 
    'SeventhThirdRelation': 'Third Relation4',      
    'SeventhThirdTopography': 'Third Topography4', 
    'SeventhThirdAge': 'Third Age4',

}, inplace=True)




def select_family_member_with_priority(df):
    """
    Selects family members with priority: Breast > Ovary 
    Checks primary relations first, then secondary relations if no match in primary
    
    Parameters:
    df (pd.DataFrame): Input dataframe with relation and topography columns
    
    Returns:
    pd.DataFrame: Dataframe with selected family member details
    """
    # Initialize output columns
    df['Family Relation'] = None
    df['Family Topography'] = None
    df['Family Age'] = None
    df['Family Degree'] = None  # 'Primary' or 'Secondary'
    
    # Define priority order
    priority_order = ['breast', 'ovary']
    
    for idx, row in df.iterrows():
        selected_case = None
        
        # First check primary relations (1-3)
        for i in range(1, 3):
            relation_col = f'Primary Relation{i}'
            topo_col = f'Primary Topography{i}'
            age_col = f'Primary Age{i}'
            
            if pd.notna(row[relation_col]) and pd.notna(row[topo_col]):
                topography = str(row[topo_col]).lower()
                for cancer_type in priority_order:
                    if cancer_type in topography:
                        # Found a match - check if higher priority than current selection
                        if selected_case is None or priority_order.index(cancer_type) < selected_case['priority']:
                            selected_case = {
                                'priority': priority_order.index(cancer_type),
                                'relation': row[relation_col],
                                'topography': row[topo_col],
                                'age': row[age_col],
                                'degree': 'Primary'
                            }
                        break  # Move to next relation once we've classified this one
        
        # If no primary match found, check secondary relations (1-4)
        if selected_case is None:
            for i in range(1, 5):
                relation_col = f'Secondary Relation{i}'
                topo_col = f'Secondary Topography{i}'
                age_col = f'Secondary Age{i}'
                
                if pd.notna(row[relation_col]) and pd.notna(row[topo_col]):
                    topography = str(row[topo_col]).lower()
                    for cancer_type in priority_order:
                        if cancer_type in topography:
                            if selected_case is None or priority_order.index(cancer_type) < selected_case['priority']:
                                selected_case = {
                                    'priority': priority_order.index(cancer_type),
                                    'relation': row[relation_col],
                                    'topography': row[topo_col],
                                    'age': row[age_col],
                                    'degree': 'Secondary'
                                }
                            break
        
        # Store the selected case if any was found
        if selected_case is not None:
            df.at[idx, 'Family Relation'] = selected_case['relation']
            df.at[idx, 'Family Topography'] = selected_case['topography']
            df.at[idx, 'Family Age'] = selected_case['age']
            df.at[idx, 'Family Degree'] = selected_case['degree']
    
    return df


df = select_family_member_with_priority(df)


def count_family_relations(df):
    """
    Counts the number of primary relations (First Degree) and secondary relations (Second Degree)
    and stores the counts in new columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing relation columns
    
    Returns:
    pd.DataFrame: Dataframe with added 'First Degree' and 'Second Degree' columns
    """
    # Initialize new columns
    df['Number of FD'] = 0
    df['Number of SD'] = 0
    df['Number of TD'] = 0
    
    # Count primary relations (Primary Relation1-3)
    for i in range(1, 4):
        relation_col = f'Primary Relation{i}'
        df['Number of FD'] += df[relation_col].notna().astype(int)
    
    # Count secondary relations (Secondary Relation1-4)
    for i in range(1, 5):
        relation_col = f'Secondary Relation{i}'
        df['Number of SD'] += df[relation_col].notna().astype(int)

    # Count secondary relations (Secondary Relation1-4)
    for i in range(1, 3):
        relation_col = f'Third Relation{i}'
        df['Number of TD'] += df[relation_col].notna().astype(int)    
    
    return df


df = count_family_relations(df)

df['Family Relation'].value_counts()

df.rename(columns={
'Number of family' : 'Number of Family',

},inplace = True)




new_column_order = [
    'DocumentCode', 
    'Number of Family', 
    'Number of FD', 
    'Number of SD',
    'Number of TD',
    'Primary Target',
    'Primary Other',
    'Secondary Target',
    'Secondary Other' ,
    'Family Relation',
    'Family Topography',
    'Family Age',
    'Primary Relation1', 
    'Primary Topography1', 
    'Primary Age1', 
    'Primary Relation2', 
    'Primary Topography2', 
    'Primary Age2', 
    'Primary Relation3', 
    'Primary Topography3', 
    'Primary Age3', 
    'Secondary Relation1',
    'Secondary Topography1', 
    'Secondary Age1', 
    'Secondary Relation2',    
    'Secondary Topography2', 
    'Secondary Age2', 
    'Secondary Relation3', 
    'Secondary Topography3', 
    'Secondary Age3',
    'Secondary Relation4',  
    'Secondary Topography4',
    'Secondary Age4',    
    'Secondary Relation5',  
    'Secondary Topography5',
    'Secondary Age5',    
    'Secondary Relation6',  
    'Secondary Topography6',
    'Secondary Age6',    
    'Secondary Relation7',  
    'Secondary Topography7',
    'Secondary Age7',    
    'Secondary Relation8',  
    'Secondary Topography8',
    'Secondary Age8',    
    'Secondary Relation9',  
    'Secondary Topography9',
    'Secondary Age9',
    'Third Relation1', 
    'Third Topography1', 
    'Third Age1', 
    'Third Relation2', 
    'Third Topography2',
    'Third Age2', 
    'Third Relation3', 
    'Third Topography3', 
    'Third Age3', 
    'Third Relation4',      
    'Third Topography4', 
    'Third Age4',
]

df = df[new_column_order]
df = df.dropna(axis = 0, how ='all') 
df = df.dropna(axis=1, how='all') 
columns_ = df.columns.tolist()
df.shape
# For each group, keep the row with the smallest number of missing values
def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]


df = (df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True))


df['Family Relation'] = df['Family Relation'].fillna(df['Primary Relation1'])
df['Family Topography'] = df['Family Topography'].fillna(df['Primary Topography1'])
df['Family Age'] = df['Family Age'].fillna(df['Primary Age1'])

df['Family Relation'] = df['Family Relation'].fillna(df['Primary Relation2'])
df['Family Topography'] = df['Family Topography'].fillna(df['Primary Topography2'])
df['Family Age'] = df['Family Age'].fillna(df['Primary Age2'])

df['Family Relation'] = df['Family Relation'].fillna(df['Primary Relation3'])
df['Family Topography'] = df['Family Topography'].fillna(df['Primary Topography3'])
df['Family Age'] = df['Family Age'].fillna(df['Primary Age3'])


df['Family Relation'] = df['Family Relation'].fillna(df['Secondary Relation1'])
df['Family Topography'] = df['Family Topography'].fillna(df['Secondary Topography1'])
df['Family Age'] = df['Family Age'].fillna(df['Secondary Age1'])


df['Family Relation'] = df['Family Relation'].fillna(df['Secondary Relation2'])
df['Family Topography'] = df['Family Topography'].fillna(df['Secondary Topography2'])
df['Family Age'] = df['Family Age'].fillna(df['Secondary Age2'])


df['Family Relation'] = df['Family Relation'].fillna(df['Secondary Relation3'])
df['Family Topography'] = df['Family Topography'].fillna(df['Secondary Topography3'])
df['Family Age'] = df['Family Age'].fillna(df['Secondary Age3'])

df['Family Relation'] = df['Family Relation'].fillna(df['Secondary Relation4'])
df['Family Topography'] = df['Family Topography'].fillna(df['Secondary Topography4'])
df['Family Age'] = df['Family Age'].fillna(df['Secondary Age4'])

df['Family Relation'] = df['Family Relation'].fillna(df['Secondary Relation5'])
df['Family Topography'] = df['Family Topography'].fillna(df['Secondary Topography5'])
df['Family Age'] = df['Family Age'].fillna(df['Secondary Age5'])

df['Family Relation'] = df['Family Relation'].fillna(df['Third Relation1'])
df['Family Topography'] = df['Family Topography'].fillna(df['Third Topography1'])
df['Family Age'] = df['Family Age'].fillna(df['Third Age1'])


df['Number of FD'].fillna(0, inplace=True)
df['Number of SD'].fillna(0, inplace=True)
df['Number of TD'].fillna(0, inplace=True)


df.rename(columns={
    'Primary Target' : 'FD Breast',
    'Primary Other': 'FD Other',
    'Secondary Target': 'SD Breast',
    'Secondary Other' : 'SD Other',

})




df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/FH.xlsx', index=False)




