from persiantools.jdatetime import JalaliDate
from datetime import datetime
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
 
df = pd.read_excel('G:/Breast Cancer/Reza Ancology/SY/Indication-allyears.xlsx' , usecols = ['DocumentCode', 'Morphology', 'Topography', 'TopographyName','MonthDuration', 'SymptomsName'])

df_symtoms = pd.read_excel('G:/Breast Cancer/Reza Ancology/SY/Indication.xlsx' , usecols = ['DocumentCode', 'MonthDuration', 'SymptomsName'])

commonlistofcolumns = ['MonthDuration', 'SymptomsName']
df2 = pd.merge(df, df_symtoms,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )
for col in commonlistofcolumns:
    df2[col] = df2[col].combine_first(df2[f'{col}_olddf'])

df2 = df2.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

commonlistofcolumns = ['SymptomsName']

df_symtoms1397 = pd.read_excel('G:/Breast Cancer/Reza Ancology/SY/Indication1397.xlsx' , usecols = ['DocumentCode', 'SymptomsName'])

df = pd.merge(df2, df_symtoms1397,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])


df_symtoms1401 = pd.read_excel('G:/Breast Cancer/Reza Ancology/SY/Indication1401.xlsx' , usecols = ['DocumentCode','SymptomsName'])

df = pd.merge(df, df_symtoms1401,  on=['DocumentCode'], how='outer', suffixes=('','_olddf') )

for col in commonlistofcolumns:
    df[col] = df[col].combine_first(df[f'{col}_olddf'])

df = df.drop(columns=[f'{col}_olddf' for col in commonlistofcolumns])

df.shape

df['Topography'] = df['Topography'].replace({
'C50.9--' : 'C50.9' ,
'C50.9-' : 'C50.9' 
})  
df['Morphology'] = df['Morphology'].replace({
'8500/3--' : '8500/3' ,
'8500/3-' : '8500/3' ,
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
df.shape
df['MonthDuration'].value_counts()

colum = ['MonthDuration']

for col in colum:  
  df = remove_outliers_iqr(df.copy(), col) 
 


columns_ = df.columns.tolist()


persian_to_english1 = {
    'Pain': 'Breast pain',
    '(ادم)Edema' : 'Edema',
    'تورم دست' : 'Edema',
    'hand edema': 'Edema',
    'axillar edema':'Edema',
    'تورم ودرد دست': 'Edema',
    "سفت شدن بازو":   'Edema',
    'Axillar Pain' : 'Edema',
    'چکاب' :'Screening',
    'چکاپ' :'Screening',
    'چكاپ' :'Screening',
    'Nipple Retraction(فرو رفتگي نوك پستان)' : 'Nipple change',
    'سفتی نیپل': 'Nipple change',
    'تورفتگی نیپل' : 'Nipple change',
    'خارش نیپل' : 'Nipple change',
    'ضخامت پوست نیپل' :'Nipple change',
    'احساس گز گزنیپل' : 'Nipple change',
    'nipple deformation': 'Nipple change',
    'Nipple Retraction' : 'Nipple change',
    'nipple redness' :'Nipple change',
    'زخم نیپل':'Nipple change',
    "ترشحات": 'Discharge' ,
    'خونریزی از نیپل':'Discharge',
    'ترشحات خونی از پستان': 'Discharge',
    'infected discharge' : 'Discharge',
    'Blood Discharge': 'Discharge',
    'Nipple Discharge' : 'Discharge',
    '(خونريزي)Bleeding':   'Discharge',
    '(خارش)Itching' : 'Skin change',
    'خارش' : 'Skin change',
    'skin nodule' : 'Skin change',
    'لکه پوستی' : 'Skin change',
    'Skin Change': 'Skin change',
    "تغییر رنگ پوست پستان": "Skin change",
    'زخم روی پوست سینه' :'Skin change',
    'تغییر حالت پوست':'Skin change',
    'زخم روی سینه':'Skin change',
    'Redness' : 'Skin change',
    'پوست پرتقالی' : 'Skin change',
    'فرورفتگی پوست پستان' : 'Skin change',
    'nipple ulcer' : 'Skin change',
    'ulcer' : 'Skin change',
    'pimple' : 'Skin change',
    'podorang' :'Skin change',
    '(آبسه)Abscess':'Skin change',
    'تغییر سایز پستان' :'Breast change',
    'سفت شدن سینه' : 'Breast change',
    'breast decrease' : 'Breast change',
    "ضخیم شدن عروق پستان" : 'Breast change',
    'breast deformation' : 'Breast change',
    "سنگینی سینه" : 'Breast change',
    'گرم شدن پستان' : 'Breast change',
    'سوزش سینه':'Breast change',
    'تیرکشیدن پستان' : 'Breast change',
    'درد وتورم پستان' : 'Breast pain',
    'دردپستان چپ': 'Breast pain',
    'تورم در ناحیه سینه':'Breast change',
    'Breast Retraction' : 'Breast change',
    'Breast Increase' : 'Breast change',
    'Hardness in breast' : 'Breast change',
    'سفت شدن پستان':'Breast change',
    'تورفتگی بافت پستان':'Breast change',
    'سفتی پستان': 'Breast change',
    'دردپستان':'Breast pain',
    'عدم تقارن در پستانها':'Breast change',
    'Lump' : 'Breast mass',
    'دردولمس توده':'Breast mass',
    'توده':'Breast mass',
    'lump increase'    :        'Breast mass',
    '(توده پستان)BREAST MASS':'Breast mass',
    'توده پستان' : 'Breast mass',
    'Lump in breast': 'Breast mass',
    'توده زیر بغل':'Breast mass',
    'Spot' : 'Axillary change',
    'سوزش زیر بغل': 'Axillary change',
    '(توده زير بغل)Axillary Mass': 'Breast mass',
    'pimple on axillary' : 'Skin change' ,    
    'تورم زیر بغل': 'Axillary change',
    'درد اگزیلاری':'Edema',   
    'درد زیر کتف': 'Axillary pain',
    'Axillar Retraction' :'Axillary change',
    'lump on neck' : 'Breast mass',
    'توده گردن':     'Breast mass',
    'استخوان درد': 'Bone pain',
    'درد تیر کشنده':'Breast pain',    
    'درد بدن': 'Body pain',
    'درد لگن': 'Pelvis pain',
    'دردستون فقرات': 'Spine pain',
    'دردکمر': 'Waist pain',
    'درد ستون فقرات': 'Spine pain',
    'درد پشت':'Spine pain',
    'دردگردن': 'Neck pain',
    'دردلگن' : 'Pelvis pain',
    'دردپشت' : 'Spine pain',
    'گردن درد' : 'Neck pain',
    'دردساق دست چپ' : 'Edema',
    'کمر درد' : 'Waist pain',
    'درد معده' : 'Stomach pain',
    'pelvic pain' : 'Pelvic pain',
    'گز گز دست(Tingling in hands)': "Edema",
    'گزگز پا(Tingling in feet)': "Leg pain",
    '(درد قفسه سينه)Chest Pain' : 'Chest Pain',
    'درد دست' : 'Edema',
    "درد شانه ودست": "Edema",
    'درد': "Edema",
    'دردقفسه سينه': "Chest Pain",
    'دردشانه' : 'Edema',
    'shoulder pain' : 'Edema',
    'درد پهلو' : 'Flank Pain',
    'Bone pain' : 'Bone pain',
    'Hand Pain' : 'Edema',
    'neck pain' : 'Neck pain',
    'body pain' : 'Body pain',
    'درد رادیکولر اندام فوقانی چپ' : 'Spine pain',
    'scapular pain': 'Spine pain',
    'Leg Pain' : 'Leg pain',
    'Flank Pain' : 'Flank Pain',
    'Flank Pain' : 'Flank Pain',
    'Leg Pain'   : 'Leg pain',
    "سوزش ناحیه ضایعه": "Pain",
    'توده سر' :'Head mass',
    'سردرد':'Head ache',
    'کاهش وزن': 'Weight Loss',
    'خونریزی واژینال': 'Vaginal Bleeding'   ,
    'سرگیجه': 'Vertigo',
    'تب': 'Fever',
    '(سرفه)Cough' : 'Cough'   ,
   '( خونريزي واژينال)Vaginal Bleeding':          'Vaginal Bleeding'   ,
    'درد معده': 'Stomach pain',
    'Backache':'Back ache',
    'گردن درد': 'Neck pain',
    'دردلگن': 'Pelvis pain',
    'تنگی نفس': 'Dyspnea',
    'دردلگن': 'Pelvis pain',
    'دردپشت': 'Spine pain',
    'Dyspnea': 'Dyspnea',
    'Headache': 'Head ache',
    "کمر درد" :'Waist pain', 
    'دردساق دست چپ' : 'Edema',
    'کمر درد': 'Waist pain',
    'توده سر': 'Head mass',
    '(استفراغ كردن)Vomiting' :'Vomiting',
    "عدم راه رفتن": "Cannot walk",
    'کاهش هوشیاری':'lethargy',
    'کاهش پلاکت' : 'platelet reduction',
    '(سرگيجه)Vertigo' : 'Vertigo',
    '(يبوست)Constipation' : 'Constipation',
    '(تشنج)Convulsion': 'Convulsion',
    'تشنج'  : 'Convulsion',
    '(تب)Fever': 'Fever',
    '(حالت تهوع)Nausea' : 'Nausea',
    '(سر درد)Headache' : 'Headache',
    'Heartburn(سوزش سر دل)':'Heartburn',
    '(ديسيوريا، سوزش ادرار)Dysuria': 'Dysuria',
    '(ديسپني، تنگي نفس)Dyspnea': 'Dyspnea',
    '(ديسيوريا، سوزش ادرار)Dysuria': 'Dysuria',
   '( خونريزي واژينال)Vaginal Bleeding' : 'Vaginal Bleeding',
    '(كاهش شنوايي)Hearing loss':'Hearing loss',
    'تب لرز' : 'Fever',
    'تهوع' : 'Nausea', 
    'دیسفاژی': 'Dysphagia',
    'بی حسی دست و پا': 'Numbness',
    'بی حسی دست وپا' :  'Numbness',
    'بی حالی': 'Fatigue',
    'بی حسی پاها': 'Numbness',
    'اختلال دید': 'Blurred vision',
    'تاری دید':  'Blurred vision',
    'دوبینی':    'Blurred vision',
    'blurred vision' : 'Blurred vision',
    'گزیدگی': 'Tingling',
    'تنگی نفس': 'Dyspnea',
    'بی اشتهایی': 'Loss of Appetite',
    'نابینایی': 'Blindness',
    'درد کشاله ران': 'Groin Pain',
    'فلج اندامها': 'Limb Paralysis',
    'عدم تعادل': 'Loss of Balance',
    'اختلالات هورمونی': 'Hormonal Disorders',
    'تورم صورت': 'Facial Swelling',
    'تب لرز':'Fever',    
    'یبوست': 'Constipation',
    '(زردي- يرقان)Icter' : 'Icter', 
    'Left Hemiparalysis':     'Left hemiparalysis' ,
    '(آمنزي-فراموشي)Amnesia' : 'Amnesia',
    
   }



# Replace Persian names with Englishnames
df[columns_] = df[columns_].replace(persian_to_english1)


# Add an incremental index within each group to pivot data later
df['idx'] = df.groupby('DocumentCode').cumcount() + 1

# Pivoting the table to get separate columns for each value of column1 and column2
result_df = df.pivot(index='DocumentCode', columns='idx', values=['SymptomsName'])

# Flatten the multi-index columns
result_df.columns = [f'{col}_{i}' for col, i in result_df.columns]

# Reset index to get 'id' as a column
result_df = result_df.reset_index()
df = pd.merge(df, result_df,  on=['DocumentCode'], how ='outer', suffixes =('','_df2')) 


def keep_row_with_smallest_missing_values(group):
    if len(group) == 1:
        return group
    else:
        missing_values_counts = group[columns_].isnull().sum(axis=1)
        min_missing_values_index = missing_values_counts.idxmin()
        return group.loc[[min_missing_values_index]]

df = df.groupby('DocumentCode', sort=False).apply(keep_row_with_smallest_missing_values).reset_index(drop=True)

cols = [ 'SymptomsName_1', 'SymptomsName_2', 'SymptomsName_3', 'SymptomsName_4', 'SymptomsName_5', 'SymptomsName_6', 'SymptomsName_7', 'SymptomsName_8',
        'SymptomsName_9', 'SymptomsName_10', 'SymptomsName_11', 'SymptomsName_12', 'SymptomsName_13', 'SymptomsName_14', 'SymptomsName_15',
        'SymptomsName_16', 'SymptomsName_17', 'SymptomsName_18', 'SymptomsName_19', 'SymptomsName_20', 'SymptomsName_21', 'SymptomsName_22',
        'SymptomsName_23', 'SymptomsName_24', 'SymptomsName_25', 'SymptomsName_26', 'SymptomsName_27', 'SymptomsName_28', 'SymptomsName_29',
        'SymptomsName_30', 'SymptomsName_31', 'SymptomsName_32', 'SymptomsName_33', 'SymptomsName_34', 'SymptomsName_35', 'SymptomsName_36',
        'SymptomsName_37', 'SymptomsName_38', 'SymptomsName_39', 'SymptomsName_40', 'SymptomsName_41', 'SymptomsName_42', 'SymptomsName_43', 
        'SymptomsName_44', 'SymptomsName_45', 'SymptomsName_46', 'SymptomsName_47', 
        'SymptomsName_48', 'SymptomsName_49']


special_values = {'Breast change', 'Axillary change', 'Skin change', 'Screening', 'Discharge', 'Edema','Nipple change'}


def find_symptoms(row):
    
    found_symptoms = list({row[col] for col in cols if row[col] in special_values})
    
    return found_symptoms[:3] if found_symptoms else [row['SymptomsName']]


symptom_columns = df.apply(find_symptoms, axis=1).apply(pd.Series)
symptom_columns.columns = ['Symptom1', 'Symptom2', 'Symptom3']

# Merge the new columns back into the original dataframe
df = pd.concat([df, symptom_columns], axis=1)


df.drop(columns=['idx','Morphology', 'Topography','TopographyName','SymptomsName'],inplace=True)
df.drop(columns=cols,inplace=True)
df.rename(columns={
'MonthDuration' : 'Time To Referral'}, inplace=True)

df.columns.tolist()
df['Symptom1'].value_counts()
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/SY.xlsx', index=False)
