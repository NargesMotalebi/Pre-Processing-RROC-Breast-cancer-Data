from persiantools.jdatetime import JalaliDate
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import csv

df_Indication = pd.read_excel('G:/Breast Cancer/Reza Ancology/Indication.xlsx')
df_Indication[df_Indication['Topography']] == 'C50.9' 
df_Indication1 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Indication1397.xlsx')
df_Indication[df_Indication['Topography']] == 'C50.9' 
df_Indication_ = pd.concat([df_Indication, df_Indication1], ignore_index=True) 
df_Indication2 = pd.read_excel('G:/Breast Cancer/Reza Ancology/Indication1401.xlsx')
df = pd.concat([df_Indication_, df_Indication2], ignore_index=True) 

df.shape
df.columns.tolist()
# Remove rows with all value missing    
df = df.dropna(axis = 0, how ='all')
# Remove columns with all value missing    
df = df.dropna(axis = 1, how ='all')


# Define the list of columns to consider
columns_ = df.columns.tolist()

# Dictionary for Persian to English translation
persian_to_english1 = {
    "عدم راه رفتن": "Cannot walk",
    'کاهش هوشیاری':'lethargy',
    'کاهش پلاکت' : 'platelet reduction',
    '(خارش)Itching' : 'Itching',
    'خارش' : 'Itching',
    '(سرگيجه)Vertigo' : 'Vertigo',
    '(يبوست)Constipation' : 'Constipation',
    '(تشنج)Convulsion': 'Convulsion',
    'تشنج'  : 'Convulsion',
    '(تب)Fever': 'Fever',
    '(حالت تهوع)Nausea' : 'Nausea',
    '(سرفه)Cough' : 'Cough',
    '(سر درد)Headache' : 'Headache',
    'Heartburn(سوزش سر دل)':'Heartburn',
    'گز گز دست(Tingling in hands)':'Tingling in hands',
    'گزگز پا(Tingling in feet)':'Tingling in feet'}
# Dictionary for Persian to English translation
persian_to_english2 = {
    '(درد قفسه سينه)Chest Pain' : 'Pain',
    "درد شانه ودست": "Pain",
    'درد': "Pain",
    'دردقفسه سينه': "Pain",
    'درد وتورم پستان' : 'Pain',
    'درد دست' : 'Pain',
    'دردپستان چپ': 'Pain',
    'دردشانه' : 'Pain',
    'shoulder pain' : 'Pain',
    'درد پهلو' : 'Pain',
    'Axillar Pain' : 'Pain',
    'Bone pain' : 'Pain',
    'Hand Pain' : 'Pain',
    'neck pain' : 'Pain',
    'body pain' : 'Pain',
    'درد رادیکولر اندام فوقانی چپ' : 'Pain',
    "سوزش ناحیه ضایعه": "Pain",
    '(ادم)Edema' : 'Edema',
    'تورم دست' : 'Edema',
    'hand edema': 'Edema',
    'axillar edema':'Edema',
    'چکاب' :'Screening',
    'چکاپ' :'Screening',
    'چكاپ' :'Screening',
   
}

# Dictionary for Persian to English translation
persian_to_english3 = {
 
    'Nipple Retraction(فرو رفتگي نوك پستان)' : 'Nipple change',
    'سفتی نیپل': 'Nipple change',
    'تورفتگی نیپل' : 'Nipple change',
    'خارش نیپل' : 'Nipple change',
    'احساس گز گزنیپل' : 'Nipple change',
    'nipple deformation': 'Nipple change',
    'Nipple Retraction' : 'Nipple change',
    'nipple redness' :'Nipple change',
    "ترشحات": 'Discharge' ,
    'ترشحات خونی از پستان': 'Discharge',
    'infected discharge' : 'Discharge',
    'Blood Discharge': 'Discharge',
    'Nipple Discharge' : 'Discharge',
    'skin nodule' : 'Skin change',
    'لکه پوستی' : 'Skin change',
    'Skin Change': 'Skin change',
    "تغییر رنگ پوست پستان": "Skin change",
    'زخم روی پوست سینه' :'Skin change',
    'تغییر حالت پوست':'Skin change',
    'Redness' : 'Skin change',
    'فرورفتگی پوست پستان' : 'Skin change',
    'nipple ulcer' : 'Skin change',
    'ulcer' : 'Skin change',
    'pimple' : 'Skin change',
    'تغییر سایز پستان' :'Breast change',
    'سفت شدن سینه' : 'Breast change',
    'breast decrease' : 'Breast change',
    "ضخیم شدن عروق پستان" : 'Breast change',
    'breast deformation' : 'Breast change',
    "سنگینی سینه" : 'Breast change',
    'گرم شدن پستان' : 'Breast change',
    'تورم در ناحیه سینه':'Breast change',
    'Breast Retraction' : 'Breast change',
    'تیرکشیدن پستان' : 'Breast change',
    'Breast Increase' : 'Breast change',
    'Hardness in breast' : 'Breast change',
    'Lump' : 'Breast mass',
    'دردولمس توده':'Breast mass',
    'توده':'Breast mass',
    '(توده پستان)BREAST MASS':'Breast mass',
    'توده پستان' : 'Breast mass',
    'Lump in breast': 'Breast mass',    
    'توده زیر بغل':'Axillary mass',
    'Spot' : 'Axillary mass',
    'سوزش زیر بغل': 'Axillary mass',
    '(توده زير بغل)Axillary Mass': 'Axillary mass',
    'pimple on axillary' : 'Axillary mass'    
}

df = df[~df['SymptomsName'].isin(['( خونريزي واژينال)Vaginal Bleeding',"کمر درد", 'دردساق دست چپ',
                                    "سفت شدن بازو", 'کمر درد','اختلالات هورمونی','تب لرز','تورم صورت',
                                    'درد معده','Backache','گردن درد','دردلگن','تنگی نفس','دردلگن','Tingling in feet',
                                    'دردپشت','توده گردن','توده سر','Dyspnea', 'گزیدگی','Headache','Cough',
                                    'Left Hemiparalysis','Nausea',])]

df['SymptomsName'].replace({
'Weight Loss'  :          np.nan   ,
'Itching' :               np.nan   ,
'Headache' :              np.nan   ,
'Nausea'  :               np.nan   ,
'Vertigo' :               np.nan   ,
'Cervical mass' :         np.nan   ,
'Heartburn'    :          np.nan   ,
'lethargy'  :             np.nan   ,
'Tingling in hands' :     np.nan   ,
'Cough' :                 np.nan  }, inplace=True)


# Replace Persian names with Englishnames
df[columns_] = df[columns_].replace(persian_to_english1)
df[columns_] = df[columns_].replace(persian_to_english2)
df[columns_] = df[columns_].replace(persian_to_english3)
def merge_duplicate_rows(df, id_column):
    # Function to merge rows
    def merge_rows(group):
        return group.ffill().bfill().iloc[0]

    # Group by ID and apply the merge function
    merged_df = df.groupby(id_column).apply(merge_rows).reset_index(drop=True)
    
    return merged_df
merged_df = merge_duplicate_rows(df, 'DocumentCode')

df = merged_df

df.shape
df.to_excel('G:/Breast Cancer/Reza Ancology/Edited/Indication.xlsx', index=False)