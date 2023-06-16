import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os

##Merged_Cases_Controls_730.parquet
##Merged_Cases_Controls_365.parquet
#ML_Dataset_A.parquet 
#ML_Dataset_B.parquet
filename = 'Merged_Cases_Controls_365.parquet'
outputfilename = 'ML_dataset_365.pkl'
prediction_window = 180


def clear_numeric_outliers(df_main):
    labvalue_cols = [col for col in df_main.columns if 'LabValue' in col]
    vital_cols = [col for col in df_main.columns if 'VitalSign' in col]
    numeric_cols = labvalue_cols + vital_cols
    for col in numeric_cols:
        q_low = df_main[col].quantile(0.001)
        q_hi  = df_main[col].quantile(0.999)
        df_main = df_main[(df_main[col].isnull()) | ((df_main[col] < q_hi) & (df_main[col] > q_low))]
    return df_main
    
    #print(df.head())

def remove_age(df_main):
    ##agefilter remove everyone above 90 and below 18
    df_main = df_main.drop(df_main[df_main.age_in_days/365 >= 90].index)
    df_main = df_main.drop(df_main[df_main.age_in_days/365 < 18].index)
    return df_main

def add_onset_year(df_main):
    df_main['date_of_birth'] = pd.to_datetime(df_main['date_of_birth'])
    df_main['year_of_birth'] = pd.DatetimeIndex(df_main['date_of_birth']).year
    df_main['date_of_birth_actual'] = pd.to_datetime(df_main.year_of_birth.astype(str) + '/' + df_main.month_of_birth.astype(str) + '/01')
    df_main['age_in_days_delta'] = pd.to_timedelta(df_main['age_in_days'],'d')
    df_main['onset_year'] = (df_main['date_of_birth_actual'] + df_main['age_in_days_delta']).dt.year
    return df_main

def split_train_test(df_main, test_year = 2018):

    df_main = df_main.drop(df_main[df_main.onset_year > 2018].index)
    df_main = df_main.drop(df_main[df_main.onset_year < 2003].index) ##no ehrs before 2013
    df_main['train_test'] = np.where(df_main.onset_year >= test_year, 'test', 'train')
    return df_main

if __name__ == "__main__":
    df_main =  pq.read_table(os.path.join('../../hype_cohorts_ml/', filename)).to_pandas()
    df_main.head()
    print("intital dataframe shape" + str(df_main.shape))
    print("intial case control distribution")
    print(df_main['HT'].value_counts())
    print(df_main.head(20))
    #df_main = clear_numeric_outliers(df_main)
    #print("dataframe shape after quantile correction" + str(df_main.shape))
    print(df_main['HT'].value_counts())
    print("number of null values in file " + str(df_main.isnull().sum().sum()))
    df_main = remove_age(df_main)
    print("case control distribution after removing age")
    print(df_main['HT'].value_counts())
    df_main = add_onset_year(df_main)
    ##remove onset years after 2018 and before 2003
    df_main = split_train_test(df_main, 2017)
    print("case control distribution after removing years before 2003 and after 2019")
    print(df_main['HT'].value_counts())
    ##add age at onset
    df_main['age_at_onset'] = df_main['age_in_days'] - prediction_window
    df_main.drop(columns=['date_of_birth', 'date_of_birth_actual', 'month_of_birth', 'year_of_birth', 'age_in_days', 'onset_year', 'age_in_days_delta'], inplace = True)
    print(df_main['train_test'].value_counts())
    print(pd.crosstab(df_main.train_test, df_main.HT))
    print("final dataframe shape before dropping NAs" + str(df_main.shape))
    print(df_main.head())
    #df_main.to_pickle(os.path.join('../../hype_cohorts_ml/', outputfilename))
    df_main = df_main.dropna(axis=0, thresh=80)
    print("final dataframe shape after dropping NAs" + str(df_main.shape))
    print(pd.crosstab(df_main.train_test, df_main.HT))
    #df_main['year_of_birth'] =  df_main.date_of_birth.str.split('-').get(0)
    #print(df_main[['medical_record_number', 'date_of_birth', 'month_of_birth', 'age_in_days', 'year_of_birth']].head(50))
