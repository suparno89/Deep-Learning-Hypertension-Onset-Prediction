import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
pd.set_option('display.max_columns', 500)

filename = 'ML_Dataset_A.parquet'
outputfilename = 'ML_dataset_A_180.pkl'
ref_filename = 'ML_dataset_365.pkl'

def get_train_test(df_main,df_ref):
    df_main = df_main.merge(df_ref[['medical_record_number', 'train_test']], how='inner', on = 'medical_record_number')
    return df_main

if __name__ == "__main__":
    df_main =  pq.read_table(os.path.join('../../hype_cohorts_ml/', filename)).to_pandas()
    df_ref = pd.read_pickle(os.path.join('../../hype_cohorts_ml/', ref_filename))
    df_main.reset_index(level=0, inplace=True)
    print("intital dataframe shape" + str(df_main.shape))
    print("intial case control distribution")
    print(df_main['HT'].value_counts())
    df_main = get_train_test(df_main,df_ref)
    print(df_main.shape)
    print(pd.crosstab(df_main.train_test, df_main.HT))
    print(df_main.head())
    df_main.to_pickle(os.path.join('../../hype_cohorts_ml/', outputfilename))