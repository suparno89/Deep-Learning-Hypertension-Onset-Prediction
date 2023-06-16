# import fiber
# from fiber.cohort import Cohort
# from fiber.condition import Patient, MRNs
import pandas as pd

cases_controls_365 = pd.read_parquet("/home/boses08/hype-prediction-longitudinal/Parquets/Final/365/Merged_Cases_Controls_365.parquet")
df = cases_controls_365[['medical_record_number','age_in_days','gender','HT']]
df.medical_record_number = pd.to_numeric(df.medical_record_number)

def Show_Gender_Stats(current_df,cohort_type):
    Cases = current_df.loc[current_df.HT == '1']
    Controls = current_df.loc[current_df.HT == '0']
    Cases_gender = Cases.gender.value_counts()
    Controls_gender = Controls.gender.value_counts()
    Cases_age_list = Cases.age_in_days
    Controls_age_list = Controls.age_in_days
    print('\n Cohort type: ',cohort_type,'\n')
    print('Cases:\n')
    print('\nGender:\n')
    for index,value in Cases_gender.items():
        print(index,':',value)
    print('\nAvg. age at onset in days: {0}'.format(Cases_age_list.mean()))
    print('\nAvg. age at onset in years: {0}\n'.format(Cases_age_list.mean()/365))
    print('\nStd. of age at onset: {0}\n'.format(Cases_age_list.std()))
    
    print('\nControls:\n')
    print('\nGender:\n')
    for index,value in Controls_gender.items():
        print(index,':',value)
    print('\nAvg. age at onset in days: {0}'.format(Controls_age_list.mean()))
    print('\nAvg. age at onset in years: {0}\n'.format(Controls_age_list.mean()/365))
    print('\nStd. of age at onset: {0}\n'.format(Controls_age_list.std()))



train_mrn_df = pd.read_csv("/home/boses08/hype_prediction_ehr/Scripts/deep_learning/Train_data.csv")
train_mrn = list(train_mrn_df.medical_record_number)

train_df = df.loc[df.medical_record_number.isin(train_mrn)]

Show_Gender_Stats(train_df,"Retrospective Cohort")

test_mrn_df = pd.read_csv("/home/boses08/hype_prediction_ehr/Scripts/deep_learning/Test_data.csv")
test_mrn = list(test_mrn_df.medical_record_number)

test_df = df.loc[df.medical_record_number.isin(test_mrn)]

Show_Gender_Stats(test_df,"Prospective Cohort")

# print(test_df.shape)








