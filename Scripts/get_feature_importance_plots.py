import xgboost
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
import inspect
import json
import sys
from pydoc import locate
#import fiber

import requests


from ml_pipeline import * ##get all the model config and utility functions from the other fine


##ML_dataset_180_no_outliers
#'ML_dataset_730.pkl'
#'ML_dataset_365.pkl'
#'ML_dataset_180.pkl'
persist = False
window = 730
input_filename = 'ML_dataset_730.pkl'
output_path = '../plots/'
output_filename = 'ML_dataset_730_renamed.pkl'

class FriendlyNamesConverter:
    def rename_columns(self, df):
        replacements = {}
        for column in df.columns:
            replacements[column] = self.get(column)
        return replacements

    def get(self, feature):
        # does not support time window information inside feature name yet
        if feature.startswith(('age', 'gender', 'religion', 'race')):
            return feature.replace('_', ' ').replace('.', '|')

        split_name = feature.split('__')
        if len(split_name) > 1: 
            if split_name[1] in [
                i[0]
                for i in inspect.getmembers(
                    sys.modules['fiber.condition'],
                    inspect.isclass
                )
            ]:
                aggregation = split_name[0]
                split_name = split_name[1:]
            else:
                aggregation = None

            if len(split_name) == 3:
                class_name, context, code = split_name
                condition_class = locate(f'fiber.condition.{class_name}')
                description = self.get_description(condition_class, code, context)
                if  "Lipid panel" in description:
                    description = "Lipid panel"
            else:
                class_name, description = split_name

            if aggregation is not None: 
                return f'{class_name} | {description.capitalize()} ({aggregation})'
            else:
                return f'{class_name} | {description.capitalize()}'
        else:
            return feature

    def get_description(self, condition_class, code, context):
        return condition_class(
            code=code,
            context=context
        ).patients_per(
            condition_class.description_column
        )[
            condition_class.description_column.name.lower()
        ].iloc[0]

def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    for transformer_in_columns in column_transformer.transformers_:#the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError: # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name


def get_shap_importanceplot(train_df, test_df):

    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int, 'uint8'] and c not in ['HT']]
    train_df['HT'] = pd.to_numeric(train_df['HT'])
    test_df['HT'] = pd.to_numeric(test_df['HT'])
    print("length of column names are ::" + str(len(categorical_cols) + len(numerical_cols)))

    column_transformer = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols),])
    
    lgb_classifier = LGBMClassifier(**lgb_param_calibrated)
    xgb_classifier = XGBClassifier(**xgb_param)

    retro_train = train_df[train_df.columns.difference(['HT'])]
    retro_label = train_df['HT']
    pros_train = test_df[test_df.columns.difference(['HT'])]
    pros_label = test_df['HT']

    
    preprocessed_data = column_transformer.fit_transform(retro_train, retro_label)
    column_names = get_column_names_from_ColumnTransformer(column_transformer)

    preprocessed_data = pd.DataFrame(preprocessed_data, columns = column_names)

    #lgb_classifier.fit(preprocessed_data, train_df['HT'])
    xgb_classifier.fit(preprocessed_data, train_df['HT'])
    #preprocessed_test_data = column_transformer.transform(pros_train)
    ##preprocessed_test_data = pd.DataFrame(preprocessed_test_data,column_names)
    shap_values = shap.TreeExplainer(xgb_classifier).shap_values(preprocessed_data)
    f = plt.figure()
    shap.summary_plot(shap_values, preprocessed_data, plot_type = 'dot')
    f.savefig(os.path.join(output_path, f'shap_explained_xgb_{window}.png'), bbox_inches='tight', dpi=600)


if __name__ == "__main__":

    if (not os.path.isfile(os.path.join('../../hype_cohorts_ml/', output_filename))) or (persist == True):
        print("renamed file not found, so creating one....")
        import fiber
        df =  pd.read_pickle(os.path.join('../../hype_cohorts_ml/', input_filename))
        df = df.drop(columns = ['patient_ethnic_group', 'deceased_indicator', 'religion'])
        rename_dict = FriendlyNamesConverter().rename_columns(df)
        print(rename_dict)
        print(df.shape)
        df = df.rename(columns=rename_dict, errors="raise")
        df = df.loc[:,~df.columns.duplicated()]
        print(df.shape)
        df.to_pickle(os.path.join('../../hype_cohorts_ml/', output_filename))
    
    else: 
        print("renamed file found, reading it from the drive....")
        df =  pd.read_pickle(os.path.join('../../hype_cohorts_ml/', output_filename))

    #Dataset_C['MRN'] = Dataset_C.index
    ##Dataset_C.reset_index(level=0, inplace=True)
    df = df.dropna(axis=0, thresh = NA_removal_threshold)
    #plt.plot(df['HT'])
    #plt.savefig('../plots/test2.png')
    df['race'] = np.where(df['race'] == "Ba", "Black American", df['race'])
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df['race'], prefix = 'Race', prefix_sep = ' | ').apply(pd.to_numeric, errors='coerce')], axis=1)
    df = df.drop(columns = ['Procedure | Msdw_not applicable'])
    print(df.shape)
    print("final dataframe shape after dropping NAs" + str(df.shape))
    print(pd.crosstab(df.train_test, df.HT))
    train_df, test_df = train_test_split(df, test_df_control_ratio) 
    get_shap_importanceplot(train_df,test_df)
    #print(train_df.shape)
    #get_ppca(train_df, test_df)
