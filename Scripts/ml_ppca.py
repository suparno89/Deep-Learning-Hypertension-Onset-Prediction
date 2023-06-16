import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import cross_val_predict, cross_validate, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from ppca import PPCA


xgb_param = {'silent': 0, 'n_jobs': -1, 'max_depth': 7, 'reg_alpha': 0.5, 'reg_lambda': 1, 'random_state': 42, 'learning_rate': 0.1,}
lgb_param = {'num_leaves':40, 'objective':'binary','max_depth':7,'learning_rate':0.1,'max_bin':128, 'metric': ['auc', 'binary_logloss'],
 'n_jobs':-1, 'reg_alpha': 0.5, 'reg_lambda': 1, 'random_state': 42,'feature_fraction': 0.25}
##ML_dataset_180_no_outliers
input_filename = 'ML_dataset_180.pkl'
output_filename = '../Intermediate_output/results.csv'
NA_removal_threshold = 80  ##atleast this many columns should have a non null value
prediction_window = 180
test_df_control_ratio = 2
THRESHOLD = 0.5
dimension = 20

def train_test_split(df, test_df_control):
    train_df = df[df.train_test == 'train']
    test_df = df[df.train_test == 'test']

    if test_df_control_ratio is not None:
        test_df_ht = test_df[test_df.HT == '1']
        test_df_not_ht = test_df[test_df.HT == '0']
        test_df_not_ht = test_df_not_ht.sample(test_df_control_ratio * test_df_ht.shape[0], random_state=42)
        test_df = pd.concat([test_df_ht,test_df_not_ht])

    train_df = train_df.drop(columns=['train_test', 'medical_record_number', 'address_zip'])
    test_df = test_df.drop(columns=['train_test', 'medical_record_number', 'address_zip'])
    return train_df, test_df
    
def evaluate_models(dict_models, train_df, test_df):

    results = []

    for key, value in dict_models.items():
        print("running model: "+ str(key))
        classifier = value
        dict_results = {}

        dict_results['model'] = key
        dict_results['prediction_window'] = prediction_window
        dict_results['test_df_control_ratio'] = test_df_control_ratio
        dict_results['NA_removal_threshold'] = NA_removal_threshold

        train_df['HT'] = pd.to_numeric(train_df['HT'])
        test_df['HT'] = pd.to_numeric(test_df['HT'])

        retro_auc = []
        retro_auprc = []
        retro_recall = []
        retro_precision = []
        retro_f1 = []
        retro_accuracy = []

        kf = KFold(n_splits=2,random_state = 42, shuffle = True)

        retro_train = train_df[train_df.columns.difference(['HT'])]
        retro_label = train_df['HT']

        categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
        numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int] and c not in ['HT']]
        ctransformer = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols)])
 
        ppca = PPCA()
        ppca.fit(data=ctransformer.fit_transform(train_df), d=dimension, verbose=False)
        transformed_train = ppca.transform()
        print("number of null values in transformed train")
        print(np.count_nonzero(np.isnan(transformed_train)))
        ppca = PPCA()
        ppca.fit(data = ctransformer.transform(test_df), d = dimension)
        transformed_test = ppca.transform()
        print("number of null values in transformed test")
        print(np.count_nonzero(np.isnan(transformed_test)))
        print(transformed_test.shape)

        for train_index, test_index in kf.split(transformed_train):           

            classifier.fit(transformed_train[train_index], retro_label.iloc[train_index])
            y_test_pred = np.where(classifier.predict_proba(transformed_train[test_index])[:, 1] > THRESHOLD, 1, 0)
            y_score = classifier.predict_proba(transformed_train[test_index])[:, 1]
            retro_auc.append(roc_auc_score(retro_label.iloc[test_index], y_score, average='micro'))
            retro_auprc.append(average_precision_score(retro_label.iloc[test_index], y_score, average='micro', pos_label = 1))
            retro_recall.append(recall_score(retro_label.iloc[test_index], y_test_pred))
            retro_precision.append(precision_score(retro_label.iloc[test_index], y_test_pred))
            retro_f1.append(f1_score(retro_label.iloc[test_index], y_test_pred, average='micro'))
            retro_accuracy.append(accuracy_score(retro_label.iloc[test_index], y_test_pred))

        ##scores = cross_validate(classifier, train_df[train_df.columns.difference(['HT'])], train_df['HT'], cv=2, n_jobs = -1, scoring=('roc_auc', 'average_precision','recall', 'precision', 'f1_micro', 'accuracy'))
        print("train cross val auc " + str(np.mean(retro_auc)))
        print("train cross val auprc " + str(np.mean(retro_auprc)))
        dict_results['train_auc'] = np.mean(retro_auc)
        dict_results['train_auprc'] = np.mean(retro_auprc)
        dict_results['train_recall'] = np.mean(retro_recall)
        dict_results['train_precision'] = np.mean(retro_precision)
        dict_results['train_f1'] = np.mean(retro_f1)
        dict_results['train_accuracy'] = np.mean(retro_accuracy)

        #print("Cross validation AUC {:.4f}".format(roc_auc_score(train_df['HT'], oof_pred[:,1])))
        #print("Cross validation AUPRC {:.4f}".format(average_precision_score(train_df['HT'], oof_pred[:,1], average='micro', pos_label = 1)))
        classifier.fit(transformed_train, train_df['HT'])
        #y_test_pred = classifier.predict(test_df[test_df.columns.difference(['HT'])])
        y_test_pred = np.where(classifier.predict_proba(transformed_test)[:, 1] > THRESHOLD, 1, 0)
        y_score = classifier.predict_proba(transformed_test)[:, 1]
        print("test auc " + str(roc_auc_score(test_df['HT'], y_score, average='micro')))
        print("test auprc " + str(average_precision_score(test_df['HT'], y_score, average='micro', pos_label = 1)))
        print("test f1 score  " + str(f1_score(test_df['HT'], y_test_pred, average='micro')))
        dict_results['test_auc'] = roc_auc_score(test_df['HT'], y_score, average='micro')
        dict_results['test_auprc'] = average_precision_score(test_df['HT'], y_score, average='micro', pos_label = 1)
        dict_results['test_recall'] = recall_score(test_df['HT'], y_test_pred)
        dict_results['test_precision'] = precision_score(test_df['HT'], y_test_pred)       
        dict_results['test_f1'] = f1_score(test_df['HT'], y_test_pred, average='micro')
        dict_results['test_accuracy'] = accuracy_score(test_df['HT'], y_test_pred)

        results.append(dict_results)
    return results
    

def model_params(train_df, test_df):

    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int] and c not in ['HT']]
    print("Number of categorical features " + str(len(categorical_cols)) + " and number of numerical features "+ str(len(numerical_cols)))
   
   
    classifier_lgb =  Pipeline([('lgbm',  LGBMClassifier(**lgb_param))])

    classifier_xgb =  Pipeline([('xgb',  XGBClassifier(**xgb_param))])
    classifier_lr =  Pipeline([('lr',  LogisticRegression(n_jobs = -1, random_state = 42, C = 0.1,  max_iter = 1000))])

    dict_models = { #'lgbm': classifier_lgb,
                    #'xgb':  classifier_xgb,
                    #'lgb_iterativeimpute': classifier_lgb_iterativeimpute, 
                    #'xgb_iterativeimpute' :classifier_xgb_iterativeimpute
                    'lr': classifier_lr
                   }

    results = evaluate_models(dict_models, train_df,test_df)
    return results


def get_ppca(train_df, test_df):

    train_df['HT'] = pd.to_numeric(train_df['HT'])
    test_df['HT'] = pd.to_numeric(test_df['HT'])
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int] and c not in ['HT']]
    ctransformer = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols)])
 
    ppca = PPCA()
    ppca.fit(data=ctransformer.fit_transform(train_df, train_df['HT']), d=10, verbose=True)
    transformed_train = ppca.transform()
    print(transformed_train.shape)
    print(np.count_nonzero(np.isnan(transformed_train)))
    #classifier_lr_ppca.set_params(ppca__d=10)
    #dict_models = {'lr_ppca': classifier_lr_ppca}
    
    #results = evaluate_models(dict_models, train_df,test_df)
    #return results
    


def write_file(df_results, filepath):
    if not os.path.isfile(filepath):
        print("output file doesn't exist, creating a new one...")
        df_results.to_csv(filepath, header=True, index = False)
    else: 
        print("output file exists, appending to the existing file...")
        df_results.to_csv(filepath, mode='a', header=False, index = False)


if __name__ == "__main__":
    df =  pd.read_pickle(os.path.join('../../hype_cohorts_ml/', input_filename))
    #Dataset_C['MRN'] = Dataset_C.index
    ##Dataset_C.reset_index(level=0, inplace=True)
    df = df.dropna(axis=0, thresh=NA_removal_threshold)
    print("final dataframe shape after dropping NAs" + str(df.shape))
    print(pd.crosstab(df.train_test, df.HT))
    train_df, test_df = train_test_split(df, test_df_control_ratio) 
    #get_ppca(train_df, test_df)
    results = model_params(train_df, test_df)
    df_results = pd.DataFrame(results)
    print(df_results.head())
    #write_file(df_results, output_filename)





""" with open('../Intermediate_output/columnnames_datasetC6months.txt', 'w') as f:
    for item in Dataset_C.columns:
        f.write("%s\n" % item)
 """




