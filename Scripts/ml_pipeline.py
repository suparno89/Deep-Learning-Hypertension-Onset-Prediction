import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import cross_val_predict, cross_validate, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from fancyimpute import IterativeSVD, BiScaler
from ppca import PPCA
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import AUC
import tensorflow as tf

tf.random.set_seed(42)
pd.set_option('display.max_columns', 100)


xgb_param = {'silent': 0, 'n_jobs': -1, 'max_depth': 7, 'reg_alpha': 0.5, 'reg_lambda': 1, 'random_state': 42, 'learning_rate': 0.01, 'max_bin': 32,
    'colsample_bytree': 0.20}
lgb_param_calibrated = {'num_leaves':20, 'objective':'binary','max_depth':7,'learning_rate':0.01,'max_bin':32, 'metric': ['auc', 'binary_logloss'],
 'n_jobs':-1, 'reg_alpha': 0.5, 'reg_lambda': 1, 'random_state': 42,'feature_fraction': 0.18}
lgb_param = {'num_leaves':40, 'objective':'binary','max_depth':7,'learning_rate':0.01,'max_bin':128, 'metric': ['auc', 'binary_logloss'],
 'n_jobs':-1, 'reg_alpha': 0.5, 'reg_lambda': 1, 'random_state': 42,'feature_fraction': 0.25}

##ML_dataset_180_no_outliers
#'ML_dataset_730.pkl'
#'ML_dataset_365.pkl'
#'ML_dataset_180.pkl'
input_filename = 'ML_dataset_180.pkl'
output_filename = '../Intermediate_output/results-180.csv'
NA_removal_threshold = 80 ##atleast this many columns should have a non null value: 80 for unsupervised
prediction_window = 180
test_df_control_ratio = 2
THRESHOLD = 0.5

def twoLayerFeedForward():

    clf = Sequential()
    clf.add(Dense(100, input_dim=389, activation='relu'))
    clf.add(Dropout(0.3))
    clf.add(Dense(30, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))

    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr=1e-3), metrics=[AUC()])

    return clf



def train_test_split(df, test_df_control):
    train_df = df[df.train_test == 'train']
    test_df = df[df.train_test == 'test']

    if test_df_control_ratio is not None:
        test_df_ht = test_df[test_df.HT == '1']
        test_df_not_ht = test_df[test_df.HT == '0']
        test_df_not_ht = test_df_not_ht.sample(test_df_control_ratio * test_df_ht.shape[0], random_state=42)
        test_df = pd.concat([test_df_ht,test_df_not_ht])
    
    mrn_train = set(train_df['medical_record_number'])
    mrn_test = set(test_df['medical_record_number'])
    train_test_common = mrn_train.intersection(mrn_test)
    train_df = train_df.drop(columns=['train_test', 'medical_record_number', 'address_zip'], errors = 'ignore')
    test_df = test_df.drop(columns=['train_test', 'medical_record_number', 'address_zip'], errors = 'ignore')
    return train_df, test_df


def get_calibration_plots(y_test, y_score, name):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_pos = y_score
    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig('../Intermediate_output/calibration_{0}.pdf'.format(name))
    
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
        retro_brier = []

        kf = KFold(n_splits=2,random_state = 42, shuffle = True)

        retro_train = train_df[train_df.columns.difference(['HT'])]
        retro_label = train_df['HT']

        print(retro_train.shape)

        roc_best = 0
        best_classifier = None

        for train_index, test_index in kf.split(retro_train):

            classifier.fit(retro_train.iloc[train_index], retro_label.iloc[train_index])
            y_test_pred = np.where(classifier.predict_proba(retro_train.iloc[test_index])[:, 1] > THRESHOLD, 1, 0)
            y_score = classifier.predict_proba(retro_train.iloc[test_index])[:, 1]
            #get_calibration_plots(retro_label.iloc[test_index], y_score,(str(key)+'_train'))
            roc_auc = roc_auc_score(retro_label.iloc[test_index], y_score, average='micro')
            if (roc_auc > roc_best):
                best_classifier = classifier
            retro_auc.append(roc_auc_score(retro_label.iloc[test_index], y_score, average='micro'))
            retro_auprc.append(average_precision_score(retro_label.iloc[test_index], y_score, average='micro', pos_label = 1))
            retro_recall.append(recall_score(retro_label.iloc[test_index], y_test_pred))
            retro_precision.append(precision_score(retro_label.iloc[test_index], y_test_pred))
            retro_f1.append(f1_score(retro_label.iloc[test_index], y_test_pred, average='micro'))
            retro_accuracy.append(accuracy_score(retro_label.iloc[test_index], y_test_pred))
            retro_brier.append(brier_score_loss(retro_label.iloc[test_index], y_score, pos_label=1))

        ##scores = cross_validate(classifier, train_df[train_df.columns.difference(['HT'])], train_df['HT'], cv=2, n_jobs = -1, scoring=('roc_auc', 'average_precision','recall', 'precision', 'f1_micro', 'accuracy'))
        print("train cross val auc " + str(np.mean(retro_auc)))
        print("train cross val auprc " + str(np.mean(retro_auprc)))
        dict_results['retro_auc'] = np.mean(retro_auc)
        dict_results['retro_auprc'] = np.mean(retro_auprc)
        dict_results['retro_recall'] = np.mean(retro_recall)
        dict_results['retro_precision'] = np.mean(retro_precision)
        dict_results['retro_f1'] = np.mean(retro_f1)
        dict_results['retro_accuracy'] = np.mean(retro_accuracy)
        dict_results['retro_brier'] = np.mean(retro_brier)

        #print("Cross validation AUC {:.4f}".format(roc_auc_score(train_df['HT'], oof_pred[:,1])))
        #print("Cross validation AUPRC {:.4f}".format(average_precision_score(train_df['HT'], oof_pred[:,1], average='micro', pos_label = 1)))
        classifier = best_classifier
        #classifier.fit(train_df[train_df.columns.difference(['HT'])], train_df['HT'])
        #y_test_pred = classifier.predict(test_df[test_df.columns.difference(['HT'])])
        y_test_pred = np.where(classifier.predict_proba(test_df[test_df.columns.difference(['HT'])])[:, 1] > THRESHOLD, 1, 0)
        y_score = classifier.predict_proba(test_df[test_df.columns.difference(['HT'])])[:, 1]
        #get_calibration_plots(test_df['HT'], y_score,(str(key)+'_test'))
        print("test auc " + str(roc_auc_score(test_df['HT'], y_score, average='micro')))
        print("test auprc " + str(average_precision_score(test_df['HT'], y_score, average='micro', pos_label = 1)))
        print("test f1 score  " + str(f1_score(test_df['HT'], y_test_pred, average='micro')))
        dict_results['prospective_auc'] = roc_auc_score(test_df['HT'], y_score, average='micro')
        dict_results['prospective_auprc'] = average_precision_score(test_df['HT'], y_score, average='micro', pos_label = 1)
        dict_results['prospective_recall'] = recall_score(test_df['HT'], y_test_pred)
        dict_results['prospective_precision'] = precision_score(test_df['HT'], y_test_pred)       
        dict_results['prospective_f1'] = f1_score(test_df['HT'], y_test_pred, average='micro')
        dict_results['prospective_accuracy'] = accuracy_score(test_df['HT'], y_test_pred)
        dict_results['prospective_brier'] = brier_score_loss(test_df['HT'], y_score, pos_label=1)

        results.append(dict_results)
    return results
    

def model_params(train_df, test_df):

    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int] and c not in ['HT']]
    print("Number of categorical features " + str(len(categorical_cols)) + " and number of numerical features "+ str(len(numerical_cols)))
   
   
    classifier_lgb =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
        #('lgbm',  LGBMClassifier(**lgb_param_calibrated))
        ('lgbm-calibrated', CalibratedClassifierCV(base_estimator=LGBMClassifier(**lgb_param_calibrated), cv=2, method='isotonic'))
        ])

    classifier_xgb =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        #RFE(estimator=lgbm, n_features_to_select=50, step=10),
       #('xgb',  XGBClassifier(**xgb_param))
       ('xgb-calibrated', CalibratedClassifierCV(base_estimator=XGBClassifier(**xgb_param), cv=2, method='isotonic'))
       ])

    classifier_xgb_iterativeimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('iterativeimpute' , IterativeImputer(random_state=42, max_iter = 50, n_nearest_features = 5)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('xgb-calibrated', CalibratedClassifierCV(base_estimator=XGBClassifier(**xgb_param), cv=2, method='isotonic'))])

    classifier_lgb_iterativeimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('iterativeimpute' , IterativeImputer(random_state=42, max_iter = 50, n_nearest_features = 5)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('lgbm-calibrated', CalibratedClassifierCV(base_estimator=LGBMClassifier(**lgb_param_calibrated), cv=2, method='isotonic'))])
    
    classifier_lgb_knnimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('KNN' , KNNImputer(n_neighbors =2)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('lgbm-calibrated', CalibratedClassifierCV(base_estimator=LGBMClassifier(**lgb_param_calibrated), cv=2, method='isotonic'))])

    classifier_xgb_knnimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('KNN' , KNNImputer(n_neighbors =2)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('xgb-calibrated', CalibratedClassifierCV(base_estimator=XGBClassifier(**xgb_param), cv=2, method='isotonic'))])
    
    classifier_lr_knnimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('KNN' , KNNImputer(n_neighbors =2)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('lr',  LogisticRegression(penalty = 'elasticnet', solver = 'saga', n_jobs = -1, random_state = 42, C = 0.0005,  max_iter = 2000, l1_ratio  = 0.5))])
    
    
    classifier_lr_iter =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('iterativeimpute' , IterativeImputer(random_state=42, max_iter = 50, n_nearest_features = 5)),
        #('softimpute', IterativeSVD(max_iters = 500)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('lr',  LogisticRegression(penalty = 'elasticnet', solver = 'saga', n_jobs = -1, random_state = 42, C = 0.0005,  max_iter = 2000, l1_ratio  = 0.5))])

    classifier_lr_simpleimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('simpleimputer' , SimpleImputer(missing_values=np.nan, strategy='mean')),
        #('softimpute', IterativeSVD(max_iters = 500)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
       ('lr',  LogisticRegression(penalty = 'elasticnet', solver = 'saga', n_jobs = -1, random_state = 42, C = 0.0005,  max_iter = 2000, l1_ratio  = 0.5))])
    
    classifier_xgb_simpleimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('simpleimputer' , SimpleImputer(missing_values=np.nan, strategy='mean')),
        #('softimpute', IterativeSVD(max_iters = 500)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
        ('xgb-calibrated', CalibratedClassifierCV(base_estimator=XGBClassifier(**xgb_param), cv=2, method='isotonic'))])

    classifier_lgb_simpleimpute =  Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
        ('simpleimputer' , SimpleImputer(missing_values=np.nan, strategy='mean')),
        #('softimpute', IterativeSVD(max_iters = 500)),
        #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
        ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
        ('lgbm-calibrated', CalibratedClassifierCV(base_estimator=LGBMClassifier(**lgb_param_calibrated), cv=2, method='isotonic'))])


    classifier_nn_simpleimpute =  Pipeline([('ct',
            ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols),])),
            ('simpleimputer' , SimpleImputer(missing_values=np.nan, strategy='mean')),
            #('softimpute', IterativeSVD(max_iters = 500)),
            #('cat', OneHotEncoder(drop_invariant = True, handle_missing = 'return_nan'), categorical_cols),]),
            ##RFE(estimator=lgbm, n_features_to_select=50, step=10),
            ('nn', KerasClassifier(twoLayerFeedForward, epochs=100, batch_size=2048, verbose=1, shuffle = True,  validation_split = 0.2))])




    dict_models = { #'lgbm-isotonic': classifier_lgb,
                    #'xgb-isotonic':  classifier_xgb,
                    #'lgb-isotonic-iterativeimpute': classifier_lgb_iterativeimpute, 
                    #'classifier_lr_knnimpute': classifier_lr_knnimpute,
                    #'xgb-isotonic-iterativeimpute': classifier_xgb_iterativeimpute,
                    #'lr_iterativeimpute': classifier_lr_iter,
                    #'lr_simpleimpute_mean': classifier_lr_simpleimpute,
                    #'lgb-isotonic-simpleimpute_mean': classifier_lgb_simpleimpute,
                    #'xgb-isotonic-simpleimpute_mean': classifier_xgb_simpleimpute,
                    'nn-simpleimpute-mean'  : classifier_nn_simpleimpute
                   }

    

    results = evaluate_models(dict_models, train_df,test_df)
    return results


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
    df = df.dropna(axis=0, thresh = NA_removal_threshold)
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




