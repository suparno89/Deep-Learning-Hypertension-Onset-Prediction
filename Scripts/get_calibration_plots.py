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
from ml_pipeline import *


##ML_dataset_180_no_outliers
#'ML_dataset_730.pkl'
#'ML_dataset_365.pkl'
#'ML_dataset_180.pkl'

window = 365
input_filename = 'ML_dataset_365.pkl'
output_path = '../plots/calibration_plots/'



def get_calibration_plots(train_df, test_df): 

    plt.rcParams.update({'font.size': 14})

    

    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int, 'uint8'] and c not in ['HT']]
    train_df['HT'] = pd.to_numeric(train_df['HT'])
    test_df['HT'] = pd.to_numeric(test_df['HT'])
    print("length of column names are ::" + str(len(categorical_cols) + len(numerical_cols)))

    column_transformer = Pipeline([('ct', ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols),])),
     ('simpleimputer' , SimpleImputer(missing_values=np.nan, strategy='mean'))])   

    retro_train = train_df[train_df.columns.difference(['HT'])]
    retro_label = train_df['HT']
    pros_train = test_df[test_df.columns.difference(['HT'])]
    pros_label = test_df['HT']

    preprocessed_data_retro = column_transformer.fit_transform(retro_train, retro_label)
    preprocessed_data_pros = column_transformer.transform(pros_train)

    lgb_classifier = LGBMClassifier(**lgb_param_calibrated)
    xgb_classifier = XGBClassifier(**xgb_param)
    lr_classifier = LogisticRegression(penalty = 'elasticnet', solver = 'saga', n_jobs = -1, random_state = 42, C = 0.0005,  max_iter = 2000, l1_ratio  = 0.5)

    
    for calibration in ['isotonic', 'sigmoid', 'uncalibrated']:
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for clf, name in [
                        (lr_classifier, f'Logistic {calibration}'),
                        (xgb_classifier, f'XGBoost {calibration}'),
                        (lgb_classifier, f'LighGBM {calibration}')]:

            if calibration != 'uncalibrated':
                isotonic = CalibratedClassifierCV(clf, cv=2, method=calibration)
            else: 
                isotonic = clf

            isotonic.fit(preprocessed_data_retro, retro_label)
            if hasattr(isotonic, "predict_proba"):
                prob_pos = isotonic.predict_proba(preprocessed_data_pros)[:, 1]
            else:  # use decision function
                prob_pos = isotonic.decision_function(preprocessed_data_pros)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(pros_label, prob_pos, n_bins=10)
            
            clf_score = brier_score_loss(pros_label, prob_pos, pos_label=1.0)


            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s (%1.3f)" % (name, clf_score))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                    histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'calibrated_{calibration}_{window}.png'), bbox_inches='tight', dpi=600)
        plt.show()



if __name__ == "__main__":

    df =  pd.read_pickle(os.path.join('../../hype_cohorts_ml/', input_filename))
    df = df.dropna(axis=0, thresh = NA_removal_threshold)
    print(df.shape)
    print("final dataframe shape after dropping NAs" + str(df.shape))
    print(pd.crosstab(df.train_test, df.HT))
    train_df, test_df = train_test_split(df, test_df_control_ratio) 
    get_calibration_plots(train_df,test_df)
    #print(train_df.shape)
    #get_ppca(train_df, test_df)