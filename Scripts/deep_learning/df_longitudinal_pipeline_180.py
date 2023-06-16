import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import concatenate,Dense
from keras.models import Model,load_model
from tensorflow.keras import regularizers

from pathlib import Path
from directories import *
import numpy as np
from parameters_config import Config
import matplotlib.pyplot as plt
import seaborn as sns

print(BASE_DIRECTORY.absolute())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
Config.VOCAB_SIZE=448

##set random seeds
np.random.seed(7)
tf.random.set_seed(seed=7)


NA_removal_threshold = 60  ##atleast this many columns should have a non null value
test_df_control_ratio = 2

BASE_PATH='/home/boses08/hype-prediction-longitudinal/ml'

# DATA_FILE_PATH="../../data/raw/"
DATA_FILE_PATH = "/home/boses08/hype-prediction-longitudinal/ml/data/raw_with_mrn/180/"
DATA_UNSUPERVISED_PATH = "/home/boses08/hype-prediction-longitudinal/ml/v2/prepare_multi_modal_data/data/diag_drug_proc_180.parquet"
MRN_PATH = "/home/boses08/hype-prediction-longitudinal/ml/v2/src/Suparno_Experiment/data/selected_MRNs/ML_dataset_180.pkl"

ATA_FILE_PATH_PROCESS="/home/boses08/hype-prediction-longitudinal/ml/v2/src/Suparno_Experiment/data/processed/"


def load_sequence_data(df_mrn):
    with open(DATA_FILE_PATH+"data.txt", "rb") as fp:
        X = pickle.load(fp)

    # with open(DATA_FILE_PATH+ "label.txt", "rb") as fp:
    #     y = pickle.load(fp)

    # df = pd.concat([pd.DataFrame(X, columns=['medical_record_number','sequence']), pd.DataFrame(y, columns=['target'])], axis=1)
    df = df_mrn.merge(pd.DataFrame(X, columns=['medical_record_number','sequence']),how='right',on='medical_record_number')

    return df[['medical_record_number', 'sequence', 'HT']]

    #return df

def load_unsupervised_data(path):
    df = pd.read_parquet(path)
#     df = pd.read_pickle(path)
    return df

def train_test_split_custom(df, test_df_control):
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
    train_df = train_df.drop(columns=['train_test', 'address_zip'])
    test_df = test_df.drop(columns=['train_test', 'address_zip'])
    return train_df, test_df

def filter_by_mrn(df,mrn_list):
    return df[df.medical_record_number.isin(mrn_list)]

def impute_unsupervised_data(train_df, test_df, train_labels_seq):
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.object] and c not in ['HT']]
    numerical_cols = [c for c in train_df.columns if train_df[c].dtype in [np.float, np.int] and c not in ['HT']]
    print("Number of categorical features " + str(len(categorical_cols)) + " and number of numerical features "+ str(len(numerical_cols)))
    
    ct = Pipeline([('ct',
        ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', TargetEncoder(drop_invariant = True, handle_missing = 'return_nan', min_samples_leaf = 10), categorical_cols)])),
        ('imputer' , SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0))])

    fitted_train= ct.fit(train_df,pd.to_numeric(train_labels_seq))
    
    train_df = fitted_train.transform(train_df)
    test_df = fitted_train.transform(test_df)
    
    return train_df,test_df

def train_lstm_model(train_sequence_data, train_unsupervised_data, train_labels):

    train_sequence_data, val_sequence_data, train_unsupervised_data, val_unsupervised_data, train_labels , val_labels = train_test_split(train_sequence_data, train_unsupervised_data, train_labels, test_size = 0.3, random_state = 42)

    print(train_sequence_data.shape)
    print(val_sequence_data.shape)
    print(train_labels.shape)
    print(val_labels.shape)

    main_input = keras.Input(shape=(train_sequence_data.shape[1],), name='main_input') # dtype='int32'

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = layers.Embedding(Config.VOCAB_SIZE, Config.EMBEDDING_DIM, input_length=Config.MAX_REVIEW_LENGTH, name='Embedding_1')(main_input)
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = layers.LSTM(100, name='lstm_1', dropout=0.5)(x)
    aux_input=keras.Input(shape=(train_unsupervised_data.shape[1],),name='aux_input')
    # We concatenate the lstm output to auxillary input
    x = concatenate([lstm_out, aux_input])
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, aux_input], outputs=[main_output])
    print(model.summary())
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=Config.METRICS)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        verbose=1,
        patience=20,
        mode='max',
        restore_best_weights=True)
    
    try:
        with tf.device('/device:GPU:2'):
            history = model.fit({'main_input': train_sequence_data, 'aux_input': train_unsupervised_data},
                {'main_output': train_labels},
                epochs=200,
                validation_data=([val_sequence_data,val_unsupervised_data],val_labels),
                batch_size=2048, verbose=1, callbacks=[early_stopping]) 
    except RuntimeError as e:
        print(e)
        
    
    val_predictions = model.predict([val_sequence_data,val_unsupervised_data])
    val_auprc = average_precision_score(val_labels, val_predictions, average='micro', pos_label = 1)
    val_auc = roc_auc_score(val_labels, val_predictions, average='micro')
    val_f1 = f1_score(val_labels, np.where(val_predictions > 0.5, 1, 0), average='micro')
    print("val AUPRC is :: " + str(val_auprc))
    print("val AUC is :: " + str(val_auc))
    print("val F1 is ::" + str(val_f1))


    
    epch = early_stopping.stopped_epoch

    validation_score = [history.history['val_loss'][epch],
        history.history['val_tp'][epch],
        history.history['val_fp'][epch],
        history.history['val_tn'][epch],
        history.history['val_fn'][epch],
        history.history['val_accuracy'][epch],
        history.history['val_precision'][epch],
        history.history['val_recall'][epch],
        history.history['val_auc'][epch],
        history.history['val_f1_score'][epch],
        history.history['val_average_precision'][epch]]
    
    print('Validation Score:',validation_score)

    return model


def evaluate_lstm_model(model, test_sequence_data, test_unsupervised_data, test_labels):
    results = model.evaluate([test_sequence_data,test_unsupervised_data], test_labels, batch_size=256, verbose=1)
    print(model.metrics_names)
    print(results)
    test_predictions = model.predict([test_sequence_data,test_unsupervised_data])
    test_auprc = average_precision_score(test_labels, test_predictions, average='micro', pos_label = 1)
    test_auc = roc_auc_score(test_labels, test_predictions, average='micro')
    test_f1 = f1_score(test_labels, np.where(test_predictions > 0.5, 1, 0), average='micro')

    precision, recall, thresholds = precision_recall_curve(test_labels, test_predictions)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f, Precision=%.3f, Recall=%.3f ' % (thresholds[ix], fscore[ix], precision[ix], recall[ix]))

    print("test AUPRC is :: " + str(test_auprc))
    print("test AUC is :: " + str(test_auc))
    print("test F1 is ::" + str(test_f1))


if __name__ == "__main__":

    ##load train test mrns
    df_mrn =  pd.read_pickle(MRN_PATH)

    print(pd.crosstab(df_mrn['train_test'], df_mrn['HT']))

    ##load sequence data
    sequence_data = load_sequence_data(df_mrn)
    
    df_mrn = df_mrn.dropna(axis=0, thresh = NA_removal_threshold)
    print("final dataframe shape after dropping NAs" + str(df_mrn.shape))
    MRN_train_df, MRN_test_df = train_test_split_custom(df_mrn, test_df_control_ratio)
    train_mrn = list(MRN_train_df.medical_record_number)
    test_mrn = list(MRN_test_df.medical_record_number)

    print(MRN_train_df['HT'].value_counts())
    print(MRN_test_df['HT'].value_counts())

    ##filter by mrns
    train_sequence_data = filter_by_mrn(sequence_data, train_mrn)
    test_sequence_data = filter_by_mrn(sequence_data, test_mrn)

    print(train_sequence_data['HT'].value_counts())
    print(test_sequence_data['HT'].value_counts())

    train_labels_seq = train_sequence_data.pop('HT').astype('int')
    test_labes_seq = test_sequence_data.pop('HT').astype('int')

    train_mrn_seq = train_sequence_data.pop('medical_record_number').astype('int')
    test_mrn_seq = test_sequence_data.pop('medical_record_number').astype('int')
    
    ##pad the sequences
    train_sequence_data = np.asarray(train_sequence_data['sequence'])
    test_sequence_data = np.asarray(test_sequence_data['sequence'])
    train_sequence_data = tf.keras.preprocessing.sequence.pad_sequences(train_sequence_data, maxlen=Config.MAX_REVIEW_LENGTH, padding='pre', truncating='pre')
    test_sequence_data = tf.keras.preprocessing.sequence.pad_sequences(test_sequence_data, maxlen=Config.MAX_REVIEW_LENGTH, padding='pre', truncating='pre')
    print('After padding the sequence with the longest length the shape is:',train_sequence_data.shape)
    print(train_sequence_data.max())
    print(test_sequence_data.max())

    ##load auxilliary data
    unsupervised_data = load_unsupervised_data(DATA_UNSUPERVISED_PATH)

    train_unsupervised_data = filter_by_mrn(unsupervised_data, train_mrn)
    test_unsupervised_data = filter_by_mrn(unsupervised_data, test_mrn)

    #train_labels = train_unsupervised_data.pop('HT').astype('int')
    #test_labels = test_unsupervised_data.pop('HT').astype('int')

    train_mrn_unsupervised = train_unsupervised_data.pop('medical_record_number').astype('int')
    test_mrn_unsupervised = test_unsupervised_data.pop('medical_record_number').astype('int')

    train_unsupervised_data.drop(['address_zip', 'mother_account_number'],inplace=True, axis = 1)
    test_unsupervised_data.drop(['address_zip', 'mother_account_number'],inplace=True, axis = 1)

    ##impute unsupervised data
    train_unsupervised_data, test_unsupervised_data = impute_unsupervised_data(train_unsupervised_data,test_unsupervised_data, train_labels_seq)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    ##sanity checks
    print("check if train labels are same")
    #assert np.array_equal(train_labels_seq.to_numpy(), train_labels.to_numpy())

    print("check if test labels are same")
    #assert np.array_equal(train_labels_seq.to_numpy(), train_labels.to_numpy())

    print("check if train mrns are same")
    assert np.array_equal(train_mrn_seq.to_numpy(), train_mrn_unsupervised.to_numpy())

    print("check if test mrns are same")
    assert np.array_equal(test_mrn_seq.to_numpy(), test_mrn_unsupervised.to_numpy())






    model = train_lstm_model(train_sequence_data, train_unsupervised_data, train_labels_seq)
    evaluate_lstm_model(model, test_sequence_data, test_unsupervised_data, test_labes_seq)