import numpy as np
import pandas as pd
from datetime import datetime, date

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score

import pickle

def load_and_process(file):
    data = pd.read_csv(file, index_col=0)
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = datetime.now().year - data['dob'].dt.year
    data['daydate_transaction'] = data['trans_date_trans_time'].dt.day
    data['dayweek_transaction'] = data['trans_date_trans_time'].dt.dayofweek
    data['month_transaction'] = data['trans_date_trans_time'].dt.month
    data['hour_transaction'] = data['trans_date_trans_time'].dt.hour
    data['is_male'] = data['gender'].replace({'F': 0, 'M': 1})
    return data


def feature_engineering(data):
    df = data.copy()
    df = df.drop(['trans_date_trans_time', 'cc_num', 'merchant',
                  'first', 'last', 'street', 'zip', 'lat', 'long',
                  'job', 'dob', 'trans_num', 'unix_time', 'merch_lat',
                  'merch_long', 'city', 'gender', 'city_pop'], axis=1)
    num_df = df.select_dtypes(exclude='object')
    obj_df = df.select_dtypes(include='object')
    obj_df_ohe = pd.get_dummies(obj_df, prefix='ohe', prefix_sep='_', dummy_na=False, sparse=False, drop_first=True)
    df_final = pd.concat([num_df, obj_df_ohe], axis=1)
    return df_final


def split_and_train(df_final):
    total_frauds = df_final[df_final['is_fraud'] == 1].shape[0]
    fraud_indices = np.array(df_final[df_final['is_fraud'] == 1].index)

    normal_indices = df_final[df_final['is_fraud'] == 0].index
    random_normal_indices = np.random.choice(normal_indices, total_frauds, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices], axis=0)

    # Under sample dataset
    df_under_sample = df_final.iloc[under_sample_indices, :]

    X_us = df_under_sample.iloc[:, df_under_sample.columns != 'is_fraud']
    y_us = df_under_sample.iloc[:, df_under_sample.columns == 'is_fraud']

    X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_us,
                                                                    y_us,
                                                                    test_size=0.3,
                                                                    random_state=0)
    rf = RandomForestClassifier()
    rf.fit(X_train_us, y_train_us)

    pickle.dump(rf, open('clfmodel_rf.pkl', 'wb'))


file = 'fraud.csv'

loaded_and_processed = load_and_process(file)
print('Loaded')
engineered = feature_engineering(loaded_and_processed)
print('Engineered')
split_and_train(engineered)
print('Trained and Saved Model')