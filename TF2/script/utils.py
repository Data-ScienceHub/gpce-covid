import pandas as pd
from pandas import DataFrame, to_datetime
import numpy as np
import os, gc
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append( '..' )
from Class.ParameterManager import ParameterManager

def scale_back(target, target_scaler, target_sequence_length):
    """
    if target was scaled, this inverse transforms the target.
    """
    if target_scaler is None:
        return target

    targets_inferred = target.shape[-1]
    target = target.reshape(-1, targets_inferred)

    upscaled = target_scaler.inverse_transform(target)

    # the square is done, since it was sqrt before doing the min max scaling in old code
    return np.square(upscaled.reshape((-1, target_sequence_length, targets_inferred)))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def SMAPE(y_true, y_pred):
    value = 2*abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))
    return np.mean(value)

def calculate_result(y_true, y_pred, split=False):

    mae = []
    rmse = []
    smape = []
    if split:
        for r in range(y_true.shape[-1]):
            mae.append(np.round(MAE(y_true[Ellipsis, r], y_pred[Ellipsis, r]), 4))
            rmse.append(np.round(RMSE(y_true[Ellipsis, r], y_pred[Ellipsis, r]), 4))
            smape.append(np.round(SMAPE(y_true[Ellipsis, r], y_pred[Ellipsis, r]), 4))
    else:
        mae.append(np.round(MAE(y_true, y_pred), 4))
        rmse.append(np.round(RMSE(y_true, y_pred), 4))
        smape.append(np.round(SMAPE(y_true, y_pred), 4))

    return mae, rmse, smape

def sumCases(y_true, y_preds, number_of_locations):
    """
    Sequences have overlapping days and thus there will be up to # prediction step predictions for a given day

    This function returns the average prediction for a day.
    """
    sequence, times, feat = y_true.shape
    dseq = int(sequence / number_of_locations)

    # Construct new matrix to store averages
    #   shape = (Location x TimeSteps x Features)

    TargetMatrix = np.zeros((number_of_locations, dseq + times - 1, feat))
    PredMatrix = np.zeros((number_of_locations, dseq + times - 1, feat))

    for f in range(feat):

        locCounter = 0
        TimeCounter = 0

        for seq in range(sequence):
            if seq != 0 and seq % dseq == 0:
                # Reset Time counter and increment locations
                locCounter += 1
                TimeCounter = 0

            for TimeStep in range(times):  # TimeStep goes from 0 to 14 (length = 15)
                TargetMatrix[locCounter, TimeCounter + TimeStep, f] = y_true[seq, TimeStep, f]
                PredMatrix[locCounter, TimeCounter + TimeStep, f] += y_preds[seq, TimeStep, f]

            TimeCounter += 1

        # Divide matrix chunk would be used if we would like to average predictions for a given day. Given
        # that we have overlapping sequences, we will also have overlapping predictions. Currently we take first

        # Divide Matrix ---> to incorporate this into the above code
        for idx, i in enumerate(TargetMatrix):
            for jdx, j in enumerate(i):
                if times-1 <= jdx <= TargetMatrix.shape[1] - times:
                    # TargetMatrix[idx,jdx] = np.divide(TargetMatrix[idx,jdx], times)
                    PredMatrix[idx, jdx, f] = np.divide(PredMatrix[idx, jdx, f], times)
                else:
                    divisor = min(abs(jdx+1), abs(TargetMatrix.shape[1]-jdx))
                    # TargetMatrix[idx,jdx] = np.divide(TargetMatrix[idx,jdx], divisor)
                    PredMatrix[idx, jdx, f] = np.divide(PredMatrix[idx, jdx, f], divisor)

    TargetMatrix = np.clip(TargetMatrix, 0, TargetMatrix.max() + 1)
    PredMatrix = np.clip(PredMatrix, 0, PredMatrix.max() + 1)

    return TargetMatrix, PredMatrix


def train_validation_test_split(df:DataFrame, parameterManager:ParameterManager, scale=True):
    if scale:
        df[parameterManager.target_column] = np.sqrt(df[parameterManager.target_column])

    train_start = parameterManager.train_start
    validation_start = parameterManager.validation_start
    test_start = parameterManager.test_start
    test_end = parameterManager.test_end

    selected_columns = [col for col in df.columns if col not in ['Date', 'Name']]

    train_data = df[(df['Date']>=train_start) & (df['Date']<validation_start)][selected_columns]
    validation_data = df[(df['Date'] >= validation_start) & (df['Date'] < test_start)][selected_columns]
    test_data = df[(df['Date']>=test_start)&(df['Date']<=test_end)][selected_columns]

    if not scale:
        return train_data, validation_data, test_data, None
    
    scaled_features = parameterManager.static_features + parameterManager.dynamic_features
    feature_scaler = MinMaxScaler()
    train_data.loc[:, scaled_features] = feature_scaler.fit_transform(train_data[scaled_features])
    validation_data.loc[:, scaled_features] = feature_scaler.transform(validation_data[scaled_features])
    test_data.loc[:, scaled_features] = feature_scaler.transform(test_data[scaled_features])
    del feature_scaler
    gc.collect()

    target_scaler = MinMaxScaler()
    target_column = parameterManager.target_column

    if len(target_column) == 1:
        train_data.loc[:, [target_column]] = target_scaler.fit_transform(train_data[[target_column]])
        validation_data.loc[:, [target_column]] = target_scaler.transform(validation_data[[target_column]])
        test_data.loc[:, [target_column]] = target_scaler.transform(test_data[[target_column]])
    else:
        train_data.loc[:, target_column] = target_scaler.fit_transform(train_data[target_column])
        validation_data.loc[:, target_column] = target_scaler.transform(validation_data[target_column])
        test_data.loc[:, target_column] = target_scaler.transform(test_data[target_column])

    return train_data, validation_data, test_data, target_scaler

def read_feature_file(dataPath, file_name):
    df = pd.read_csv(os.path.join(dataPath, file_name))
    # drop empty column names in the feature file
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def valid_date(date):
    try:
        pd.to_datetime(date)
        return True
    except:
        return False

def convert_cumulative_to_daily(original_df):
    df = original_df.copy()

    date_columns = [col for col in df.columns if valid_date(col)]
    df_advanced = df[date_columns].shift(periods=1, axis=1, fill_value=0)
    df[date_columns] -= df_advanced[date_columns]
    return df

def missing_percentage(df):
    return df.isnull().mean().round(4).mul(100).sort_values(ascending=False)

def fix_outliers(original_df, verbose=False):
    df = original_df.copy()

    date_columns = sorted([col for col in df.columns if valid_date(col)])
    for i in range(df.shape[0]):
        county_data = df.loc[i, date_columns]
        county_data[county_data<0] = 0

        median = np.percentile(county_data,50)
        q1 = np.percentile(county_data, 25) 
        q3 = np.percentile(county_data, 75)

        iqr = q3-q1
        upper_limit = q3 + 7.5*iqr
        lower_limit = q1 - 7.5*iqr
        global_outliers = ((county_data > upper_limit) | (county_data < lower_limit)) & (median > 0)
        
        # when no outliers found
        if sum(global_outliers) == 0:
            continue

        if verbose:
            print(f'FIPS {df.iloc[i, 0]}, outliers found {county_data[global_outliers].shape[0]}.')
        county_data[global_outliers] = median
        df.loc[i, date_columns] = county_data
    
    return df