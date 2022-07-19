import pandas as pd
from pandas import DataFrame, to_datetime
import numpy as np
from typing import List
import os, gc
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append( '..' )
from Class.Parameters import Parameters

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

def calculate_result(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    msle = mean_squared_log_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage(y_true, y_pred)
    nnse = normalized_nash_sutcliffe_efficiency(y_true, y_pred)

    return mae, rmse, msle, smape, nnse

def fix_dynamic_outliers(original_df, verbose=False):
    df = original_df.copy()

    date_columns = sorted([col for col in df.columns if valid_date(col)])
    for i in range(df.shape[0]):
        county_data = df.loc[i, date_columns]

        median = np.percentile(county_data,50)
        q1 = np.percentile(county_data, 25) 
        q3 = np.percentile(county_data, 75)
        county_data[county_data<0] =0

        iqr = q3-q1
        upper_limit = q3 + 7.5*iqr
        lower_limit = q1 - 7.5*iqr
        global_outliers = ((county_data > upper_limit) & (county_data < lower_limit))
        
        # when no outliers found
        if sum(global_outliers) == 0:
            continue

        if verbose:
            print(f'FIPS {df.iloc[i, 0]}, outliers found {county_data[global_outliers].shape[0]}.')
        county_data[global_outliers] = median
        df.loc[i, date_columns] = county_data
    
    return df

def global_outliers(dfc, fips):
    df = dfc.copy()
    for i in fips:
        median = np.percentile(df[i],50)
        q1 = np.percentile(df[i], 25) 
        q3 = np.percentile(df[i], 75)

        extreme = (df[i] < 0)
        # print(df[i][extreme])
        df[i][extreme] = 0

        iqr = q3-q1
        upper = q3 + 7.5*iqr
        lower = q1 - 7.5*iqr
        glob_out = ((df[i] > upper) & (df[i] < lower))

        df[i][glob_out] = median
    
    return df


def upscale_prediction(targets:List[str], predictions, target_scaler, target_sequence_length:int):
    """
    if target was scaled, this inverse transforms the target.
    """
    if target_scaler is None:
        return predictions

    df = pd.DataFrame({targets[i]: predictions[i].flatten() for i in range(len(targets))})
    df[targets] = target_scaler.inverse_transform(df[targets])

    return [df[target].values.reshape((-1, target_sequence_length)) for target in targets]

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
def normalized_nash_sutcliffe_efficiency(y_true, y_pred):
    NSE = 1 - sum (np.square(y_true - y_pred) ) / sum( np.square(y_true - np.mean(y_true)) )
    return 1 / ( 2 - NSE)

# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.SMAPE.html?highlight=smape
def symmetric_mean_absolute_percentage(y_true, y_pred):
    value = 2*abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))
    # for cases when both ground truth and predicted value are zero
    value = np.where(np.isnan(value), 0, value)
    
    return np.mean(value)

def get_start_dates(parameters: Parameters):
    max_encoder_length = parameters.model_parameters.input_sequence_length
    split = parameters.data.split

    train_start = split.train_start
    validation_start = split.validation_start
    test_start = split.test_start

    train_start = train_start + pd.to_timedelta(((split.train_end - train_start).days +1) % max_encoder_length, unit='D')
    validation_start = validation_start + pd.to_timedelta(((split.validation_end - validation_start).days+1) % max_encoder_length, unit='D')
    test_start = test_start + pd.to_timedelta(((split.test_end - test_start).days+1) % max_encoder_length, unit='D')

    print('Modifying start dates of the splits to adapt to encoder input sequence length')
    print(f'Start dates for train {train_start}, validation {validation_start}, test {test_start}')

    split.train_start = train_start
    split.validation_start = validation_start
    split.test_start = test_start
    parameters.data.split = split

    return train_start, validation_start, test_start

def train_validation_test_split(
    df:DataFrame, parameters:Parameters,
):
    split = parameters.data.split

    selected_columns = [col for col in df.columns if col not in ['Date', 'Name']]

    train_data = df[(df['Date']>=split.train_start) & (df['Date']<=split.train_end)][selected_columns]
    input_sequence_length = parameters.model_parameters.input_sequence_length
    
    # at least input_sequence_length prior days data is needed to start prediction
    # this ensures prediction starts from date validation_start. 
    earliest_validation_start = split.validation_start - pd.to_timedelta(input_sequence_length, unit='day')
    if earliest_validation_start in df['Date'].values:
        validation_data = df[(df['Date'] >= earliest_validation_start) & (df['Date'] <= split.validation_end)][selected_columns]
    else:
        validation_data = df[(df['Date'] >= split.validation_start) & (df['Date'] <= split.validation_end)][selected_columns]

    earliest_test_start = split.test_start - pd.to_timedelta(input_sequence_length, unit='day')
    if earliest_test_start in df['Date'].values:
        test_data = df[(df['Date'] >= earliest_test_start) & (df['Date'] <= split.test_end)][selected_columns]
    else:
        test_data = df[(df['Date'] >= split.test_start) & (df['Date'] <= split.test_end)][selected_columns]

    print(f'Train samples {train_data.shape[0]}, validation samples {validation_data.shape[0]}, test samples {test_data.shape[0]}')

    train_days = (split.train_end - split.train_start).days + 1
    validation_days = (split.validation_end - split.validation_start).days + 1
    test_days = (split.test_end - split.test_start).days + 1

    print(f'{train_days} days of training, {validation_days} days of validation data, {test_days} days of test data.')

    return train_data, validation_data, test_data

def train_test_split(
    df:DataFrame, parameters:Parameters,
):
    split = parameters.data.split

    selected_columns = [col for col in df.columns if col not in ['Date', 'Name']]

    train_data = df[(df['Date']>=split.train_start) & (df['Date']<=split.train_end)][selected_columns]
    train_data = train_data.sample(frac=1, random_state=parameters.model_parameters.seed) # randomly shuffle the train data, not required for others

    input_sequence_length = parameters.model_parameters.input_sequence_length
    
    # at least input_sequence_length prior days data is needed to start prediction
    # this ensures prediction starts from date test start. 
    earliest_test_start = split.test_start - pd.to_timedelta(input_sequence_length, unit='day')
    if earliest_test_start in df['Date'].values:
        test_data = df[(df['Date'] >= earliest_test_start) & (df['Date'] <= split.test_end)][selected_columns]
    else:
        test_data = df[(df['Date'] >= split.test_start) & (df['Date'] <= split.test_end)][selected_columns]

    print(f'Train samples {train_data.shape[0]}, test samples {test_data.shape[0]}')

    train_days = (split.train_end - split.train_start).days + 1
    test_days = (split.test_end - split.test_start).days + 1

    print(f'{train_days} days of training, {test_days} days of test data.')

    return train_data, test_data

def scale_data(
        train_data:DataFrame, validation_data:DataFrame, test_data:DataFrame, parameters:Parameters
    ):
    train, test = train_data.copy(), test_data.copy()
    if validation_data is None:
        validation = None
    else:
        validation = validation_data.copy()

    if parameters.preprocess.scale_input:
        scaled_features = parameters.data.static_features + parameters.data.dynamic_features

        print(f'Scaling static and dynamic input features: {scaled_features}')
        feature_scaler = MinMaxScaler()
        train.loc[:, scaled_features] = feature_scaler.fit_transform(train[scaled_features])
        if validation is not None:
            validation.loc[:, scaled_features] = feature_scaler.transform(validation[scaled_features])
        test.loc[:, scaled_features] = feature_scaler.transform(test[scaled_features])
        del feature_scaler
        gc.collect()

    target_scaler = None
    if parameters.preprocess.scale_target:
        target_features = parameters.data.targets
        print(f'Scaling targets {target_features}')

        target_scaler = MinMaxScaler()
        train.loc[:, target_features] = target_scaler.fit_transform(train[target_features])
        if validation is not None:
            validation.loc[:, target_features] = target_scaler.transform(validation[target_features])
        test.loc[:, target_features] = target_scaler.transform(test[target_features])
        
    return train, validation, test, target_scaler


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