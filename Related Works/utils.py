import numpy as np
from pandas import DataFrame, to_timedelta, concat
from typing import List
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import tensorflow as tf

# https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
def normalized_nash_sutcliffe_efficiency(y_true, y_pred):
    NSE = 1 - sum (np.square(y_true - y_pred) ) / sum( np.square(y_true - np.mean(y_true)) )
    return 1 / ( 2 - NSE)

# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.SMAPE.html?highlight=smape
def symmetric_mean_absolute_percentage(y_true, y_pred):
    numerator = 2*abs(y_true - y_pred)
    denominator = abs(y_true) + abs(y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[denominator == 0] = 0
    
    return np.mean(result)

def calculate_result(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    smape = symmetric_mean_absolute_percentage(y_true, y_pred)
    nnse = normalized_nash_sutcliffe_efficiency(y_true, y_pred)

    return mae, rmse, rmsle, smape, nnse

def show_result(df: DataFrame, targets:List[str]):    
    for target in targets:
        predicted_column = 'Predicted_'+ target
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, rmsle, smape, nnse = calculate_result(y_true, y_pred)
        print(f'Target {target}, MAE {mae:.5g}, RMSE {rmse:.5g}, RMSLE {rmsle:0.5g}, SMAPE {smape:0.5g}. NNSE {nnse:0.5g}.')
    print()

def split_data(
    df:DataFrame, split:dataclass, input_sequence_length:int
):
    train_df = df[(df['Date']>=split.train_start) & (df['Date']<split.validation_start)]

    validation_start = max(split.validation_start - to_timedelta(input_sequence_length, unit='day'), df['Date'].min())
    val_df = df[(df['Date']>=validation_start) & (df['Date']<split.test_start)]

    test_start = max(split.test_start - to_timedelta(input_sequence_length, unit='day'), df['Date'].min())
    test_df = df[(df['Date']>=test_start) & (df['Date']<=split.test_end)]

    print(f'Shapes: train {train_df.shape}, validation {val_df.shape}, test {test_df.shape}.')
    return train_df, val_df, test_df

def scale_data(
    train_df, val_df, test_df, features, targets
):
    feature_scaler = StandardScaler()
    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])

    target_scaler = StandardScaler()
    train_df[targets] = target_scaler.fit_transform(train_df[targets])
    val_df[targets] = target_scaler.transform(val_df[targets])
    test_df[targets] = target_scaler.transform(test_df[targets])

    return train_df, val_df, test_df, feature_scaler, target_scaler

def prepare_dataset(
    df:DataFrame, config:dataclass, 
    disable_progress_bar:bool=False
    ):
    data, labels = [], []
    assert df.shape[0] >= (config.input_sequence_length + config.output_sequence_length), f"Data size ({df.shape[0]}) too small for a complete sequence"

    for (_, county) in tqdm(df.groupby(config.group_id), disable=disable_progress_bar):
        feature_df = county[config.selected_columns]
        target_df = county[config.targets]

        for index in range(config.input_sequence_length, county.shape[0]-config.output_sequence_length+1):
            indices = range(index-config.input_sequence_length, index)
            data.append(feature_df.iloc[indices])
            
            indices = range(index, index + config.output_sequence_length)
            labels.append(target_df.iloc[indices])

    data = np.array(data).reshape((len(data), -1, len(config.selected_columns)))
    labels = np.array(labels).reshape((len(labels), -1))
    print(f'Shapes: data {data.shape}, labels {labels.shape}.')
    return data, labels

def cache_data(
    x, y, batch_size:int=64, buffer_size:int=None
):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if buffer_size is None:
        return data.cache().batch(batch_size)
    return data.cache().shuffle(buffer_size).batch(batch_size)

def process_prediction(
    target_df:DataFrame, y_pred:np.ndarray,
    config:dataclass
):
    counties = []
    prediction_counter = 0
    targets = config.targets
    group_id = config.group_id

    zeroes_df = DataFrame(np.zeros_like(target_df[targets]), columns=targets)
    zeroes_df[group_id] = target_df[group_id].values
    
    for (fips, county) in zeroes_df.groupby(group_id):
        df = county[config.targets].reset_index(drop=True)
        # keeps counter of how many times each index appeared in prediction
        indices_counter = np.zeros(df.shape[0])

        for index in range(config.input_sequence_length, df.shape[0]-config.output_sequence_length+1):
            indices = range(index, index + config.output_sequence_length)
            df.loc[indices] += y_pred[prediction_counter]
            indices_counter[indices] += 1
            prediction_counter += 1

        for index in range(config.input_sequence_length+1, df.shape[0]-1):
            if indices_counter[index] > 0:
                df.loc[index] /= indices_counter[index]

        df[group_id] = fips
        counties.append(df)

    prediction_df = concat(counties, axis=0).reset_index(drop=True)
    
    for target in targets:
        # target values here can not be negative
        prediction_df[prediction_df[target]<0] = 0
        # both case and death can only be an integer number
        prediction_df[target] = prediction_df[target].round() 

        prediction_df.rename(columns={target: f'Predicted_{target}'}, inplace=True)

    prediction_df.drop(group_id, axis=1, inplace=True)
    # now attach the predictions to the ground truth dataframe along column axis
    # need to better generalize for seriality (should join on date/time index too)
    prediction_df = concat([target_df, prediction_df], axis=1)

    # drop the input_sequence_length timesteps, since prediction starts after that
    prediction_start_date = prediction_df['Date'].min()+ to_timedelta(config.input_sequence_length + 1, unit='D')
    prediction_df = prediction_df[prediction_df['Date']>prediction_start_date]

    return prediction_df