# %% [markdown]
# # Imports

# %%
import pandas as pd
# disable chained assignments
pd.options.mode.chained_assignment = None 
import numpy as np
import matplotlib.pyplot as plt
import os, gc
from pandas import to_timedelta

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Reshape
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 7
tf.random.set_seed(SEED)
SHOW_IMAGE = False
VERBOSE = 2

# %% [markdown]
# ## Plot configuration

# %%
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlepad=15)

# set tick width
plt.rcParams['xtick.major.size'] = 15 # default 3.5
plt.rcParams['xtick.major.width'] = 2 # default 0.8 

plt.rcParams['ytick.major.size'] = 15 # default 3.5
plt.rcParams['ytick.major.width'] = 2 # 0.8 

plt.rcParams['lines.linewidth'] = 2.5

DPI = 200
FIGSIZE = (12.5, 7)
DATE_TICKS = 5

# %% [markdown]
# ## Google Colab

# %%
# from google.colab import drive

# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Datasets

# %% [markdown]
# ## Result folder
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# %% [markdown]
# # Preprocessing

# %%
df = pd.read_csv('../TFT-pytorch/2022_May_cleaned/Total.csv')
df.head()

# %%
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class Split:
    train_start = pd.to_datetime("2020-02-29")
    validation_start = pd.to_datetime("2021-11-30")
    test_start = pd.to_datetime("2021-12-15")
    test_end = pd.to_datetime("2021-12-29")

# %%
features = ['AgeDist', 'HealthDisp', 'DiseaseSpread', 'Transmission', 'VaccinationFull', 'SocialDist', 'TimeFromStart', 'SinWeekly', 'CosWeekly']
targets = ['Cases']
group_id = 'FIPS'
selected_columns = features + targets
input_sequence_length = 13
output_sequence_length = 15
BATCH_SIZE = 128
BUFFER_SIZE = 1000
EPOCHS = 1
LEARNING_RATE = 1e-5
df['Date'] = pd.to_datetime(df['Date'])

# %% [markdown]
# ## Split and scale

# %%
def split_data(df):
    train_df = df[(df['Date']>=Split.train_start) & (df['Date']<Split.validation_start)]

    validation_start = max(Split.validation_start - to_timedelta(input_sequence_length, unit='day'), df['Date'].min())
    val_df = df[(df['Date']>=validation_start) & (df['Date']<Split.test_start)]

    test_start = max(Split.test_start - to_timedelta(input_sequence_length, unit='day'), df['Date'].min())
    test_df = df[(df['Date']>=test_start) & (df['Date']<=Split.test_end)]

    print(f'Shapes: train {train_df.shape}, validation {val_df.shape}, test {test_df.shape}.')
    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df, features, targets):
    feature_scaler = StandardScaler()
    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])

    target_scaler = StandardScaler()
    train_df[targets] = target_scaler.fit_transform(train_df[targets])
    val_df[targets] = target_scaler.transform(val_df[targets])
    test_df[targets] = target_scaler.transform(test_df[targets])

    return train_df, val_df, test_df, feature_scaler, target_scaler

# %%
train_df, val_df, test_df = split_data(df)
train_df, val_df, test_df, feature_scaler, target_scaler = scale_data(
    train_df, val_df, test_df, features, targets
)

# %% [markdown]
# ## Window generator

# %%
from tqdm import tqdm
def prepare_dataset(df, id=group_id):
    data, labels = [], []
    assert df.shape[0] >= (input_sequence_length+output_sequence_length), f"Data size ({df.shape[0]}) too small for a complete sequence"

    for (_, county) in df.groupby(id):
        feature_df = county[selected_columns]
        target_df = county[targets]

        for index in range(input_sequence_length, county.shape[0]-output_sequence_length+1):
            indices = range(index-input_sequence_length, index)
            data.append(feature_df.iloc[indices])
            
            indices = range(index, index + output_sequence_length)
            labels.append(target_df.iloc[indices])

    data = np.array(data).reshape((len(data), -1, len(selected_columns)))
    labels = np.array(labels).reshape((len(labels), -1))
    print(f'Shapes: data {data.shape}, labels {labels.shape}.')
    return data, labels

# %%
x_train, y_train = prepare_dataset(train_df)
x_val, y_val = prepare_dataset(val_df)
x_test, y_test = prepare_dataset(test_df)

# %% [markdown]
# ## Tensors

# %%
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.cache().batch(BATCH_SIZE)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.cache().batch(BATCH_SIZE)

# %% [markdown]
# # Training

# %% [markdown]
# ## Utils

# %%
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
loss_formatter = StrMethodFormatter('{x:,.3g}')

def plot_train_history(
    history, title, figure_name=None, 
    figsize=FIGSIZE, base=None
    ):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure(figsize=figsize)

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.gca().yaxis.set_major_formatter(loss_formatter)

    if base is not None:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=base))

    plt.title(title)
    plt.legend()

    if figure_name is not None:
        plt.savefig(os.path.join(output_folder, figure_name), dpi=200)

    if SHOW_IMAGE:
        plt.show()

# %% [markdown]
# ## Model

# %%
def build_LSTM(loss:str='mse', verbose:bool=False):
    model = Sequential([
        LSTM(64, return_sequences=True),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64),
        Dense((output_sequence_length * len(targets))),
        Reshape([output_sequence_length, len(targets)])
    ])
    if verbose:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=loss, optimizer=adam)
    return model

def build_BiLSTM(loss:str='mse', verbose:bool=False):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64),
        Dense((output_sequence_length * len(targets))),
        Reshape([output_sequence_length, len(targets)])
    ])
    if verbose:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=loss, optimizer=adam)
    return model

# %%
model = build_LSTM()
early_stopping = EarlyStopping(patience = 3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='model.h5', save_best_only=True, save_weights_only=True)

history = model.fit(
    train_data, epochs=EPOCHS, validation_data=val_data, 
    callbacks=[early_stopping, model_checkpoint], verbose=VERBOSE
)
gc.collect()
model.load_weights(model_checkpoint.filepath)

# %% [markdown]
# ## History

# %%
plot_train_history(
    history, 'Multi-Step, Multi-Output Training and Validation Loss', 'history.jpg'
)

# %% [markdown]
# # Prediction

# %% [markdown]
# ## Utils

# %% [markdown]
# ### Process prediction

# %%
def process_prediction(target_df, y_pred, id):
    counties = []
    prediction_counter = 0

    zeroes_df = pd.DataFrame(np.zeros_like(target_df[targets]), columns=targets)
    zeroes_df[group_id] = target_df[group_id].values
    
    for (fips, county) in zeroes_df.groupby(id):
        df = county[targets].reset_index(drop=True)
        # keeps counter of how many times each index appeared in prediction
        indices_counter = np.zeros(df.shape[0])

        for index in range(input_sequence_length, df.shape[0]-output_sequence_length+1):
            indices = range(index, index + output_sequence_length)
            df.loc[indices] += y_pred[prediction_counter]
            indices_counter[indices] += 1
            prediction_counter += 1

        for index in range(input_sequence_length+1, df.shape[0]-1):
            if indices_counter[index] > 0:
                df.loc[index] /= indices_counter[index]

        df[id] = fips
        counties.append(df)

    prediction_df = pd.concat(counties, axis=0).reset_index(drop=True)
    
    for target in targets:
        # target values here can not be negative
        prediction_df[prediction_df[target]<0] = 0
        # both case and death can only be an integer number
        prediction_df[target] = prediction_df[target].round() 

        prediction_df.rename(columns={target: f'Predicted_{target}'}, inplace=True)

    prediction_df.drop(group_id, axis=1, inplace=True)
    # now attach the predictions to the ground truth dataframe along column axis
    # need to better generalize for seriality (should join on date/time index too)
    prediction_df = pd.concat([target_df, prediction_df], axis=1)

    # drop the input_sequence_length timesteps, since prediction starts after that
    prediction_start_date = prediction_df['Date'].min()+ pd.to_timedelta(input_sequence_length + 1, unit='D')
    prediction_df = prediction_df[prediction_df['Date']>prediction_start_date]

    return prediction_df

# %% [markdown]
# ### Evaluation Metrics

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

# https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
def normalized_nash_sutcliffe_efficiency(y_true, y_pred):
    NSE = 1 - sum (np.square(y_true - y_pred) ) / sum( np.square(y_true - np.mean(y_true)) )
    return 1 / ( 2 - NSE)

def calculate_result(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    # smape = symmetric_mean_absolute_percentage(y_true, y_pred)
    nnse = normalized_nash_sutcliffe_efficiency(y_true, y_pred)

    return mae, rmse, rmsle, nnse

def show_result(df: pd.DataFrame, targets):    
    for target in targets:
        predicted_column = 'Predicted_'+ target
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, smape, nnse = calculate_result(y_true, y_pred)
        print(f'Target {target}, MAE {mae:.5g}, RMSE {rmse:.5g}, SMAPE {smape:0.5g}. NNSE {nnse:0.5g}.')
    print()

# %% [markdown]
# ### Plot prediction

# %%
def plot_predition(df, target, plot_error=True, figure_name=None, figsize=FIGSIZE):
    x_major_ticks = DATE_TICKS
    predicted = 'Predicted_'+ target

    # make sure to do this before the aggregation
    mae, rmse, rmsle, nnse = calculate_result(df[target].values, df[predicted].values)
    title = f'{target} MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, NNSE {nnse:0.3g}'

    df = df.groupby('Date')[
        [target, predicted]
    ].aggregate('sum').reset_index()
    
    _, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.plot(df['Date'], df[target], color='blue', label='Ground Truth')
    plt.plot(df['Date'], df[predicted], color='green', label='Predicted')
    if plot_error:
        plt.plot(df['Date'], abs(df[target] - df[predicted]), color='red', label='Error')
    ax.set_ylim(0, ax.get_ylim()[-1]*1.05)

    label_text, scale, unit = [], 1e3, 'K'
    for loc in plt.yticks()[0]:
        if loc == 0:
            label_text.append('0')
        else:
            label_text.append(f'{loc/scale:0.5g}{unit}')
        
    ax.set_yticks(plt.yticks()[0])
    ax.set_yticklabels(label_text)
    
    plt.ylabel(f'Daily {target}') 

    x_first_tick = df['Date'].min()
    x_last_tick = df['Date'].max()
    ax.set_xticks(
        [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
    )

    if plot_error:
        plt.legend(framealpha=0.3, edgecolor="black", ncol=3)
    else:
        plt.legend(framealpha=0.3, edgecolor="black", ncol=2)
    
    if figure_name is not None:
        plt.savefig(os.path.join(output_folder, figure_name), dpi=200)

    if SHOW_IMAGE:
        plt.show()

# %% [markdown]
# ## Train data

# %%
y_pred = model.predict(x_train, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = train_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
train_prediction_df = process_prediction(target_df, y_pred, group_id)
print(train_prediction_df.describe())

# %%
for target in targets:
    plot_predition(train_prediction_df, target, figure_name=f'Summed_{target}_Train.jpg')

# %% [markdown]
# ## Validation data

# %%
y_pred = model.predict(x_val, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = val_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
val_prediction_df = process_prediction(target_df, y_pred, group_id)
val_prediction_df.describe()

# %%
for target in targets:
    plot_predition(val_prediction_df, target, figure_name=f'Summed_{target}_Validation.jpg')

# %% [markdown]
# ## Test data

# %%
y_pred = model.predict(x_test, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = test_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
test_prediction_df = process_prediction(target_df, y_pred, group_id)
test_prediction_df.describe()

# %%
for target in targets:
    plot_predition(test_prediction_df, target, figure_name=f'Summed_{target}_Test.jpg')

# %% [markdown]
# ## Dump

# %%
train_prediction_df['Split'] = 'train'
val_prediction_df['Split'] = 'validation'
test_prediction_df['Split'] = 'test'
merged_df = pd.concat([train_prediction_df, val_prediction_df, test_prediction_df], axis=0)
merged_df.to_csv(os.path.join(output_folder, 'predictions.csv'), index=False)