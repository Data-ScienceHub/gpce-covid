# %% [markdown]
# # Imports

# %%
import pandas as pd
# disable chained assignments
pd.options.mode.chained_assignment = None 
import os

import tensorflow as tf
from datetime import datetime

from models import *
from plotter import *
from utils import *
from splits import *

SEED = 7
tf.random.set_seed(SEED)
SHOW_IMAGE = False
VERBOSE = 1
Split = Baseline

# %% [markdown]
# ## Result folder

# %%
start = datetime.now()
output_folder = 'scratch/results_LSTM'
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# %% [markdown]
# # Preprocessing

# %%
df = pd.read_csv('../TFT-pytorch/2022_May_cleaned/Top_100.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(df.head(3))

# %% [markdown]
# ## Config

# %%
@dataclass
class Config:
    static_features = ['AgeDist', 'HealthDisp']
    past_features = ['DiseaseSpread', 'Transmission', 'VaccinationFull', 'SocialDist']
    known_future = ['SinWeekly', 'CosWeekly']
    time_index = 'TimeFromStart' # note that this is an index feature commonly used by all timeseries models

    features =  [time_index] + static_features + past_features + known_future
    targets = ['Cases']
    group_id = 'FIPS'
    selected_columns = features + targets
    input_sequence_length = 13
    output_sequence_length = 15
    batch_size = 64
    buffer_size = 1000
    epochs = 1
    learning_rate = 1e-6
    early_stopping_patience = 5
    loss = 'mse'

targets = Config.targets
group_id = Config.group_id
input_sequence_length = Config.input_sequence_length
output_sequence_length = Config.output_sequence_length

# %% [markdown]
# ## Split and scale

# %%
train_df, val_df, test_df = split_data(df, Split, input_sequence_length)
train_df, val_df, test_df, feature_scaler, target_scaler = scale_data(
    train_df, val_df, test_df, Config.features, targets
)

# %% [markdown]
# ## Window generator

# %%
x_train, y_train = prepare_dataset(
    train_df, Config, disable_progress_bar=(VERBOSE!=1)
)
x_val, y_val = prepare_dataset(
    val_df, Config, disable_progress_bar=(VERBOSE!=1)
)
x_test, y_test = prepare_dataset(
    test_df, Config, disable_progress_bar=(VERBOSE!=1)
)

# %% [markdown]
# ## Tensors

# %%
train_data = cache_data(
    x_train, y_train, batch_size=Config.batch_size, 
    buffer_size=Config.buffer_size
)
val_data = cache_data(
    x_val, y_val, batch_size=Config.batch_size, 
)
test_data = cache_data(
    x_test, y_test, batch_size=Config.batch_size, 
)

# %% [markdown]
# # Training

# %% [markdown]
# ## Model

# %%
output_size = len(targets) * output_sequence_length
model = build_LSTM(
    x_train.shape[1:], output_size=output_size, loss=Config.loss, 
    summarize=False, learning_rate=Config.learning_rate
)

# %%
# print(f'Best model by validation loss saved at {model_checkpoint.filepath}.')
print(f'Loading best model.')
model.build(x_train.shape)
model.load_weights(os.path.join(output_folder, "model.h5"))

# %% [markdown]
# # Prediction

# %% [markdown]
# ## Train data

# %%
print('\nTrain prediction')
train_data = cache_data(
    x_train, y_train, batch_size=Config.batch_size, 
)
y_pred = model.predict(train_data, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = train_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
train_prediction_df = process_prediction(target_df, y_pred, Config)
print(train_prediction_df.describe())

# %%
show_result(train_prediction_df, targets)
for target in targets:
    plot_predition(
        train_prediction_df, target, show_image=SHOW_IMAGE, plot_error=True,
        figure_path=os.path.join(output_folder, f'Summed_{target}_Train.jpg')
    )

# %% [markdown]
# ## Validation data

# %%
print('\nValidation prediction')
y_pred = model.predict(val_data, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = val_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
val_prediction_df = process_prediction(target_df, y_pred, Config)
print(val_prediction_df.describe())

# %%
show_result(val_prediction_df, targets)
for target in targets:
    plot_predition(
        val_prediction_df, target, show_image=SHOW_IMAGE,
        figure_path=os.path.join(output_folder, f'Summed_{target}_Validation.jpg')
    )

# %% [markdown]
# ## Test data

# %%
print('\nTest prediction')
y_pred = model.predict(test_data, verbose=VERBOSE)

# upscale prediction
y_pred = target_scaler.inverse_transform(
    y_pred.reshape((-1, len(targets)))
).reshape((-1, output_sequence_length, len(targets)))

# upscale ground truth
target_df = test_df[[group_id, 'Date'] + targets].copy().reset_index(drop=True)
target_df[targets] = target_scaler.inverse_transform(target_df[targets])

# align predictions with ground truth
test_prediction_df = process_prediction(target_df, y_pred, Config)
print(test_prediction_df.describe())

# %%
show_result(test_prediction_df, targets)
for target in targets:
    plot_predition(
        test_prediction_df, target=target, show_image=SHOW_IMAGE,
        figure_path=os.path.join(output_folder, f'Summed_{target}_Test.jpg')
    )

# %% [markdown]
# ## Dump

# %%
train_prediction_df['Split'] = 'train'
val_prediction_df['Split'] = 'validation'
test_prediction_df['Split'] = 'test'
merged_df = pd.concat([train_prediction_df, val_prediction_df, test_prediction_df], axis=0)
merged_df.to_csv(os.path.join(output_folder, 'predictions.csv'), index=False)
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')
