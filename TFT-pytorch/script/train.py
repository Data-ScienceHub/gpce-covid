# %% [markdown]
# # Introduction
# 
# This notebook uses the Temporal Fusion Transformer (TFT) model implemented in PyTorch forecasting. It follows the tutorial notebook [here](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html). It loads the merged feature file created by the Data preparation notebook/script, then splits that into train/validation/test split. Then uses the pytorch trainer to fit the model. And finally plots the predictions and interpretations. 

# %% [markdown]
# # Imports

# %%
import os, gc
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', None)

# %% [markdown]
# # Initial setup

# %% [markdown]
# ## GPU

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %% [markdown]
# ## Google colab
# 
# Set `running_on_colab` to true if you are running on google colab. They don't have these libraries installed by default. Uncomment the codes too if needed. They might be commented out since in .py script inline commands show errors.
# 
# Use only the pip install part if you are on rivanna, using a default tensorflow kernel.

# %%
running_on_colab = False

# if running_on_colab:
#     !pip install pytorch_lightning
#     !pip install pytorch_forecasting

#     from google.colab import drive

#     drive.mount('/content/drive')
#     %cd /content/drive/My Drive/TFT-pytorch/notebook

# %% [markdown]
# ## Pytorch lightning and forecasting

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import RMSE, MultiLoss

# %% [markdown]
# # Load input

# %%
from dataclasses import dataclass

@dataclass
class args:
    result_folder = '../results/total_target_cleaned_scaled'
    figPath = os.path.join(result_folder, 'figures')
    checkpoint_folder = os.path.join(result_folder, 'checkpoints')
    input_filePath = '../2022_May_target_cleaned/Total.csv'

    # pass your intented configuration here
    # input features are always normalized. But keeping the targets features unscaled improves results
    # if you want to change some config, but not to create a new config file, just change the value
    # of the corresponding parameter in the config section
    configPath = '../configurations/total_target_cleaned_scaled.json'

    # Path/URL of the checkpoint from which training is resumed
    ckpt_model_path = None # os.path.join(checkpoint_folder, 'latest-epoch=7.ckpt')
    
    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = False

# %%
total_data = pd.read_csv(args.input_filePath)
print(total_data.shape)
total_data.head()

# %% [markdown]
# # Config

# %%
import json
import sys
sys.path.append( '..' )
from Class.Parameters import Parameters
from script.utils import *

with open(args.configPath, 'r') as input_file:
  config = json.load(input_file)

parameters = Parameters(config, **config)

# %%
targets = parameters.data.targets
time_idx = parameters.data.time_idx
tft_params = parameters.model_parameters

# google colab doesn't utilize GPU properly for pytorch
# so increasing batch size forces more utilization
# not needed on rivanna or your local machine

if running_on_colab: 
    tft_params.batch_size *= 4

max_prediction_length = tft_params.target_sequence_length
max_encoder_length = tft_params.input_sequence_length

# Don't use LinearSpace as time known feature, as it has no relation with time
# parameters.data.time_varying_known_features = [
#     feature for feature in parameters.data.time_varying_known_features if feature != 'LinearSpace'
# ]

# %% [markdown]
# # Seed

# %%
import random

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

seed_torch(tft_params.seed)

# %% [markdown]
# # Processing

# %%
total_data['Date'] = pd.to_datetime(total_data['Date'].values) 
total_data['FIPS'] = total_data['FIPS'].astype(str)
print(f"There are {total_data['FIPS'].nunique()} unique counties in the dataset.")

# %% [markdown]
# ## Adapt input to encoder length
# Input data length needs to be a multiple of encoder length to created batch data loaders.

# %%
train_start, validation_start, test_start = get_start_dates(parameters)
total_data = total_data[total_data['Date']>=train_start]
total_data[time_idx] = (total_data["Date"] - total_data["Date"].min()).apply(lambda x: x.days)

# %% [markdown]
# ## Train validation test split and scaling

# %%
train_data, validation_data, test_data = train_validation_test_split(
    total_data, parameters
)

# %%
train_scaled, validation_scaled, test_scaled, target_scaler = scale_data(
    train_data, validation_data, test_data, parameters
)

# %% [markdown]
# ## Create dataset and dataloaders

# %%
def prepare_data(data: pd.DataFrame, pm: Parameters, train=False):

  data_timeseries = TimeSeriesDataSet(
    data,
    time_idx= time_idx,
    target=targets,
    group_ids=pm.data.id, 
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_reals=pm.data.static_features,
    static_categoricals=['FIPS'],
    time_varying_known_reals = pm.data.time_varying_known_features,
    time_varying_unknown_reals = pm.data.time_varying_unknown_features,
    target_normalizer = MultiNormalizer(
      [GroupNormalizer(groups=pm.data.id) for _ in range(len(targets))]
    )
  )

  if train:
    dataloader = data_timeseries.to_dataloader(train=True, batch_size=tft_params.batch_size)
  else:
    dataloader = data_timeseries.to_dataloader(train=False, batch_size=tft_params.batch_size*16)

  return data_timeseries, dataloader

# %%
train_timeseries, train_dataloader = prepare_data(train_scaled, parameters, train=True)
_, validation_dataloader = prepare_data(validation_scaled, parameters)
_, test_dataloader = prepare_data(test_scaled, parameters)

del validation_scaled, test_scaled
gc.collect()

# %% [markdown]
# # Training

# %% [markdown]
# ## Evaluation metric

# %%
def show_result(df: pd.DataFrame, targets=targets):    
    for target in targets:
        predicted_column = f'Predicted_{target}'
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, msle, smape, nnse = calculate_result(y_true, y_pred)
        print(f'Target {target}, MAE {mae:.5g}, RMSE {rmse:.5g}, MSLE {msle:.5g}, SMAPE {smape:0.5g}. NNSE {nnse:0.5g}.')
    print()

# %% [markdown]
# ## Trainer and logger

# %% [markdown]
# If you have troubles training the model and get an error ```AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'```, consider either uninstalling tensorflow or first execute the following

# %%
import tensorflow as tf
# click this and locate the lightning_logs folder path and select that folder. 
# this will load tensorbaord visualization
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# %%
# notice that the early stopping patience is very high (60) for the old
# TF1 notebook. To reproduce that, replace patience=60
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=tft_params.early_stopping_patience
    , verbose=True, mode="min"
)

# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
best_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=args.checkpoint_folder, monitor="val_loss", filename="best-{epoch}"
)
latest_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=args.checkpoint_folder, every_n_epochs=1, filename="latest-{epoch}"
)

logger = TensorBoardLogger(args.result_folder)  # logging results to a tensorboard

# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
trainer = pl.Trainer(
    max_epochs = tft_params.epochs,
    accelerator = 'auto',
    weights_summary = "top",
    gradient_clip_val = tft_params.clipnorm,
    callbacks = [early_stop_callback, best_checkpoint, latest_checkpoint],
    logger = logger,
    enable_progress_bar = args.show_progress_bar,
    check_val_every_n_epoch = 1,
    # max_time="00:12:00:00",
    # auto_scale_batch_size = False 
)

# %% [markdown]
# ## Model

# %%
# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
tft = TemporalFusionTransformer.from_dataset(
    train_timeseries,
    learning_rate= tft_params.learning_rate,
    hidden_size= tft_params.hidden_layer_size,
    attention_head_size=tft_params.attention_head_size,
    dropout=tft_params.dropout_rate,
    loss=MultiLoss([RMSE(reduction='mean') for _ in targets]), # RMSE(reduction='sqrt-mean')
    optimizer='adam',
    log_interval=1,
    # reduce_on_plateau_patience=2
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# %% [markdown]
# The training speed is here mostly determined by overhead and choosing a larger `batch_size` or `hidden_size` (i.e. network size) does not slow training linearly making training on large datasets feasible. During training, we can monitor the tensorboard which can be spun up with `tensorboard --logdir=lightning_logs`. For example, we can monitor examples predictions on the training and validation set.

# %%
from datetime import datetime

gc.collect()

start = datetime.now()
print(f'\n----Training started at {start}----\n')

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=validation_dataloader,
    ckpt_path=args.ckpt_model_path
)
end = datetime.now()
print(f'\n----Training ended at {end}, elapsed time {end-start}')
print(f'Best model by validation loss saved at {trainer.checkpoint_callback.best_model_path}')

# %% [markdown]
# # Prediction Processor

# %%
from Class.PredictionProcessor import PredictionProcessor

processor = PredictionProcessor(
    time_idx, parameters.data.id[0], max_prediction_length, targets, 
    train_start, max_encoder_length
)

# %% [markdown]
# # Evaluate - final model

# %% [markdown]
# ## PlotResults

# %%
from Class.Plotter import *

plotter = PlotResults(args.figPath, targets, show=args.show_progress_bar)

# %% [markdown]
# ## Train results

# %% [markdown]
# ### Average

# %%
# not a must, but increases inference speed 
_, train_dataloader = prepare_data(train_scaled, parameters) 
print(f'\n---Training results--\n')

# mode="prediction" would return only the prediction. mode="raw" returns additional keys needed for interpretation
train_raw_predictions, train_index = tft.predict(
    train_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
for key in train_raw_predictions.keys():
    item = train_raw_predictions[key]
    if type(item)==list: print(key, f'list of length {len(item)}', item[0].shape)
    else: print(key, item.shape)

train_predictions = upscale_prediction(targets, train_raw_predictions['prediction'], target_scaler, max_prediction_length)
train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged)
plotter.summed_plot(train_result_merged, type='Train' , base=45)
gc.collect()

# %% [markdown]
# ### By future days

# %%
gc.collect()
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
    show_result(df)
    plotter.summed_plot(df, type=f'Train_day_{day}', base=45)
    break

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')
validation_raw_predictions, validation_index = tft.predict(
    validation_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
validation_predictions = upscale_prediction(targets, validation_raw_predictions['prediction'], target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged)
plotter.summed_plot(validation_result_merged, type='Validation')
gc.collect()

# %% [markdown]
# ## Test results

# %% [markdown]
# ### Average

# %%
print(f'\n---Test results--\n')
test_raw_predictions, test_index = tft.predict(
    test_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
test_predictions = upscale_prediction(targets, test_raw_predictions['prediction'], target_scaler, max_prediction_length)

test_result_merged = processor.align_result_with_dataset(test_data, test_predictions, test_index)
show_result(test_result_merged)
plotter.summed_plot(test_result_merged, 'Test')
gc.collect()

# %% [markdown]
# ### By future days

# %%
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(test_data, test_predictions, test_index, target_time_step = day)
    show_result(df)
    # plotter.summed_plot(df, type=f'Test_day_{day}')
    # break

# %% [markdown]
# ## Dump results

# %%
train_result_merged['split'] = 'train'
validation_result_merged['split'] = 'validation'
test_result_merged['split'] = 'test'
df = pd.concat([train_result_merged, validation_result_merged, test_result_merged])
df.to_csv(os.path.join(args.result_folder, 'predictions_case_death.csv'), index=False)

df.head()

# %%
del train_predictions, validation_predictions, test_predictions
gc.collect()

# %% [markdown]
# ## Evaluation by county

# %%
fips_codes = train_result_merged['FIPS'].unique()
names_df = total_data[['FIPS', 'Name']]

print(f'\n---Per county train results--\n')
count = 5

for index, fips in enumerate(fips_codes):
    if index == count: break

    name = names_df[names_df['FIPS']==fips]['Name'].values[0]
    print(f'County {name}, FIPS {fips}')
    df = train_result_merged[train_result_merged['FIPS']==fips]
    show_result(df, targets)
    print()

# %%
del train_result_merged, validation_result_merged, test_result_merged, df

# %% [markdown]
# ## Attention weights

# %%
plotWeights = PlotWeights(args.figPath, max_encoder_length, tft, show=args.show_progress_bar)

# %% [markdown]
# ### Train

# %%
# interpret_output has high memory requirement
# results in out-of-memery for Total.csv and a model of hidden size 64, even with 64GB memory
if 'Total.csv' not in args.input_filePath:
    attention_mean = processor.get_mean_attention(
        tft.interpret_output(train_raw_predictions), train_index
    )
    plotWeights.plot_attention(
        attention_mean, figure_name='Train_daily_attention', base=45, 
        limit=0, enable_markers=False, title='Attention with dates'
    )

    attention_weekly = processor.get_attention_by_weekday(attention_mean)
    plotWeights.plot_weekly_attention(attention_weekly, figure_name='Train_weekly_attention')

# %% [markdown]
# ### Validation

# %%
attention_mean = processor.get_mean_attention(
    tft.interpret_output(validation_raw_predictions), validation_index
)
plotWeights.plot_attention(
    attention_mean, figure_name='Validation_daily_attention', target_day=4
)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Validation_weekly_attention')

# %% [markdown]
# ### Test

# %%
attention_mean = processor.get_mean_attention(
    tft.interpret_output(test_raw_predictions), test_index
)
plotWeights.plot_attention(attention_mean, figure_name='Test_daily_attention', target_day=4)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Test_weekly_attention')

# %% [markdown]
# ## Variable Importance

# %% [markdown]
# ### Train

# %%
if 'Total.csv' not in args.input_filePath:
    interpretation = tft.interpret_output(train_raw_predictions, reduction="sum")
    for key in interpretation.keys():
        print(key, interpretation[key])
        
    figures = plotWeights.plot_interpretation(interpretation)
    for key in figures.keys():
        figure = figures[key]
        figure.savefig(os.path.join(plotter.figpath, f'Train_{key}.jpg'), dpi=DPI)
    

# %% [markdown]
# ### Test

# %%
interpretation = tft.interpret_output(test_raw_predictions, reduction="sum")
for key in interpretation.keys():
    print(key, interpretation[key])
    
figures = plotWeights.plot_interpretation(interpretation)
for key in figures.keys():
    figure = figures[key]
    figure.savefig(os.path.join(plotter.figpath, f'Test_{key}.jpg'), dpi=DPI)

# %% [markdown]
# ## Clear up

# %%
del tft, attention_mean, attention_weekly, interpretation
del train_raw_predictions, validation_raw_predictions, test_raw_predictions
gc.collect()

# %% [markdown]
# # Evaluate - best model
# Best model checkpointed by validation loss.

# %%
best_model_path = trainer.checkpoint_callback.best_model_path
print(f'Loading best model from {best_model_path}')
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# %%
plotter = PlotResults(f'{args.figPath}_best', targets, show=args.show_progress_bar)

# %% [markdown]
# ## Train results

# %% [markdown]
# ### Average

# %%
print(f'\n---Training results--\n')
train_raw_predictions, train_index = best_tft.predict(
    train_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
train_predictions = upscale_prediction(targets, train_raw_predictions['prediction'], target_scaler, max_prediction_length)

train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged)
plotter.summed_plot(train_result_merged, type='Train', base=45)
gc.collect()

# %% [markdown]
# ### By future days

# %%
# for day in range(1, max_prediction_length+1):
#     print(f'Day {day}')
#     df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
#     show_result(df)
    # plotter.summed_plot(df, type=f'Train_day_{day}', base=45)

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')

validation_raw_predictions, validation_index = best_tft.predict(
    validation_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
validation_predictions = upscale_prediction(targets, validation_raw_predictions['prediction'], target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged)
plotter.summed_plot(validation_result_merged, type='Validation')
gc.collect()

# %% [markdown]
# ## Test results

# %% [markdown]
# ### Average

# %%
print(f'\n---Test results--\n')

test_raw_predictions, test_index = best_tft.predict(
    test_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)
test_predictions = upscale_prediction(targets, test_raw_predictions['prediction'], target_scaler, max_prediction_length)

test_result_merged = processor.align_result_with_dataset(total_data, test_predictions, test_index)
show_result(test_result_merged)
plotter.summed_plot(test_result_merged, 'Test')
gc.collect()

# %% [markdown]
# ### By future days

# %%
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(test_data, test_predictions, test_index, target_time_step = day)
    show_result(df)
    # plotter.summed_plot(df, type=f'Test_day_{day}')

# %% [markdown]
# ## Dump results

# %%
train_result_merged['split'] = 'train'
validation_result_merged['split'] = 'validation'
test_result_merged['split'] = 'test'
df = pd.concat([train_result_merged, validation_result_merged, test_result_merged])
df.to_csv(os.path.join(args.result_folder, 'best_predictions_case_death.csv'), index=False)

df.head()

# %%
del train_predictions, validation_predictions, test_predictions
gc.collect()

# %% [markdown]
# ## Evaluation by county

# %%
fips_codes = test_result_merged['FIPS'].unique()
names_df = total_data[['FIPS', 'Name']]

print(f'\n---Per county test results--\n')
count = 5

for index, fips in enumerate(fips_codes):
    if index == count: break

    name = names_df[names_df['FIPS']==fips]['Name'].values[0]
    print(f'County {name}, FIPS {fips}')
    df = test_result_merged[test_result_merged['FIPS']==fips]
    show_result(df, targets)
    print()

# %%
del train_result_merged, validation_result_merged, test_result_merged, df

# %% [markdown]
# ## Attention weights

# %%
plotWeights = PlotWeights(
    args.figPath+'_best', max_encoder_length, best_tft, show=args.show_progress_bar
)

# %% [markdown]
# ### Train

# %%
# interpret_output has high memory requirement
# results in out-of-memery for Total.csv and a model of hidden size 64, even with 64GB memory
if 'Total.csv' not in args.input_filePath:
    attention_mean = processor.get_mean_attention(
        best_tft.interpret_output(train_raw_predictions), train_index
    )
    plotWeights.plot_attention(
        attention_mean, figure_name='Train_daily_attention', base=45, 
        limit=0, enable_markers=False, title='Attention with dates'
    )
    gc.collect()
    attention_weekly = processor.get_attention_by_weekday(attention_mean)
    plotWeights.plot_weekly_attention(attention_weekly, figure_name='Train_weekly_attention')

# %% [markdown]
# ### Validation

# %%
attention_mean = processor.get_mean_attention(
    best_tft.interpret_output(validation_raw_predictions), 
    validation_index
)
plotWeights.plot_attention(attention_mean, figure_name='Validation_daily_attention', target_day=4)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Validation_weekly_attention')

# %% [markdown]
# ### Test

# %%
attention_mean = processor.get_mean_attention(
    best_tft.interpret_output(test_raw_predictions), 
    test_index
)
plotWeights.plot_attention(attention_mean, figure_name='Test_daily_attention', target_day=4)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Test_weekly_attention')

# %% [markdown]
# ## Variable Importance

# %% [markdown]
# ## Train

# %%
if 'Total.csv' not in args.input_filePath:
    interpretation = best_tft.interpret_output(train_raw_predictions, reduction="sum")
    for key in interpretation.keys():
        print(key, interpretation[key])

    figures = plotWeights.plot_interpretation(interpretation)
    for key in figures.keys():
        figure = figures[key]
        figure.savefig(os.path.join(plotter.figpath, f'Train_{key}.jpg'), dpi=DPI)    

# %% [markdown]
# ## Test

# %%
interpretation = best_tft.interpret_output(test_raw_predictions, reduction="sum")
for key in interpretation.keys():
    print(key, interpretation[key])
    
figures = plotWeights.plot_interpretation(interpretation)
for key in figures.keys():
    figure = figures[key]
    figure.savefig(os.path.join(plotter.figpath, f'Test_{key}.jpg'), dpi=DPI)    

# %%
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')


