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
# Uncomment the following if you are running on google colab. They don't have these libraries installed by default. 
# 
# Only uncomment the pip install part if you are on rivanna, using a default pytorch kernel.

# %%
# !pip install pytorch_lightning
# !pip install pytorch_forecasting

# from google.colab import drive

# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Projects/Covid/notebooks

# %% [markdown]
# ## Pytorch lightning and forecasting
# 
# Pytorch forecasting has direct support for TFT https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html. 

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
    outputPath = '../results/top_500_target_cleaned_scaled'
    figPath = os.path.join(outputPath, 'figures')
    checkpoint_folder = os.path.join(outputPath, 'checkpoints')
    input_filePath = '../2022_May_target_cleaned/Top_500.csv'

    # pass your intented configuration here
    configPath = '../configurations/top_500_target_cleaned_scaled.json'

    final_model_path = os.path.join(checkpoint_folder, "model.ckpt")
    # Path/URL of the checkpoint from which training is resumed
    ckpt_model_path = None # "some/path/to/my_checkpoint.ckpt")
    
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

batch_size = tft_params.batch_size
max_prediction_length = tft_params.target_sequence_length
max_encoder_length = tft_params.input_sequence_length

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
    time_varying_known_reals = pm.data.time_varying_known_features,
    time_varying_unknown_reals = pm.data.time_varying_unknown_features,
    target_normalizer = MultiNormalizer(
      [GroupNormalizer(groups=pm.data.id, transformation="softplus") for _ in range(len(targets))]
    )
  )

  if train:
    dataloader = data_timeseries.to_dataloader(train=True, batch_size=batch_size)
  else:
    dataloader = data_timeseries.to_dataloader(train=False, batch_size=batch_size*20)

  return data_timeseries, dataloader

# %%
train_timeseries, train_dataloader = prepare_data(train_scaled, parameters, train=True)
_, validation_dataloader = prepare_data(validation_scaled, parameters, train=False)
_, test_dataloader = prepare_data(test_scaled, parameters, train=False)

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
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.checkpoint_folder, monitor="val_loss"
)

logger = TensorBoardLogger(args.outputPath)  # logging results to a tensorboard

# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
trainer = pl.Trainer(
    max_epochs = tft_params.epochs,
    accelerator = 'auto',
    weights_summary = "top",
    gradient_clip_val = tft_params.clipnorm,
    callbacks = [early_stop_callback, checkpoint_callback],
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
    loss=MultiLoss([RMSE() for _ in targets]),
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
trainer.save_checkpoint(args.final_model_path)

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
_, train_dataloader = prepare_data(train_scaled, parameters, train=False) 
print(f'\n---Training results--\n')

train_predictions, train_index = tft.predict(train_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
train_predictions = upscale_prediction(targets, train_predictions, target_scaler, max_prediction_length)
train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged)
plotter.summed_plot(train_result_merged, type='Train' , base=45)

# %% [markdown]
# ### By future days

# %%
gc.collect()
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
    show_result(df)
    # plotter.summed_plot(df, type=f'Train_day_{day}', base=45)

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')
validation_predictions, validation_index = tft.predict(validation_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
validation_predictions = upscale_prediction(targets, validation_predictions, target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged)
plotter.summed_plot(validation_result_merged, type='Validation')

# %%
print('Minimum and maximum time index from validation data and its index')
for df in [validation_data, validation_index]:
    print(df[time_idx].min(), df[time_idx].max())

# %% [markdown]
# ## Test results

# %% [markdown]
# ### Average

# %%
print(f'\n---Test results--\n')
test_predictions, test_index = tft.predict(test_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
test_predictions = upscale_prediction(targets, test_predictions, target_scaler, max_prediction_length)

test_result_merged = processor.align_result_with_dataset(test_data, test_predictions, test_index)
show_result(test_result_merged)
plotter.summed_plot(test_result_merged, 'Test')

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
df.to_csv(os.path.join(args.outputPath, 'predictions_case_death.csv'), index=False)

df.head()

# %% [markdown]
# ## Evaluation by county

# %%
fips_codes = total_data['FIPS'].unique()
names_df = pd.read_csv('../../dataset_raw/CovidMay17-2022/Age Distribution.csv')[['FIPS','Name']]
names_df['FIPS'] = names_df['FIPS'].astype(str)

print(f'\n---Per county training results--\n')
count = 5

for index, fips in enumerate(fips_codes):
    if index == count: break

    name = names_df[names_df['FIPS']==fips]['Name'].values[0]
    print(f'County {name}, FIPS {fips}')
    df = train_result_merged[train_result_merged['FIPS']==fips]
    show_result(df, targets)
    print()

# %% [markdown]
# ## Attention weights

# %%
plotWeights = PlotWeights(args.figPath, parameters, show=args.show_progress_bar)

# %%
# tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
train_raw_predictions = tft.predict(train_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
validation_raw_predictions = tft.predict(validation_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
test_raw_predictions = tft.predict(test_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)

# %% [markdown]
# ### Train

# %%
attention_mean = processor.get_mean_attention(
    tft.interpret_output(train_raw_predictions), train_index
)
plotWeights.plot_attention(
    attention_mean, figure_name='Train_daily_attention', base=45, 
    limit=0, enable_markers=False, title='Attention with dates'
)

# %%
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
del tft
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
train_predictions, train_index = best_tft.predict(train_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
train_predictions = upscale_prediction(targets, train_predictions, target_scaler, max_prediction_length)

train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged)
plotter.summed_plot(train_result_merged, type='Train', base=45)

# %% [markdown]
# ### By future days

# %%
gc.collect()
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
    show_result(df)
    # plotter.summed_plot(df, type=f'Train_day_{day}', base=45)

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')

validation_predictions, validation_index = best_tft.predict(validation_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
validation_predictions = upscale_prediction(targets, validation_predictions, target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged)
plotter.summed_plot(validation_result_merged, type='Validation')

# %% [markdown]
# ## Test results

# %% [markdown]
# ### Average

# %%
print(f'\n---Test results--\n')

test_predictions, test_index = best_tft.predict(test_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
test_predictions = upscale_prediction(targets, test_predictions, target_scaler, max_prediction_length)

test_result_merged = processor.align_result_with_dataset(total_data, test_predictions, test_index)
show_result(test_result_merged)
plotter.summed_plot(test_result_merged, 'Test')

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
df.to_csv(os.path.join(args.outputPath, 'best_predictions_case_death.csv'), index=False)

df.head()

# %% [markdown]
# ## Evaluation by county

# %%
fips_codes = total_data['FIPS'].unique()
names_df = pd.read_csv('../../dataset_raw/CovidMay17-2022/Age Distribution.csv')[['FIPS','Name']]
names_df['FIPS'] = names_df['FIPS'].astype(str)

print(f'\n---Per county test results--')
count = 5

for index, fips in enumerate(fips_codes):
    if index == count: break

    name = names_df[names_df['FIPS']==fips]['Name'].values[0]
    print(f'\nCounty {name}, FIPS {fips}')
    df = test_result_merged[train_result_merged['FIPS']==fips]
    show_result(df, targets)

# %% [markdown]
# ## Attention weights

# %%
plotWeights = PlotWeights(args.figPath+'_best', parameters, show=args.show_progress_bar)

# %%
# tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
train_raw_predictions = best_tft.predict(train_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
validation_raw_predictions = best_tft.predict(validation_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
test_raw_predictions = best_tft.predict(test_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)

# %% [markdown]
# ### Train

# %%
attention_mean = processor.get_mean_attention(
    best_tft.interpret_output(train_raw_predictions), train_index
)
plotWeights.plot_attention(
    attention_mean, figure_name='Train_daily_attention', base=45, 
    limit=0, enable_markers=False, title='Attention with dates'
)

# %%
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


