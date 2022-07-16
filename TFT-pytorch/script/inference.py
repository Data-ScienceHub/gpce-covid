# %% [markdown]
# # Imports

# %%
import os, gc
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
# Uncomment the following if you are running on google colab. They don't have these libraries installed by default. Only uncomment the pip install part if you are on rivanna, using a default pytorch kernel.

# %%
# !pip install pytorch_lightning
# !pip install pytorch_forecasting

# from google.colab import drive

# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Projects/Covid/notebooks

# %% [markdown]
# ## Pytorch lightning and forecasting

# %%
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# %% [markdown]
# # Load input

# %%
from dataclasses import dataclass

@dataclass
class args:
    outputPath = '../top_100_early_stopped_target_unscaled'
    figPath = os.path.join(outputPath, 'figures')
    checkpoint_folder = os.path.join(outputPath, 'checkpoints')
    input_filePath = '../2022_May_target_cleaned/Top_100.csv'
    configPath = '../config_2022_May.json'

    load_from_checkpoint = False
    model_path = os.path.join(checkpoint_folder, 'model.ckpt')

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
# # Processing

# %%
total_data['Date'] = pd.to_datetime(total_data['Date'].values) 
total_data['FIPS'] = total_data['FIPS'].astype(str)
print(f"There are {total_data['FIPS'].nunique()} unique counties in the dataset.")

# %%
## Fill missing values
# Currently not needed as they are filled with 0 during the data preparation stage. 
# But TFT doesn't support NULL values. So if you have any, replace them.

df = missing_percentage(total_data)

print(df[df>0])

if df[df>0].shape[0]>0:
    print('Filling null values with 0.')
    total_data.fillna(0, inplace=True)

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
_, train_dataloader = prepare_data(train_scaled, parameters, train=True)
_, validation_dataloader = prepare_data(validation_scaled, parameters)
_, test_dataloader = prepare_data(test_scaled, parameters)

del validation_scaled, test_scaled
gc.collect()

# %% [markdown]
# # Evaluation metric

# %%
def show_result(df: pd.DataFrame, targets=targets):    
    for target in targets:
        predicted_column = f'Predicted_{target}'
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, msle, smape, nnse = calculate_result(y_true, y_pred)
        print(f'Target {target}, MAE {mae:5g}, RMSE {rmse:5g}, MSLE {msle:5g}, SMAPE {smape:5g}. NNSE {nnse:5g}.')
    print()

# %% [markdown]
# # Model

# %%
tft = TemporalFusionTransformer.load_from_checkpoint(args.model_path)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# %% [markdown]
# # Prediction Processor and PlotResults

# %%
from Class.PredictionProcessor import PredictionProcessor

processor = PredictionProcessor(
    time_idx, parameters.data.id[0], max_prediction_length, targets, 
    train_start, max_encoder_length
)

# %%
from Class.Plotter import *

plotter = PlotResults(args.figPath, targets)

# %% [markdown]
# # Evaluate

# %% [markdown]
# ## Train results

# %% [markdown]
# ### Average

# %%
print(f'\n---Training results--\n')

train_predictions, train_index = tft.predict(train_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
train_predictions = upscale_prediction(targets, train_predictions, target_scaler, max_prediction_length)
train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged)

# %%
plotter.summed_plot(train_result_merged, type='Train_avg' , save=True, base=35)
# df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = 1)
# plotter.summed_plot(df, type='Train_day_1' , save=True)

# %% [markdown]
# ### By future days

# %%
gc.collect()
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
    show_result(df)

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')
validation_predictions, validation_index = tft.predict(validation_dataloader, mode="prediction", return_index=True, show_progress_bar=args.show_progress_bar)
validation_predictions = upscale_prediction(targets, validation_predictions, target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged)

# %%
plotter.summed_plot(validation_result_merged, type='Validation_avg', save=True)
# df = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index, target_time_step = 1)
# plotter.summed_plot(df, type='Validation_day_1', save=True)

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

# %%
plotter.summed_plot(test_result_merged, 'Test_avg', save=True)
# df = processor.align_result_with_dataset(test_data, test_predictions, test_index, target_time_step = 1)
# plotter.summed_plot(df, 'Test_day_1', save=True)

# %% [markdown]
# ### By future days

# %%
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(test_data, test_predictions, test_index, target_time_step = day)
    show_result(df)

# %% [markdown]
# ## Dump results

# %%
# train_result_merged['split'] = 'train'
# validation_result_merged['split'] = 'validation'
# test_result_merged['split'] = 'test'
# df = pd.concat([train_result_merged, validation_result_merged, test_result_merged])
# df.to_csv(os.path.join(args.outputPath, 'predictions_case_death.csv'), index=False)

# df.head()

# %% [markdown]
# ## Evaluation by county

# %%
fips_codes = total_data['FIPS'].unique()
names_df = pd.read_csv('../../dataset_raw/CovidMay17-2022/Age Distribution.csv')[['FIPS','Name']]
names_df['FIPS'] = names_df['FIPS'].astype(str)

# %%
print(f'\n---Per county training results--\n')
count = 10

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
plotWeights = PlotWeights(args.figPath, parameters)

# %%
# tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
train_raw_predictions = tft.predict(train_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
validation_raw_predictions = tft.predict(validation_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)
test_raw_predictions = tft.predict(test_dataloader, mode="raw", show_progress_bar=args.show_progress_bar)

train_interpretation = tft.interpret_output(train_raw_predictions)
for key in train_interpretation.keys():
    print(key, train_interpretation[key].shape)

# %% [markdown]
# ### Train

# %%
attention_mean = processor.get_mean_attention(
    train_interpretation, train_index
)
plotWeights.plot_attention(attention_mean, figure_name='Train_attention', base=35)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Train_weekly_attention')

# %% [markdown]
# ### Validation

# %%
attention_mean = processor.get_mean_attention(
    tft.interpret_output(validation_raw_predictions), 
    validation_index
)
plotWeights.plot_attention(attention_mean, figure_name='Validation_attention', base=3)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Validation_weekly_attention')

# %% [markdown]
# ### Test

# %%
attention_mean = processor.get_mean_attention(
    tft.interpret_output(test_raw_predictions), 
    test_index
)
plotWeights.plot_attention(attention_mean, figure_name='Test_attention', base=3)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Test_weekly_attention')

# %% [markdown]
# ## Variable Importance

# %% [markdown]
# ## Train

# %%
interpretation = tft.interpret_output(train_raw_predictions, reduction="sum")
for key in interpretation.keys():
    print(key, interpretation[key])
    
figures = plotWeights.plot_interpretation(interpretation)
for key in figures.keys():
    figure = figures[key]
    figure.savefig(os.path.join(plotter.figpath, f'Train_{key}.jpg'), dpi=DPI)

# %% [markdown]
# ## Test

# %%
interpretation = tft.interpret_output(test_raw_predictions, reduction="sum")
for key in interpretation.keys():
    print(key, interpretation[key])
    
figures = plotWeights.plot_interpretation(interpretation)
for key in figures.keys():
    figure = figures[key]
    figure.savefig(os.path.join(plotter.figpath, f'Test_{key}.jpg'), dpi=DPI)


