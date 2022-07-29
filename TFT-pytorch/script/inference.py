# %% [markdown]
# # Imports

# %%
import os, gc
import torch
from datetime import datetime

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
# %cd /content/drive/My Drive/TFT-pytorch/notebook

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
    result_folder = '../results/total_target_cleaned_scaled'
    figPath = os.path.join(result_folder, 'figures')
    checkpoint_folder = os.path.join(result_folder, 'checkpoints')
    input_filePath = '../2022_May_target_cleaned/Total.csv'
    configPath = '../configurations/total_target_cleaned_scaled.json'

    model_path = os.path.join(checkpoint_folder, 'latest-epoch=59.ckpt')

    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = False

    # interpret_output has high memory requirement
    # results in out-of-memery for Total.csv and a model of hidden size 64, even with 64GB memory
    interpret_train = 'Total.csv' not in input_filePath

# %%
start = datetime.now()
print(f'Started at {start}')

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

# only kept for comparability with TF1. 
# Should not be used uere since LinearSpace is just a static categorical feature
parameters.data.time_varying_known_features.append('LinearSpace')

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
    # static_categoricals=['FIPS'],
    time_varying_known_reals = pm.data.time_varying_known_features,
    time_varying_unknown_reals = pm.data.time_varying_unknown_features,
    target_normalizer = MultiNormalizer(
      [GroupNormalizer(groups=pm.data.id) for _ in range(len(targets))]
    )
  )

  if train:
    dataloader = data_timeseries.to_dataloader(train=True, batch_size=batch_size)
  else:
    dataloader = data_timeseries.to_dataloader(train=False, batch_size=batch_size*8)

  return data_timeseries, dataloader

# %%
_, train_dataloader = prepare_data(train_scaled, parameters)
_, validation_dataloader = prepare_data(validation_scaled, parameters)
_, test_dataloader = prepare_data(test_scaled, parameters)

del validation_scaled, test_scaled
gc.collect()

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

plotter = PlotResults(args.figPath, targets, show=args.show_progress_bar)

# %% [markdown]
# # Evaluate

# %% [markdown]
# ## Train results

# %% [markdown]
# ### Average

# %%
print(f'\n---Training results--\n')

train_raw_predictions, train_index = tft.predict(
    train_dataloader, mode="raw", return_index=True, show_progress_bar=args.show_progress_bar
)

train_predictions = upscale_prediction(targets, train_raw_predictions['prediction'], target_scaler, max_prediction_length)
train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged, targets)

plotter.summed_plot(train_result_merged, type='Train', base=45)
plotter.summed_plot(train_result_merged, type='Train_error', base=45, plot_error=True)
gc.collect()

# %%
predicted_columns = [f'Predicted_{target}' for target in targets]
if target_scaler is not None:
    train_result_merged.loc[:, predicted_columns] = target_scaler.transform(train_result_merged[predicted_columns])
    train_result_merged.loc[:, targets] = target_scaler.transform(train_result_merged[targets])
else:
    scaler = MinMaxScaler()
    train_result_merged.loc[:, targets] = scaler.fit_transform(train_result_merged[targets])
    train_result_merged.loc[:, predicted_columns] = scaler.transform(train_result_merged[predicted_columns])
    
show_result(train_result_merged, targets)
plotter.summed_plot(train_result_merged, type='Train_scaled', base=45)

# %% [markdown]
# ### By future days

# %%
gc.collect()
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(train_data, train_predictions, train_index, target_time_step = day)
    show_result(df, targets)
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
show_result(validation_result_merged, targets)
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
show_result(test_result_merged, targets)
plotter.summed_plot(test_result_merged, 'Test')
plotter.summed_plot(test_result_merged, type='Test_error',  plot_error=True)
gc.collect()

# %% [markdown]
# ### By future days

# %%
for day in range(1, max_prediction_length+1):
    print(f'Day {day}')
    df = processor.align_result_with_dataset(test_data, test_predictions, test_index, target_time_step = day)
    show_result(df, targets)
    # plotter.summed_plot(df, type=f'Test_day_{day}')

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
del train_result_merged, validation_result_merged, test_result_merged

# %% [markdown]
# ## Attention weights

# %%
plotWeights = PlotWeights(args.figPath, max_encoder_length, tft, show=args.show_progress_bar)
attentions, attention_means = [], []

# %% [markdown]
# ### Train

# %%
if args.interpret_train:
    attention_mean, attention = processor.get_mean_attention(
        tft.interpret_output(train_raw_predictions), train_index,return_attention=True
    )
    plotWeights.plot_attention(
        attention_mean, figure_name='Train_daily_attention', base=45, 
        limit=0, enable_markers=False, title='Attention with dates'
    )
    gc.collect()
    attention_weekly = processor.get_attention_by_weekday(attention_mean)
    plotWeights.plot_weekly_attention(attention_weekly, figure_name='Train_weekly_attention')

    attention_mean['split'], attention['split'] = 'train', 'train'
    attention_means.append(attention_mean)
    attentions.append(attention)

# %% [markdown]
# ### Validation

# %%
attention_mean, attention = processor.get_mean_attention(
    tft.interpret_output(validation_raw_predictions), validation_index, return_attention=True
)
plotWeights.plot_attention(
    attention_mean, figure_name='Validation_daily_attention', target_day=4
)

attention_mean['split'], attention['split'] = 'validation', 'validation'
attention_means.append(attention_mean)
attentions.append(attention)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Validation_weekly_attention')

# %% [markdown]
# ### Test

# %%
attention_mean, attention = processor.get_mean_attention(
    tft.interpret_output(test_raw_predictions), test_index, return_attention=True
)
plotWeights.plot_attention(attention_mean, figure_name='Test_daily_attention', target_day=4)
attention_mean['split'], attention['split'] = 'test', 'test'
attention_means.append(attention_mean)
attentions.append(attention)

# %%
attention_weekly = processor.get_attention_by_weekday(attention_mean)
plotWeights.plot_weekly_attention(attention_weekly, figure_name='Test_weekly_attention')

# %% [markdown]
# ### Dump

# %%
attention_mean_df = pd.concat(attention_means)
attention_mean_df.round(3).to_csv(os.path.join(args.figPath, 'attention_mean.csv'), index=False)

attention_df = pd.concat(attentions)
attention_df.round(3).to_csv(os.path.join(args.figPath, 'attention.csv'), index=False)

# %% [markdown]
# ## Variable Importance

# %% [markdown]
# ## Train

# %%
if args.interpret_train:
    interpretation = tft.interpret_output(train_raw_predictions, reduction="mean")
    for key in interpretation.keys():
        print(key, interpretation[key])
        
    figures = plotWeights.plot_interpretation(interpretation)
    for key in figures.keys():
        figure = figures[key]
        figure.savefig(os.path.join(plotter.figPath, f'Train_{key}.jpg'), dpi=DPI)

# %% [markdown]
# ## Test

# %%
interpretation = tft.interpret_output(test_raw_predictions, reduction="mean")
for key in interpretation.keys():
    print(key, interpretation[key])
    
figures = plotWeights.plot_interpretation(interpretation)
for key in figures.keys():
    figure = figures[key]
    figure.savefig(os.path.join(plotter.figPath, f'Test_{key}.jpg'), dpi=DPI)

# %%
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')


