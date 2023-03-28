# %% [markdown]
# # Imports

# %%
# python .\sensitivity_analysis.py --config=age_groups_old.json --input-file=../2022_May_age_groups_old/Top_100.csv --output=../results/age_subgroup/AGE019 --show-progress=True
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
# ## Pytorch lightning and forecasting

# %%
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# %% [markdown]
# # Load input

# %%
import json
import sys
sys.path.append( '..' )
from Class.Parameters import Parameters
from script.utils import *
from dataclasses import dataclass
from argparse import ArgumentParser

parser = ArgumentParser(description='Sensitivity analysis using Morris method')

parser.add_argument(
   '--config', help='config filename in the configurations folder',
   default='age_groups.json'
)
parser.add_argument(
   '--input-file', help='path of the input feature file',
   default='../2022_May_age_groups/Total.csv'
)
parser.add_argument(
   '--output', default='../results/age_subgroup/AGE1829',
   help='output result folder. Anything written in the scratch folder will be ignored by Git.'
)
parser.add_argument(
   '--show-progress', help='show the progress bar.',
   default=False, type=bool
)
arguments = parser.parse_args()

@dataclass
class args:
    result_folder = arguments.output
    figPath = os.path.join(result_folder, 'figures_morris')
    checkpoint_folder = os.path.join(result_folder, 'checkpoints')
    input_filePath = arguments.input_file

    configPath = os.path.join('../configurations', arguments.config)
    model_path = get_best_model_path(checkpoint_folder)

    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = arguments.show_progress

if not os.path.exists(args.figPath):
    os.makedirs(args.figPath, exist_ok=True)

# %%
start = datetime.now()
print(f'Started at {start}')

total_data = pd.read_csv(args.input_filePath)
print(total_data.shape)
print(total_data.head(3))

# %% [markdown]
# # Model

# %%
tft = TemporalFusionTransformer.load_from_checkpoint(args.model_path)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# %% [markdown]
# # Config

# %%
with open(args.configPath, 'r') as input_file:
  config = json.load(input_file)

parameters = Parameters(config, **config)

# this is only because the subgroups were trained using individual static features
parameters.data.static_features = tft.static_variables

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

# %% [markdown]
# ## Adapt input to encoder length
# Input data length needs to be a multiple of encoder length to created batch data loaders.

# %%
train_start = parameters.data.split.train_start
total_data = total_data[total_data['Date']>=train_start]
total_data[time_idx] = (total_data["Date"] - total_data["Date"].min()).apply(lambda x: x.days)

# %% [markdown]
# ## Train validation test split and scaling

# %%
train_data, validation_data, test_data = train_validation_test_split(
    total_data, parameters
)

# %%
train_scaled, _, _, target_scaler = scale_data(
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

  return dataloader

# %%
train_dataloader = prepare_data(train_scaled, parameters)
gc.collect()

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
# # Train results

# %%
print(f'\n---Training prediction--\n')

# [number of targets, number of examples, prediction length (15)]
train_predictions = tft.predict(
    train_dataloader, show_progress_bar=args.show_progress_bar
)

train_predictions = upscale_prediction(
    targets, train_predictions, target_scaler, max_prediction_length
)
gc.collect()

# %% [markdown]
# # Morris method

# %% [markdown]
# ## Scale

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

features = parameters.data.static_features + parameters.data.dynamic_features

minmax_scaler = MinMaxScaler()
train_minmax_scaled = minmax_scaler.fit_transform(train_data[features])

standard_scaler = StandardScaler().fit(train_data[features])

# %% [markdown]
# ## Calculate

# %%
# delta_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
delta_values = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5]
results = {
    'Delta': [],
    'Feature': [],
    'Mu_star':[],
    'Prediction_change':[],
    'Morris_sensitivity':[],
    'Absolute_mu_star': [],
    'Absolute_prediction_change':[],
    'Absolute_morris_sensitivity':[],
}

# %%
for delta in delta_values:
    print(f'---Delta {delta}---\n')
    for index, feature in enumerate(features):
        print(f'Feature {feature}')
        # add delta at min max scale
        data = train_minmax_scaled.copy()
        data[index] += delta
        data = minmax_scaler.inverse_transform(data) # return to original scale

        # replace the value in the standard normalized data
        data = standard_scaler.transform(data)
        train_scaled_copy = train_scaled.copy()
        train_scaled_copy[feature] = data[:, index]

        # infer on the changed data
        dataloader = prepare_data(train_scaled_copy, parameters)
        new_predictions = tft.predict(
            dataloader, show_progress_bar=args.show_progress_bar
        )
        # scale back to original
        new_predictions = upscale_prediction(
            targets, new_predictions, target_scaler, max_prediction_length
        )

        # sum up the change in prediction
        delta_prediction, abs_delta_prediction, N = 0, 0, 0
        for target_index in range(len(targets)):
            diff = new_predictions[target_index] - train_predictions[target_index]
            
            delta_prediction += np.sum(diff)
            abs_delta_prediction += np.sum(abs(diff))

            N += len(new_predictions[target_index]) * max_prediction_length
        
        mu_star = delta_prediction / (N*delta)
        # since delta is added to min max normalized value, std from same scaling is needed
        standard_deviation = train_minmax_scaled[:, index].std()
        morris = mu_star * standard_deviation

        print(f'Prediction change {delta_prediction:.5g}, mu_star {mu_star:.5g}, \
              sensitivity {morris:.5g}')

        results['Delta'].append(delta)
        results['Feature'].append(feature)

        results['Prediction_change'].append(delta_prediction)
        results['Mu_star'].append(mu_star)
        results['Morris_sensitivity'].append(morris)

        abs_mu_star = abs_delta_prediction / (N*delta)
        abs_morris = abs_mu_star * standard_deviation
        print(f'Absolute prediction change {abs_delta_prediction:.5g}, \
              mu {abs_mu_star:.5g}, morris {abs_morris:.5g}')
        
        results['Absolute_prediction_change'].append(abs_delta_prediction)
        results['Absolute_mu_star'].append(abs_mu_star)
        results['Absolute_morris_sensitivity'].append(abs_morris)
        print()
    #     break
    # break
    print()

# %% [markdown]
# ## Dump

# %%
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(args.figPath, 'Morris.csv'), index=False)
result_df

# %% [markdown]
# ## Plot

# %%
from Class.PlotConfig import *
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useOffset=True)
formatter.set_powerlimits((-3, 3))

# %%
for delta in delta_values:
    print(delta)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(features, result_df[result_df['Delta']==delta]['Morris_sensitivity'])
    
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel("Scaled Morris Index")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figPath, f'delta_{delta}.jpg'), dpi=200)
    plt.show()

# %% [markdown]
# # End

# %%
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')


