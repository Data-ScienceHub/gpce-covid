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
# ## Pytorch lightning and forecasting

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss, MultiLoss, MultivariateDistributionLoss, RMSE

# %% [markdown]
# # Load input

# %%
from dataclasses import dataclass

@dataclass
class args:
    outputPath = '../results/DeepVAR_top_100'
    figPath = os.path.join(outputPath, 'figures')
    checkpoint_folder = os.path.join(outputPath, 'checkpoints')
    input_filePath = '../2022_May_cleaned/Top_100.csv'
    configPath = '../configurations/baseline.json'

    # Path/URL of the checkpoint from which training is resumed
    ckpt_model_path = None # os.path.join(checkpoint_folder, 'latest-epoch=2.ckpt')
    
    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = False

@dataclass
class Config:
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3
    early_stopping_patience = 3

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
model_params = parameters.model_parameters

max_prediction_length = model_params.target_sequence_length
max_encoder_length = model_params.input_sequence_length

# %% [markdown]
# # Seed

# %%
import random

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

seed_torch(model_params.seed)

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
total_data[time_idx] = (total_data["Date"] - train_start).apply(lambda x: x.days)

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
    # static_categoricals=["FIPS"],
    time_varying_known_reals=pm.data.time_varying_known_features, # known features go to encoder. for DeepAR set(encoder) - set(targets) == set(decoder)
    time_varying_unknown_reals = targets, # unknown features got to decoder
    target_normalizer = MultiNormalizer(
      [GroupNormalizer(groups=pm.data.id) for _ in range(len(targets))]
    )
  )

  # batch_sampler="synchronized" is a must for DeepAR or DeepVAR here
  if train:
    dataloader = data_timeseries.to_dataloader(
      train=True, batch_size=model_params.batch_size, batch_sampler="synchronized"
    )
  else:
    dataloader = data_timeseries.to_dataloader(
      train=False, batch_size=model_params.batch_size*8, batch_sampler="synchronized"
    )

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
# ## Trainer and checkpointer

# %%
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=model_params.early_stopping_patience
    , verbose=True, mode="min"
)

# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
best_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=args.checkpoint_folder, monitor="val_loss", filename="best-{epoch}"
)
# latest_checkpoint = pl.callbacks.ModelCheckpoint(
#     dirpath=args.checkpoint_folder, every_n_epochs=1, filename="latest-{epoch}"
# )

logger = TensorBoardLogger(args.outputPath)  # logging results to a tensorboard

# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
trainer = pl.Trainer(
    max_epochs = model_params.epochs,
    accelerator = 'auto',
    enable_model_summary=True,
    gradient_clip_val = model_params.clipnorm,
    callbacks = [early_stop_callback, best_checkpoint],
    enable_progress_bar = args.show_progress_bar,
    check_val_every_n_epoch = 1
)

# %% [markdown]
# ## Model

# %%
@dataclass
class Model_Config:
    epochs = 10
    learning_rate = 1e-4
    early_stopping_patience = 3
    rnn_layers = 3
    hidden_size = 64
    dropout = 0
    optimizer = 'adam'

# %%
# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html
model = DeepAR.from_dataset(
    train_timeseries,
    rnn_layers = Model_Config.rnn_layers, 
    learning_rate= Model_Config.learning_rate,
    hidden_size= Model_Config.hidden_size,
    dropout=Model_Config.dropout,
    optimizer=Model_Config.optimizer,
    log_interval=1,
     # Multivariate loss is what makes this DeepAR a DeepVAR model
    loss=MultiLoss([MultivariateNormalDistributionLoss() for _ in targets])
)

print(f"Number of parameters in network: {model.size()/1e3:.1f}k")

# %%
from datetime import datetime

gc.collect()

start = datetime.now()
print(f'\n----Training started at {start}----\n')

trainer.fit(
    model,
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
# # Evaluate - best model

# %%
best_model_path = trainer.checkpoint_callback.best_model_path
print(f'Loading best model from {best_model_path}')
model = DeepAR.load_from_checkpoint(best_model_path)

# %% [markdown]
# ## PlotResults

# %%
from Class.Plotter import *

plotter = PlotResults(args.figPath, targets, show=args.show_progress_bar)

# %% [markdown]
# ## Train results

# %%
# not a must, but increases inference speed 
_, train_dataloader = prepare_data(train_scaled, parameters) 
print(f'\n---Training results--\n')

train_predictions, train_index = model.predict(
    train_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)

# %%
train_predictions = upscale_prediction(targets, train_predictions, target_scaler, max_prediction_length)
train_result_merged = processor.align_result_with_dataset(train_data, train_predictions, train_index)
show_result(train_result_merged, targets)

plotter.summed_plot(train_result_merged, type='Train_error', plot_error=True)
gc.collect()

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')
validation_predictions, validation_index = model.predict(
    validation_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)
validation_predictions = upscale_prediction(targets, validation_predictions, target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged, targets)
plotter.summed_plot(validation_result_merged, type='Validation')
gc.collect()

# %% [markdown]
# ## Test results

# %%
print(f'\n---Test results--\n')
test_predictions, test_index = model.predict(
    test_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)
test_predictions = upscale_prediction(targets, test_predictions, target_scaler, max_prediction_length)

test_result_merged = processor.align_result_with_dataset(test_data, test_predictions, test_index)
show_result(test_result_merged, targets)
plotter.summed_plot(test_result_merged, 'Test')
gc.collect()

# %% [markdown]
# ## Dump results

# %%
train_result_merged['split'] = 'train'
validation_result_merged['split'] = 'validation'
test_result_merged['split'] = 'test'
df = pd.concat([train_result_merged, validation_result_merged, test_result_merged])
df.to_csv(os.path.join(args.outputPath, 'predictions_case_death.csv'), index=False)

df.head()

# %%
del train_predictions, validation_predictions, test_predictions
del train_result_merged, validation_result_merged, test_result_merged, df
gc.collect()

# %%
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')


