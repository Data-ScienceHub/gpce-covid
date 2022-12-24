# %% [markdown]
# # Imports

# %%
import pandas as pd
# disable chained assignments
pd.options.mode.chained_assignment = None 
import os, gc
import optuna, optuna_dashboard

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

from models import build_BiLSTM
from utils import *
from splits import *

SEED = 7
tf.random.set_seed(SEED)
VERBOSE = 0
Split = Baseline

# %% [markdown]
# ## Result folder

# %%
output_folder = 'scratch/tune_BiLSTM'
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# %% [markdown]
# # Preprocessing

# %%
df = pd.read_csv('../TFT-pytorch/2022_May_cleaned/Top_100.csv')
df['Date'] = pd.to_datetime(df['Date'])

# %% [markdown]
# ## Config

# %%
from dataclasses import dataclass
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
    epochs = 200
    early_stopping_patience = 5
    loss = 'mse'
    n_trials = 2

targets = Config.targets
group_id = Config.group_id
input_sequence_length = Config.output_sequence_length
output_sequence_length = Config.output_sequence_length
output_size = len(targets) * output_sequence_length

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

# %% [markdown]
# # Training

# %% [markdown]
# ## Model

# %%
def create_model(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
    dropout = trial.suggest_float("dropout", 0, 0.3, step=0.1)
    layers = trial.suggest_int("layers", 2, 4, step=1)

    model = build_BiLSTM(
        x_train.shape[1:], output_size=output_size, loss=Config.loss, 
        hidden_size=hidden_size, dropout=dropout, 
        learning_rate=learning_rate, layers=layers
    )
    return model

def create_dataset(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_data = cache_data(
        x_train, y_train, batch_size=batch_size, 
        buffer_size=Config.buffer_size
    )
    val_data = cache_data(
        x_val, y_val, batch_size=batch_size, 
    )
    return train_data, val_data

def objective(trial):
    model = create_model(trial)
    train_data, val_data = create_dataset(trial)
    
    early_stopping = EarlyStopping(
        patience = Config.early_stopping_patience, 
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_folder, 'model.h5'), 
        save_best_only=True, save_weights_only=True
    )
    model.fit(
        train_data, validation_data=val_data,
        epochs=2,  
        callbacks=[early_stopping, model_checkpoint],
        verbose=VERBOSE
    )
    model.load_weights(model_checkpoint.filepath)
    val_loss = model.evaluate(val_data, verbose=VERBOSE)

    return val_loss

# %%
study_name = 'BiLSTM'
storage_name = f"sqlite:///{study_name}.db"

study = optuna.create_study(
    study_name=study_name, storage=storage_name, direction='minimize', load_if_exists=True
)
study.optimize(
    objective, n_trials=Config.n_trials,
    gc_after_trial=True, show_progress_bar=(VERBOSE==1)
)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Parameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
fig =optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_optimization_history(study)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.round(6).to_csv(os.path.join(output_folder, 'trials.csv'), index=False)


