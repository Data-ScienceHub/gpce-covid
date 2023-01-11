# %% [markdown]
# # Imports

# %%
import pandas as pd

# disable chained assignments
pd.options.mode.chained_assignment = None 
import os, gc
from darts import TimeSeries
import optuna

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from datetime import datetime

# this stops pytorch from logging GPU info each time your model predicts something
# https://github.com/Lightning-AI/lightning/issues/3431
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# for some warning bugs from darts
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from utils import *
from splits import *
from plotter import *

# make sure to set these False for scripts, otherwise it'll print lots of logs
VERBOSE = False
Split = Baseline

# %%
# ## Result folder
output_folder = 'results/tune_NBEATS'
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# %% [markdown]
# # Config

# %%
@dataclass
class Config:
    static_features = ['AgeDist', 'HealthDisp']
    past_features = ['DiseaseSpread', 'Transmission', 'VaccinationFull', 'SocialDist']
    known_future = ['SinWeekly', 'CosWeekly']
    time_index = 'TimeFromStart' # note that this is an index feature commonly used by all timeseries models

    features =  static_features + past_features + known_future
    targets = ['Cases']
    group_id = 'FIPS'
    selected_columns = features + targets
    input_sequence_length = 13
    output_sequence_length = 15
    epochs = 60
    early_stopping_patience = 3
    n_trials = 25
    seed = 7

seed_everything(Config.seed)
targets = Config.targets
group_id = Config.group_id
time_index = Config.time_index
input_sequence_length = Config.input_sequence_length
output_sequence_length = Config.output_sequence_length

# %% [markdown]
# # Preprocessing

# %%
df = pd.read_csv('../TFT-pytorch/2022_May_cleaned/Total.csv')
df['Date'] = to_datetime(df['Date'])
df[time_index] = df[time_index].astype(int)

print(df.head(3))

# %% [markdown]
# ## Split and scale

# %%
train_df, val_df, test_df = split_data(df, Split, input_sequence_length)
train_df, val_df, test_df, feature_scaler, target_scaler = scale_data(
    train_df, val_df, test_df, Config.features, targets
)

# %% [markdown]
# ## Create covariates

# %%
from numpy import round, mean, float32

def get_covariates(df:pd.DataFrame, tail_cut=False):
    if tail_cut:
        cutoff = df[time_index].max() - output_sequence_length + 1
        df = df[df[time_index]<cutoff]

    series = TimeSeries.from_group_dataframe(
        df, time_col=time_index, group_cols=group_id,
        static_cols=Config.static_features, value_cols=targets,
    )
    past_covariates = TimeSeries.from_group_dataframe(
        df, group_cols=group_id,
        time_col = time_index, value_cols=Config.past_features
    )

    # timeseries has default precision float64, this doesn't match 
    # with pl trainer which has precision float32
    for covariates in [series, past_covariates]:
        for index in range(len(covariates)):
            covariates[index] = covariates[index].astype(float32)

    return series, past_covariates

# %%
train_series, train_past_covariates = get_covariates(train_df)
val_series, val_past_covariates = get_covariates(val_df)

# %% [markdown]
# # Tune

# %% [markdown]
# ## Build

# %%
from darts.models import NBEATSModel
from pytorch_lightning.trainer import Trainer
from torch.nn.modules import MSELoss
from torch.optim import Adam

# %%
def create_model(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0, 0.3, step=0.1)
    layers = trial.suggest_int("layers", 2, 4, step=1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    return NBEATSModel(
        input_chunk_length=input_sequence_length, 
        output_chunk_length=output_sequence_length,
        loss_fn=MSELoss(), optimizer_cls=Adam,
        batch_size=batch_size, num_layers=layers, 
        dropout=dropout,   
        optimizer_kwargs={'lr': learning_rate}
    )

# %% [markdown]
# ## Util

# %%
from tqdm.auto import tqdm

def historical_forecast(
    model, df, series_list, past_list, target_scaler
):
    prediction_start = df[time_index].min() + input_sequence_length

    preds = []
    fips_codes = df[group_id].unique()

    for index in tqdm(range(len(fips_codes)), disable=not VERBOSE):
        fips, series, past = fips_codes[index], series_list[index], past_list[index]

        if len(series) > (input_sequence_length + output_sequence_length):
            # list of predictions with sliding window
            county_preds = model.historical_forecasts(
                series, 
                past_covariates=past,
                start=prediction_start,
                retrain=False, last_points_only=False, verbose=False,
                forecast_horizon=output_sequence_length, stride=1,
            )
            # reseting index here is ok since only one time column
            county_preds = pd.concat(
                [pred.pd_dataframe().reset_index() for pred in county_preds], axis=0
            )
        else:
            county_preds = model.predict(
                output_sequence_length,
                series, n_jobs=-1,
                past_covariates=past,
                verbose=False
            )
            county_preds = county_preds.pd_dataframe().reset_index()

        county_preds[group_id] = fips
        preds.append(county_preds)

    # conver the predicted list to a dataframe
    preds = pd.concat(preds, axis=0).reset_index(drop=True)
    # scale up
    # preds[targets] = target_scaler.inverse_transform(
    #     preds[targets].values
    # )
    # round and remove negative targets since infection can't be neg
    preds[targets] = preds[targets].apply(round)
    for target in targets:
        preds.loc[preds[target]<0, target] = 0
        
    # since this is sliding window, some cases will have multiple prediction with different forecast horizon
    preds = preds.groupby([group_id, time_index], axis=0)[targets].aggregate(mean)

    preds.rename({target:'Predicted_'+target for target in targets}, axis=1, inplace=True)

    target_df = df[[group_id, time_index, 'Date'] + targets].copy().reset_index(drop=True)
    # target_df[targets] = target_scaler.inverse_transform(target_df[targets]).astype(int)

    merge_keys = [group_id, time_index]
    prediction_df = preds.merge(target_df[['Date'] + merge_keys + targets], on=merge_keys, how='inner')
    gc.collect()

    return prediction_df

# %% [markdown]
# ## Training

# %%
def objective(trial):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=Config.early_stopping_patience,
        min_delta=0
    )

    checkpoint = ModelCheckpoint(
        dirpath=output_folder, monitor="val_loss"
    )

    model = create_model(trial)
    val_series, val_past_covariates = get_covariates(val_df)
    model.fit(
        train_series, val_series=val_series, verbose=False,
        past_covariates=train_past_covariates, val_past_covariates=val_past_covariates,
        trainer = Trainer(
            accelerator= "auto", max_epochs=Config.epochs,
            callbacks=[early_stopping, checkpoint], 
            logger=False, enable_progress_bar=False
        )
    )
    # gc.collect()
    model.load(checkpoint.best_model_path)
    val_series, val_past_covariates = get_covariates(val_df, tail_cut=True)
    val_prediction_df = historical_forecast(
        model, val_df, val_series, val_past_covariates, target_scaler
    )

    val_loss = mean_squared_error(
        val_prediction_df['Cases'], val_prediction_df["Predicted_Cases"]
    )
    return val_loss

# %%
study_name = 'nbeats'
storage_name = f"sqlite:///{study_name}.db"
load_only = False

if load_only:
    study = optuna.load_study(
        study_name=study_name, storage=storage_name
    )
else:
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction='minimize', load_if_exists=True
    )
    study.optimize(
        objective, n_trials=Config.n_trials, n_jobs=-1, 
        gc_after_trial=True, show_progress_bar=VERBOSE
    )

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.round(6).to_csv(os.path.join(output_folder, 'trials.csv'), index=False)
