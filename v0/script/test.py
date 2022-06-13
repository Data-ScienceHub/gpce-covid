import tensorflow as tf
import os, gc, json
import pandas as pd
from pandas import to_datetime
from utils import train_validation_test_split
import argparse
from utils import scale_back, calculate_result, sumCases

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Apply the default theme
sns.set_theme()
sns.set(font_scale = 1.5)

import sys
sys.path.append( '..' )
from Class.Trainer import Trainer
from Class.ParameterManager import ParameterManager
from Class.DataProcessor import DataProcessor
from Class.Plotter import PlotResults, PlotWeights

def main():
    parser = argparse.ArgumentParser(
        description='Test Temporal Fusion Transformer model on covid dataset', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--configPath',help='Path to the json config file', 
        type=str, default='../config_2022_May.json'
    )
    parser.add_argument(
        '-d', '--dataPath', help='Directory where input feature file is located', 
        type=str, default='../2022_May/Population_cut.csv'
    )
    parser.add_argument(
        '-o', '--outputPath', help='Directory where outputs will be saved. This path will be created if it does not exist',
        type=str, default='../output'
    )

    parser.add_argument(
        '-p', '--checkpoint', help='Directory where checkpoints will be saved',
        type=str, default='../output/checkpoints'
    )

    args = parser.parse_args()

    # output paths
    checkpoint_folder = args.checkpoint
    figure_folder = os.path.join(args.outputPath, "figures")

    # this eventually creates output folder if it doesn't exist
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder, exist_ok=True)

    print(f'Loading config.json from {args.configPath}')
    with open(args.configPath) as inputfile:
        config = json.load(inputfile)
        inputfile.close()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    """## Load data"""
    print(f'Loading input data from {args.dataPath}')
    df = pd.read_csv(args.dataPath)
    print(f'Input feature file shape {df.shape}')

    # scale vaccination percentage within 0 to 1
    df['Date'] = to_datetime(df['Date']) 
    df['FIPS'] = df['FIPS'].astype(str)

    parameterManager = ParameterManager(config)
    print(f'Column mappings: {parameterManager.col_mappings}\n')

    """# Train validation split and Scaling"""
    train_data, validation_data, test_data, target_scaler = train_validation_test_split(df, parameterManager, scale=True)
    print(f'Number train data is {train_data.shape[0]}, validation {validation_data.shape[0]}, test {test_data.shape[0]}')

    gc.collect()
    
    trainer = Trainer(parameterManager)
    model = trainer.create_model()
    
    optimizer_params = parameterManager.optimizer_params
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=optimizer_params['learning_rate'], clipnorm=optimizer_params['clipnorm']
    )

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpointManager = tf.train.CheckpointManager(checkpoint, checkpoint_folder, max_to_keep=1)

    model = trainer.load_from_checkpoint(checkpoint, checkpointManager.latest_checkpoint)
    if model is None:
        sys.exit(-1)

    
    """# Create batches"""
    dataProcessor = DataProcessor(
        parameterManager.total_sequence_length, parameterManager.col_mappings, parameterManager.data_params
    )
    """### Train"""
    train_batch = dataProcessor.prepare_batch(train_data)
    train_preds, train_actuals, train_attn_weights = trainer.predict(model, train_batch)

    train_actuals = scale_back(train_actuals, target_scaler, parameterManager.target_sequence_length)
    train_preds = scale_back(train_preds, target_scaler, parameterManager.target_sequence_length)
    
    train_mae, train_rmse, train_smape = calculate_result(train_actuals, train_preds)
    print(f'Train MAE {train_mae}, RMSE {train_rmse}, SMAPE {train_smape}')
    gc.collect()

    """### Validation"""
    validation_batch = dataProcessor.prepare_batch(validation_data)
    validation_preds, validation_actuals, _ = trainer.predict(model, validation_batch)

    validation_preds = scale_back(validation_preds, target_scaler, parameterManager.target_sequence_length)
    validation_actuals = scale_back(validation_actuals,  target_scaler, parameterManager.target_sequence_length)
    
    validation_mae, validation_rmse, validation_smape = calculate_result(validation_actuals, validation_preds)
    print(f'Validation MAE {validation_mae}, RMSE {validation_rmse}, SMAPE {validation_smape}')

    """### Test"""

    test_batch = dataProcessor.prepare_batch(test_data)
    test_preds, test_actuals, _ = trainer.predict(model, test_batch)

    test_actuals = scale_back(test_actuals, target_scaler, parameterManager.target_sequence_length) 
    test_preds = scale_back(test_preds, target_scaler, parameterManager.target_sequence_length)

    test_mae, test_rmse, test_smape = calculate_result(test_actuals, test_preds)
    print(f'Test MAE {test_mae}, RMSE {test_rmse}, SMAPE {test_smape}')

    del model
    gc.collect()

    """## Plot"""
    number_of_locations = df[parameterManager.col_mappings['ID']].nunique().values[0]
    print(f'Number of locations {number_of_locations}')
    locs = df[parameterManager.col_mappings['ID']].iloc[:number_of_locations, 0].values

    """
    Train prediction
    """
    targets, predictions = sumCases(train_actuals, train_preds, number_of_locations)

    resultPlotter = PlotResults(targets, predictions, parameterManager.train_start, locs, figure_folder)
    plot_title = f'Summed plot (train) MAE {train_mae:0.3f}, RMSE {train_rmse:0.3f}'

    resultPlotter.makeSummedPlot(plot_title, figure_name='Summed plot - train', figsize=(24, 8))

    """
    Validation prediction
    """
    targets, predictions = sumCases(validation_actuals, validation_preds, number_of_locations)
    resultPlotter = PlotResults(targets, predictions, parameterManager.validation_start, locs, figure_folder)
    plot_title = f'Summed plot (Validation) MAE {validation_mae:0.3f}, RMSE {validation_rmse:0.3f}, SMAPE {validation_smape:0.3f}'

    resultPlotter.makeSummedPlot(plot_title, figure_name='Summed plot - validation')

    """
    Test prediction
    """
    targets, predictions = sumCases(test_actuals, test_preds, number_of_locations)
    PlotC = PlotResults(targets, predictions, parameterManager.test_start, locs, figure_folder)
    plot_title = f'Summed plot (Validation) MAE {validation_mae:0.3f}, RMSE {validation_rmse:0.3f}, SMAPE {validation_smape:0.3f}'

    PlotC.makeSummedPlot(plot_title, figure_name='Summed plot - test')

    """
    Interpret
    """
    plotter = PlotWeights(parameterManager.col_mappings, train_attn_weights, figure_folder)
    """## Static variables"""

    plotter.plot_static_weights()

    """## Future known input"""

    plotter.plot_future_weights()

    """## Observed weights"""

    plotter.plotObservedWeights()

if __name__ == '__main__':
    main()