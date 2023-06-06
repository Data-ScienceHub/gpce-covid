# Introduction

This folder contains the `Temporal Fushion Transformer` implemented in [`PytorchForecasting`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html) framework. It supports dual prediction (case and death). However, the framework is generic, so both single and multiple outputs are easy to implement.

## Folder structure
* `2022_May_cleaned`: Contains the merged feature files. After removing outliers from both inputs and outputs.
  * `Total.csv`: All 3,142 counties.
  * `Top_N.csv`: Subset of the features that use top N counties by population.
* `Class`
  * `DataMerger`: Merges raw data files into a single csv file for model training.
  * `Parameters`: Handles training, data and experiment parameters.
  * `PlotConfig`: Basic matplotlib plot configurations.
  * `Plotter`: Plot model output, attention and others.
  * `PredictionProcessor`: Process multi-horizon output from the models.

* `configurations`: Folder to save some common configurations.
* `notebooks`: Notebook version of the scripts. Use these for debugging or new implementation purpose.
  * Data preparation.ipynb
  * Train.ipynb
  * Inference.ipynb
* `output`
  * `checkpoints`
    * epoch=X-step=X.ckpt: model checkpointed by best validation loss.
    * model.ckpt: final model saved after finishing traning.
  * `figures`: saves the figures plotted by the final model obtained after finishing the training.
  * `figures_best`: figures plotted using the model with best validation loss. 
  * `lightning_logs`: This folder is used by tensorboard to log the training and validation visualization. You can point this folder by clicking the line before `import tensorboard as tb` in the training code (both script and notebook), that says `launch tensorboard session`. VSCode will automatically suggest the extensions needed for it. It can also run from cmd line, using `tensorboard --logdir=lightning_logs`, then it'll show something like `TensorBoard 2.9.0 at http://localhost:6006/ (Press CTRL+C to quit)`. Copy paste the URL in your local browser. To save the images, check `show data download links in the top left`.
  
* `script`: Contains scripts for submitting batch jobs. For details on how to use then, check the readme inside the folder.
  * `prepare_data.py`: Prepare merged data from raw feature files.
  * `train.py`: Train model on merged data, then interpret using the best model by validation loss.
  * `inference.py`: Inference from a saved checkpoint.
  * `utils.py`: Contains utility methods.
* `config_2022_May.json`: Configuration file to reproduce the experiments using the raw dataset from [CovidMay17-2022](../dataset_raw/CovidMay17-2022/). This is dataset that will be used in the experiment.

## Configuration

This section describes how the configuration files work. The purpose of the configuration files are

* Record TFT model and experiment parameters
  * So hidden layer size, loss metric, epoch, learning rate all are supposed to be here.
* Provide data feature maps and the feature file locations.
  * If you want to add or remove features, static or dynamic add the feature to corresponding raw file mapping here.
  * Unlike old code, this release can handle multiple features from a single file. E.g. you can replace `"Vaccination.csv": "VaccinationFull"` with `"Vaccination.csv": ["VaccinationFull", "VaccinationSingleDose"]` if  Or `Vaccination.csv` has feature columns for both of them.
* Paths and mapping for supporting files used, like `Population.csv`.
* Start and end dates for train, test, validation split.

## How to replicate

* Write down your intended configuration in the config.json file. Or reuse an existing one.
  * Change TFT related parameters in the `"model_parameters"` section.
  * To add/remove new features or for different population cut use the `"data"` section.
  * `Data` section also has the train/validation/test split.
  * Use the `preprocess` section to remove outliers during the data preparation.
  * **Note**: Changing the order or spelling of the json keys will require chaning the [Parameters.py](/Class/Parameters.py) accordingly.
  
* Use the data prepration [notebook](/notebooks/Data_preparation.ipynb) or [script](/script/prepare_data.py) to create the merged data.
  * Make sure to pass the correct config.json file.
  * Check the folder paths in `args` class, whether they are consistent.
  * Depending on your configuration it can create the merged file of all counties, based on a population cut (e.g. top 500 counties) or rurality cut. All counties are saved in `Total.csv`, population cut in `Top_X.csv` where `X` is the number of top counties by population. 
  * Currently there is a option to either remove outliers from the input and target, or not. Removing target outliers can decrease anomalies in the learning. But changing the ground truth like this is not often desirable, so you can set it to false in the `preprocess` section in the configuration.
  * Note that, scaling and splitting are not done here, but later during training and infering.
  
* Use the training [notebook](/notebooks/Train.ipynb) or [script](/script/train.py) to train and interpret the model.
  * Make sure to pass the correct config.json file.
  * Check the folder paths in `args` class, whether they are consistent.
  * This file reads the merged feature file, splits it into train/validation/test, scales the input and target if needed, then passes them to the model.
  * The interpretation is done for both the final model and the model with best validation loss.
  * Note the path where the models are checkpointed.
  * Using vscode you can also open the tensorboard to review the training logs.
  * The prediction is saved as csv file for any future visualizations.
  
* The inference [notebook](/notebooks/Inference.ipynb) or [script](/script/inference.py) can be used to infer a previously checkpointed model and interpret it.
  * Same as before, recheck the config.json file and `args` class. Make sure they are same as the model you are going to infer.
  * Set the model path to the checkpoint model you want to infer.