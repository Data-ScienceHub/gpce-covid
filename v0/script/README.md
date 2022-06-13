# Introduction

This folder contains the scripts necessary to run the experiments.

## Data preparation

The `prepare_data.py` script prepared the merged data from the raw features files in the [dataset_raw](../../dataset_raw/) folder. This can create three types of features based on your configuration json.

* `Total.csv`: Contains feature for all counties.
* `Population_cut.csv`: Contains feature for top counties by population as specified by `"Population cut"` value in the config.json file. This is useful for small scale experiments.
* `Rurality_cut.csv`: Contains feature for counties filtered by rurality score and MAD range. Determined by `"Rurality cut"`, `"RuralityRange"` and `"MADRange"` keys.

```bash
Create merged feature file

options:
-h, --help            show this help message and exit
-c CONFIGPATH, --configPath CONFIGPATH
                        Path to the json config file (default: ../config_2022_May.json)
-d DATAPATH, --dataPath DATAPATH
                        Directory where raw input feature files are located (default: ../../dataset_raw/CovidMay17-2022)
-o OUTPUTPATH, --outputPath OUTPUTPATH
                        Directory where outputs will be saved. This path will be created if it does not exist. (default: ../2022_May/)
-s SUPPORTPATH, --supportPath SUPPORTPATH
                        Directory where input support files (e.g. population, rurality) are located (default: ../../dataset_raw/Support files)
```

## Train and interpret

Use the `train.py` to train the model on merged data created from pervius data preparation script. This saves the best model based on validation loss, then early stops if validation loss doesn't improve for few epochs. After training is finished this script also checks the results on train, validation and test data. Then interprets the results. The default arguments are good to go. Run using `python train.py`.

```bash
Train the Temporal Fusion Transformer model on covid dataset

options:
  -h, --help            show this help message and exit
  -c CONFIGPATH, --configPath CONFIGPATH
                        Path to the json config file (default: ../config_2022_May.json)
  -d DATAPATH, --dataPath DATAPATH
                        Directory where input feature file is located (default: ../2022_May/Population_cut.csv)
  -o OUTPUTPATH, --outputPath OUTPUTPATH
                        Directory where outputs will be saved. This path will be created if it does not exist (default: ../output)
  -r RESTORE, --restore RESTORE
                        Whether the model should restore from a checkpoint (default: False)
  -p CHECKPOINT, --checkpoint CHECKPOINT
                        Directory where checkpoints will be saved (default: ../output/checkpoints)
```

## Test/Inference

Use the `test.py` to load the trained model from checkpoint and check the results on train, validation and test data. Then interpret the results. The default arguments are good to go. Run using `python test.py`.

```bash
Test Temporal Fusion Transformer model on covid dataset

options:
  -h, --help            show this help message and exit
  -c CONFIGPATH, --configPath CONFIGPATH
                        Path to the json config file (default: ../config_2022_May.json)
  -d DATAPATH, --dataPath DATAPATH
                        Directory where input feature file is located (default: ../2022_May/Population_cut.csv)
  -o OUTPUTPATH, --outputPath OUTPUTPATH
                        Directory where outputs will be saved. This path will be created if it does not exist (default: ../output)
  -p CHECKPOINT, --checkpoint CHECKPOINT
                        Directory where checkpoints will be saved (default: ../output/checkpoints)
```