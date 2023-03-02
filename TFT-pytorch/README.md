# Introduction

This folder contains the `Temporal Fushion Transformer` implemented in [`PytorchForecasting`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html) framework. It supports dual prediction (case and death). However, the framework is generic, so both single and multiple outputs are easy to implement.

## Folder structure

* `2022_May`: Contains the merged feature files. After removing outliers from inputs.
  * `Total.csv`: All 3,142 counties.
  * `Top_N.csv`: Subset of the features that use top N counties by population.
* `2022_May_target_cleaned`: Contains the merged feature files. After removing outliers from both inputs and outputs.
  * Same files as `2022_May`
* `Class`
  * DataMerger
  * DataProcessor
  * Parameters
  * Plotter

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
* `config_2022_May.json`: Configuration file to reproduce the experiments using the raw dataset from [CovidMay17-2022](../dataset_raw/CovidMay17-2022/). This is new dataset that will be used in the experiment.

## Configuration

This section describes how the configuration files work. The purpose of the configuration files are

* Record TFT model and experiment parameters
  * So hidden layer size, loss metric, epoch, learning rate all are supposed to be here.
* Provide data feature maps and the feature file locations.
  * If you want to add or remove features, static or dynamic add the feature to corresponding raw file mapping here.
  * Unlike old code, this release can handle multiple features from a single file. E.g. you can replace `"Vaccination.csv": "VaccinationFull"` with `"Vaccination.csv": ["VaccinationFull", "VaccinationSingleDose"]` if  Or `Vaccination.csv` has feature columns for both of them.
* Paths and mapping for supporting files used, like `Population.csv`, `Rurality_Median_Mad.csv`.
* Start and end dates for train, test, validation split.

**`Note that`**, old config files would not run with these scripts.

## Demo

TODO

## Environment

### Runtime

Currently on Rivanna with batch size 64, each epoch with

* Top 100 counties takes around 2-3 minutes.
* Top 500 counties takes around 12-13 minutes, memory 24GB.
* Total 3,142 counties takes around 40-45 minutes, memory 32GB.

### Google Colab

A training notebook run on colab is shared [here](https://colab.research.google.com/drive/1yhI1PesOXYlB6iYXHre9zXMks1a4P6U2?usp=sharing). Feel free to copy and run on your colab and let me know if there are any issues.

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Uncomment the installation commands in the code or set `running_on_colab` to `True` in the code. Upload the TFT-pytorch folder in your drive and set that path in the notebook colab section.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the path accordingly in the notebook.

### Rivanna/CS server

On **Rivanna**, the default python environment doesn't have all the libraries we need. The [requirements.txt](../requirements.txt) file contains a list of libraries we need. There are two ways you can run the training there

#### Default Environment

Rivanna provides a bunch of python kernels readily available. You can check them from an interactive Jupyterlab session, on the top-right side of the notebook. I have tested with the `Tensorflow 2.8.0/Keras Py3.9` kernel and uncommented the following snippet in the code.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

You can choose different kernels and install the additional libraries. 

#### Virtual Environment

You can directly create a python virtual environment using the [environment.yml](../environment.yml) file and Anaconda. Then you won't have to install the libraries each time. Copy this file to your home directory and run the following command,

```bash
conda create --name <env> --file <this file>

# for example
conda create --name ml --file environment.yml

# then activate the environment with
conda activate ml
# now you should be able to run the files from your cmd line without error
# if you are on notebook select this environment as your kernel
```

You can also create the environment in this current directory, then the virtual environment will be saved in this folder instead, not in the home directory.

#### GPU 

Next, you might face issues getting GPU running on Rivanna. Even on a GPU server the code might not recognize the GPU hardware if cuda and cudnn are not properly setup. Try to log into an interactive session in a GPU server, then run the following command

```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If this is still 0, then you'll have to install the cuda and cudnn versions that match version in `nvidia-smi` command output. Also see if you tensorflow version is for CPU or GPU.

## How to replicate

* Write down your intended configuration in the config.json file. Or reuse an existing one.
  * Change TFT related parameters in the `"model_parameters"` section.
  * To train all 60 epochs change `"early_stopping_patience"` to 60. Default is `5`.
  * To add/remove new features or for different population cut use the `"data"` section.
  * To create rurality cut set `"rurality_cut"` to  `true`. Default is `false`. Since we are not using it currently.
  * `Data` section also has the train/validation/test split.
  * Use the `preprocess` section to remove outliers during the data preparation.
  * **Note**: Changing the order or spelling of the json keys will require chaning the [Parameters.py](/Class/Parameters.py) accordingly.
  
* Use the data prepration [notebook](/notebooks/Data_preparation.ipynb) or [script](/script/prepare_data.py) to create the merged data.
  * Make sure to pass the correct config.json file.
  * Check the folder paths in `args` class, whether they are consistent.
  * Depending on your configuration it can create the merged file of all counties, based on a population cut (e.g. top 500 counties) or rurality cut. All counties are saved in `Total.csv`, population cut in `Top_X.csv` where `X` is the number of top counties by population. Rurality is saved in `Rurality_cut.csv`.
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

## Usage guideline

* Please do not add temporarily generated files in this repository.
* Make sure to clean your tmp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git.
  * To check which files git says untracked `git status -u`. 
  * If you have folders you want to exclude add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.

## Features

Most of the features are results from the [COVID-19 Pandemic Vulnerability Index (PVI) Dashboard](https://covid19pvi.niehs.nih.gov/) maintained by National Institute of Environmental Health Sciences. They have two different versions of the dashboard model (11.2 and 12.4). Since model 12.4 only has data since 2021, we have used model 11.2. These are the features currently being used in this model following our recent [poster presentation](GPCE%20Poster%20at%20BII.pdf).

<div align="center">

| Feature        | Type       |
|:------------------------:|:------------:|
| Age Distribution       | Static     |
| Health Disparities     | Static     |
| Disease Spread         | Dynamic    |
| Social Distancing      | Dynamic    |
| Transmissible Cases    | Dynamic    |
| Vaccination Full Dose   | Dynamic    |
| SinWeekly | TimeEmbedding |
| CosWeekly | TimeEmbedding | 

</div>

<h3 class="accordion-toggle accordion-toggle-icon">Details of Features from PVI Model (11.2)</h4>
<div class="accordion-content">
<table class="pop_up_table" summary="Datasets comprising the current PVI model">
<thead>
<tr>
<th scope="col">Data Domain  <br /> Component(s)</th>
<th colspan="2" scope="col">Update Freq.</th>
<th scope="col">Description/Rationale</th>
<th scope="col">Source(s)</th>
</tr>

</thead>
<tbody>

<tr>
<td colspan="5"><strong>Age Distribution</strong></td>
</tr>
<tr>
<td>% age 65 and over</td>
<td>Static</td>
<td style="background: #9A42C8;"></td>
<td><em>Aged 65 or Older</em>. Older ages have been associated with more severe outcomes from COVID-19 infection.</td>
<td><span><a href="https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US_COUNTY.csv" target="_blank">2020 CDC Social Vulnerability Index</a></span></td>
</tr>

<tr>
<td colspan="5"><strong>Health Disparities</strong></td>
</tr>
<tr>
<td>Uninsured</td>
<td>Static</td>
<td style="background: #C885EC;"></td>
<td><em>Percentage uninsured in the total civilian noninstitutionalized population estimate. </em>. Individuals without insurance are more likely to be undercounted in infection statistics, and may have more severe outcomes due to lack of treatment.</td>
<td><span><a href="https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US_COUNTY.csv" target="_blank">2020 CDC Social Vulnerability Index</a></span></td>
</tr>

<tr>
<td colspan="5"><strong>Transmissible Cases</strong></td>
</tr>
<tr>
<td></td>
<td>Daily</td>
<td style="background: #CC3333;"></td>
<td><em>Cases from the last 14 days per 100k population</em>. Because of the 14-day incubation period, the cases identified in that time period are the most likely to be transmissible. This metric is the number of such &ldquo;contagious&rdquo; individuals relative to the population, so a greater number indicates more likely continued spread of disease.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a></span></span></td>
</tr>
<tr>
<td colspan="5"><strong>Disease Spread</strong></td>
</tr>
<tr>
<td></td>
<td>Daily</td>
<td style="background: #E64D4D;"></td>
<td><em>Fraction of total cases that are from the last 14 days (one incubation period)</em>. Because COVID-19 is thought to have an incubation period of about 14 days, only a sustained decline in new infections over 2 weeks is sufficient to signal reduction in disease spread. This metric is always between 0 and 1, with values near 1 during exponential growth phase, and declining linearly to zero over 14 days if there are no new infections.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a></span></span></td>
</tr>

<tr>
<td colspan="5"><strong>Social Distancing</strong></td>
</tr>
<tr>
<td></td>
<td>Daily</td>
<td style="background: #4258C9;"></td>
<td><em>Unacast social distancing scoreboard grade is assigned by looking at the change in overall distance travelled and the change in nonessential visits relative to baseline (previous year), based on cell phone mobility data</em>. The grade is converted to a numerical score, with higher values being less social distancing (worse score) is expected to increase the spread of infection because more people are interacting with other.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">Unacast</a></span></td>
</tr>

<tr>
<td colspan="5"><strong>Vaccination Full Dose</strong></td>
</tr>
<tr>
<td>Series_Complete_Pop_Pct</td>
<td>Daily</td>
<td style="background: #4258C9;"></td>
<td> Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">CDC</a></span></td>
</tr>

</tbody>
</table>
