# Interpreting County Level COVID-19 Infection and Feature Sensitivity using Deep Learning Time Series Models

## Introduction
This work combines sensitivity analysis with heterogeneous time-series deep learning model prediction, which corresponds to the interpretations of Spatio-temporal features from what the model has actually learned. We forecast county-level COVID-19 infection using the Temporal Fusion Transformer (TFT). We then use the sensitivity analysis extending Morris Method to see how sensitive the outputs are with respect to perturbation to our static and dynamic input features. We have collected more than 2.5 years of socioeconomic and health features over 3142 US counties. Using the proposed framework, we conduct extensive experiments and show our model can learn complex interactions and perform predictions for daily infection at the county level. 

## Folder Structure

* **Archives**: Unused codes.
* **dataset_raw**: Contains the collected raw dataset and the supporting files. To update use the [Update dynamic dataset](/dataset_raw/Update%20dynamic%20features.ipynb) notebook. Static dataset is already update till the onset of COVID-19 using [Update static dataset](/dataset_raw/Update%20static%20features.ipynb) notebook.
* **papers**: Related papers. 
* **Related Works**: Contains the models and results used to compare the TFT performance with related works. 
* **TFT-PyTorch**: Contains all codes and merged feature files used during the TFT experimentation setup and interpretation. For more details, check the [README.md](/TFT-PyTorch/README.md) file inside it. The primary results are highlighted in [results.md](/TFT-PyTorch/results.md). 


## Reproduce

### Create Virtual Environment
First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`.

#### Option A: Anaconda
If you have `Anaconda` installed locally, follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). An example code,

```
conda create -n ml python=3.10
conda activate ml
```
This will activate the venv `ml`.


#### Option B: Python

If you only have `python` installed but no `pip`, installed pip and activate a virtual env using the following commands from [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/),

On linux/macOS :

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv ml
source ml/bin/activate
python3 -m pip install -r requirements.txt
```

On windows :
```bash
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv ml
.\env\Scripts\activate
py -m pip install -r requirements.txt
```

Follow the instructions [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to make sure you have the `pip` and `virtualenv` installed and working. Then create a virtual environement (e.g. name ml) or install required libraries in the default env, using the 

### Install Libraries
Once you have the virtual environment created and running, you can download the libraries using, the [requirement.txt](/requirements.txt) file. 

On linux/macOS :

```bash
python3 -m pip install -r requirements.txt
```

On windows :
```bash
py -m pip install -r requirements.txt
```

You can test whether the environment has been installed properly using a small dataset in the [`train.py`](/TFT-pytorch/script/train_simple.py) file.

### Installing CUDA
The default versions installed with `pytorch-forecasting` might not work and print cpu instead for the following code. Since it doesn't install CUDA with pytorch.

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} backend')
```

In such case, replace existing CUDA with the folowing version. Anything newer didn't work for now.
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Singularity

You can either pull the singularity container from the remote library,
```bash
singularity pull tft_pytorch.sif library://khairulislam/collection/tft_pytorch:latest
```
Or create the container locally using the [singularity.def](/TFT-pytorch/singularity.def) file. Executeg the following command. This uses the definition file to create the container from scratch. Note that is uses `sudo` and requires root privilege. After compilation, you'll get a container named `tft_pytorch.sif`. 

```bash
sudo singularity build tft_pytorch.sif singularity.def
```

Then you can use the container to run the scripts. For example, 
```bash
cd original-TFT-baseline/script/

singularity run --nv ../../tft_pytorch.sif python train.py --config=baseline.json --output=../scratch/TFT_baseline
```

### Google Colab

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Add the following installation commands in the code. Upload the TFT-pytorch folder in your drive and set that filepaths accordingly.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the paths accordingly in the notebook.

## Features

Note that, past values of target and known futures are also used as observed inputs by TFT.

<div align="center">

<table border="1">
<caption> <h2>Details of Features </h2> </caption>
<thead style="border:2px solid">
<tr>
<th>Feature</th>
<th>Type</th>
<th>Update Frequency</th>
<th>Description/Rationale</th>
<th>Source(s)</th>
</tr>

</thead>
<tbody>
<tr>
<td><strong>Age Distribution</strong> <br> (% age 65 and over)</td>
<td rowspan="2">Static</td>
<td rowspan="2">Once</td>
<td><em>Aged 65 or Older from 2016-2020 American Community Survey (ACS)</em>. Older ages have been associated with more severe outcomes from COVID-19 infection.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td><strong>Health Disparities</strong> <br>(Uninsured)</td>
<td><em>Percentage uninsured in the total civilian noninstitutionalized population estimate, 2016- 2020 ACS</em>. Individuals without insurance are more likely to be undercounted in infection statistics, and may have more severe outcomes due to lack of treatment.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td><strong>Transmissible Cases</strong></td>
<td rowspan="4">Observed</td>
<td rowspan="7">Daily</td>
<td><em>Cases from the last 14 days per 100k population</em>. Because of the 14-day incubation period, the cases identified in that time period are the most likely to be transmissible. This metric is the number of such "contagious" individuals relative to the population, so a greater number indicates more likely continued spread of disease.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a> , <a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a> (for population estimate)</span></span></td>
</tr>

<tr>
<td><strong>Disease Spread</strong></td>
<td><em>Cases that are from the last 14 days (one incubation period) divided by cases from the last 28 days </em>. Because COVID-19 is thought to have an incubation period of about 14 days, only a sustained decline in new infections over 2 weeks is sufficient to signal reduction in disease spread. This metric is always between 0 and 1, with values near 1 during exponential growth phase, and declining linearly to zero over 14 days if there are no new infections.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a></span></span></td>
</tr>

<tr>
<td><strong>Social Distancing</strong></td>
<td><em>Unacast social distancing scoreboard grade is assigned by looking at the change in overall distance travelled and the change in nonessential visits relative to baseline (previous year), based on cell phone mobility data</em>. The grade is converted to a numerical score, with higher values being less social distancing (worse score) is expected to increase the spread of infection because more people are interacting with other.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">Unacast</a></span></td>
</tr>

<tr>
<td><strong>Vaccination Full Dose</strong><br>(Series_Complete_Pop_Pct)</td>
<td> Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.</td>
<td><span><a href="https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh" target="_blank">CDC</a></span></td>
</tr>

<tr>
<td><strong>SinWeekly</strong></td>
<td rowspan="2">Known Future</td>
<td> <em>Sin (day of the week / 7) </em>.</td>
<td rowspan="2">Date</td>
</tr>

<tr>
<td><strong>CosWeekly</strong></td>
<td> <em>Cos (day of the week / 7) </em>.</td>
</tr>

<tr>
<td><strong>Case</strong></td>
<td>Target</td>
<td> COVID-19 infection at county level.</td>
<td><span><a href="https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/" target="_blank">USA Facts</a></span></td>
</tr>
</tbody>
</table>

</div>

## Usage guideline

* Please do not add temporarily generated files in this repository.
* Make sure to clean your tmp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git.
  * To check which files git says untracked: `git status -u`. 
  * If you have folders you want to exclude, add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.
