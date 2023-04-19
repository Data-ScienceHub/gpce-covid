# Interpreting County Level COVID-19 Infection and Feature Sensitivity using Deep Learning Time Series Models

## Introduction
This work combines sensitivity analysis with heterogeneous time-series deep learning model prediction, which corresponds to the interpretations of Spatio-temporal features from what the model has actually learned. We forecast county-level COVID-19 infection using the Temporal Fusion Transformer (TFT). We then use the sensitivity analysis extending Morris Method to see how sensitive the outputs are with respect to perturbation to our static and dynamic input features. We have collected more than 2.5 years of socioeconomic and health features over 3142 US counties. Using the proposed framework, we conduct extensive experiments and show our model can learn complex interactions and perform predictions for daily infection at the county level. 

## Folder Structure

* **dataset_raw**: Contains the collected raw dataset and the supporting files. To update use the [Update dynamic dataset](/dataset_raw/Update%20dynamic%20features.ipynb) notebook. Static dataset is already update till the onset of COVID-19 using [Update static dataset](/dataset_raw/Update%20static%20features.ipynb) notebook.
* **TFT-PyTorch**: Contains all codes and merged feature files used during the TFT experimentation setup and interpretation. For more details, check the [README.md](/TFT-PyTorch/README.md) file inside it. The primary results are highlighted in [results.md](/TFT-PyTorch/results.md). 


## How to Reproduce

### Virtual Environment

To create the virtual environment
* By pip, use the [requirement.txt](/requirements.txt).
* By anaconda, use the [environment.yml](/environment.yml).

You can directly create a python virtual environment using the [environment.yml](environment.yml) file and Anaconda. Copy this file to your home directory and run the following command,

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

A training notebook run on colab is shared [here](https://colab.research.google.com/drive/1yhI1PesOXYlB6iYXHre9zXMks1a4P6U2?usp=sharing). Feel free to copy and run on your colab and let me know if there are any issues.

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Uncomment the installation commands in the code or set `running_on_colab` to `True` in the code. Upload the TFT-pytorch folder in your drive and set that path in the notebook colab section.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the path accordingly in the notebook.

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
<td> Sin (day of the week / 7).</td>
<td rowspan="2">Date</td>
</tr>

<tr>
<td><strong>CosWeekly</strong></td>
<td> Cos (day of the week / 7).</td>
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
  * To check which files git says untracked `git status -u`. 
  * If you have folders you want to exclude add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.
