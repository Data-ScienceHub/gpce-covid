# Interpreting County Level COVID-19 Infection and Feature Sensitivity using Deep Learning Time Series Models

## Introduction
This work combines sensitivity analysis with heterogeneous time-series deep learning model prediction, which corresponds to the interpretations of Spatio-temporal features from what the model has actually learned. We forecast county-level COVID-19 infection using the Temporal Fusion Transformer (TFT). We then use the sensitivity analysis extending Morris Method to see how sensitive the outputs are with respect to perturbation to our static and dynamic input features. We have collected more than 2.5 years of socioeconomic and health features over 3142 US counties. Using the proposed framework, we conduct extensive experiments and show our model can learn complex interactions and perform predictions for daily infection at the county level. 

## Folder Structure
* **dataset_raw**: Contains the collected raw dataset and the supporting files. To update use the [Update dynamic dataset](/dataset_raw/Update%20dynamic%20features.ipynb) notebook. Static dataset is already update till the onset of COVID-19 using [Update static dataset](/dataset_raw/Update%20static%20features.ipynb) notebook.
* **TFT-PyTorch**: Contains all codes and merged feature files used during the TFT experimentation setup and interpretation. For more details, check the [README.md](/TFT-PyTorch/README.md) file inside it. The primary results are highlighted in [results.md](/TFT-PyTorch/results.md). 


## How to Reproduce

### Singularity
You can either pull the singularity container from the remote library,
```bash
singularity pull tft_pytorch.sif library://khairulislam/collection/tft_pytorch:latest
```
Or create the container locally using the [singularity.def](/TFT-pytorch/singularity.def) file. Executeg the following command. This uses the definition file to create the container from scratch. Note that is uses `sudo` and requires root privilege. After compilation, you'll get a container named `tft_pytorch.sif`. 

```bash
sudo singularity build singultft_pytorchatft_pytorchrity.sif singularity.def
```

## Features

Note that, past values target and known futures are also used as observed inputs by TFT.

<div align="center">

| Feature        | Type       |
|:------------------------:|:------------:|
| Age Distribution       | Static     |
| Health Disparities     | Static     |
| Disease Spread         | Dynamic    |
| Social Distancing      | Dynamic    |
| Transmissible Cases    | Dynamic    |
| Vaccination Full Dose   | Dynamic    |
| SinWeekly | Known Future |
| CosWeekly | Known Future | 

</div>

<h2 class="accordion-toggle accordion-toggle-icon">Details of Features in Oct 25 data update</h4>
<div class="accordion-content">
<table class="pop_up_table">
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
<td><em>Aged 65 or Older from 2016-2020 American Community Survey (ACS)</em>. Older ages have been associated with more severe outcomes from COVID-19 infection.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td colspan="5"><strong>Health Disparities</strong></td>
</tr>
<tr>
<td>Uninsured</td>
<td>Static</td>
<td style="background: #C885EC;"></td>
<td><em>Percentage uninsured in the total civilian noninstitutionalized population estimate, 2016- 2020 ACS</em>. Individuals without insurance are more likely to be undercounted in infection statistics, and may have more severe outcomes due to lack of treatment.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td colspan="5"><strong>Transmissible Cases</strong></td>
</tr>
<tr>
<td></td>
<td>Daily</td>
<td style="background: #CC3333;"></td>
<td><em>Cases from the last 14 days per 100k population</em>. Because of the 14-day incubation period, the cases identified in that time period are the most likely to be transmissible. This metric is the number of such &ldquo;contagious&rdquo; individuals relative to the population, so a greater number indicates more likely continued spread of disease.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a> , <a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a> (for population estimate)</span></span></td>
</tr>
<tr>
<td colspan="5"><strong>Disease Spread</strong></td>
</tr>
<tr>
<td></td>
<td>Daily</td>
<td style="background: #E64D4D;"></td>
<td><em>Cases that are from the last 14 days (one incubation period) divided by cases from the last 28 days </em>. Because COVID-19 is thought to have an incubation period of about 14 days, only a sustained decline in new infections over 2 weeks is sufficient to signal reduction in disease spread. This metric is always between 0 and 1, with values near 1 during exponential growth phase, and declining linearly to zero over 14 days if there are no new infections.</td>
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