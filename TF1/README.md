# TFT1 v1 Beta Release
## Data
There are two datasets in provided. **CovidDecemeber12-2021** and **CovidMay17-2022**. Features used in the datasets are listed below.

## Features
| Feature        | Type       |  Currently in Use?      |
|------------------------|------------|-------------------------|
| <mark>Age Distribution</mark>      | Static     | Yes    |
| Air Pollution    | Static     | No    |
| Comorbidities          | Static     | No    |
| Demographics    | Static     | No    |
| <mark>Health Disparities</mark>    | Static     | Yes    |
| Hospital Beds   | Static     | No    |
| Mobility   | Static     | No    |
| Residential Density    | Static     | No                     |
| Voting    | Static     | No    |
| <mark>Disease Spread</mark>       | Dynamic    | Yes     |
| <mark>Social Distancing</mark>    | Dynamic    | Yes    |
| Testing    | Dynamic    | No           |
| <mark>Transmission</mark>    | Dynamic    | Yes    |
| Vaccination >=1 Dose   | Dynamic    | No |
| <mark>Vaccination Full</mark>  | Dynamic    | Yes |

## Temporal Fusion Transformer v1
**TFT1_v1_Train.ipynb:** minimal TFT1 notebook which only supports model training and inferencing \
**TFT1_v1_Morris_PCA.ipynb:** TFT1 + Sensitivity Analysis (Morris and PCA)

### Notebook Setup
```
COLABROOTDIR
│   TFT1_v1.ipynb   
│
└───COVIDJuly2020
│   │
|   └───checkpoints
│   └───CovidDecember12-2021 (data folder)
│       │   Age Distribution.csv
│       │   Air Pollution.csv
│       │   ...
│   
└───GPCE
    │   
    └───TFToriginal
        │   ...
```

The file structure is shown above. The notebook can be setup with the following steps:

1. Organize the files with the structure shown above. The COLABROOTDIR is the root dir of the notebook. The GPCE folder contains the code of TFT model and is the place where TFT1 saves checkpoints and tmp data.
2. In the 'Setup File Systems' code block, change COLABROOTDIR to your root dir of the notebook.
```
COLABROOTDIR="/content/drive/MyDrive/UVA_Research/COVID_Research
```
3. In the 'Start Training' code block, change the directory here to your directory of GPCE/TFToriginal/.
```
%cd "/content/drive/MyDrive/UVA_Research/COVID_Research/GPCE/TFToriginal/"
```
4. Set up 'RunName' and the dataset to use in the 'Set up RunName and Dataset Directory' code block. 'DATASET_NAME' can be set to 'CovidDecember12-2021' or 'CovidMay17-2022'.
```
RunName = 'CovidA21-TFT2Extended-Di-NewData-Unpreprocessed-500TopCounties'
DATASET_NAME = 'CovidMay17-2022'
```
5. Feature selection settings are in 'Science Data Arrays' code block. To use a feature, set the boolean value to True.
```
AgeDist = (True, ['Age Distribution.csv'])
```
6. To train the model, set TFTMode to 1. To do inferencing, set TFTMode to 0.
```
TFTMode = 1
```
