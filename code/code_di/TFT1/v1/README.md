# TFT1 v1
## Data
The data is located in /CovidDecember12-2021. Details about the data can be found here: https://github.com/Data-ScienceHub/gpce-covid

## Temporal Fusion Transformer v1
The TFT1 v1 notebook is TFT1_v1.ipynb

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

1. Organize the files with the structure shown above. The COLABROOTDIR is the root dir of the notebook. The GPCE folder contains the code of TFT model and can be downloaded here:
2. In the 'Setup File Systems' code block, change COLABROOTDIR to your root dir of the notebook.
```
COLABROOTDIR="/content/drive/MyDrive/UVA_Research/COVID_Research
```
3. In the 'Start Training' code block, change the directory here to your directory of GPCE/TFToriginal/.
```
%cd "/content/drive/MyDrive/UVA_Research/COVID_Research/GPCE/TFToriginal/"
```
