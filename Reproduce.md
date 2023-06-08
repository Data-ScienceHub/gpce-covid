# Preparing the Environment
For the environment you can either prepare it from scratch or use the containers (both `docker` and `singularity`) we have already created.

## 1. Virtual Environment for Python
First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`.

## 1.1 Option A: Anaconda
If you have `Anaconda` installed locally, follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). An example code,

```
conda create -n ml python=3.10
conda activate ml
```
This will activate the venv `ml`.


## 1.1 Option B: Python PIP

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

## 1.2 Install Required Libraries
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

## 1.3 Installing CUDA
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

## 2. Containers
If you don't want to setup the environment from scratch and would rather use contaier, use one of the following options.

## 2.1 Singularity
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

## 2.2 Docker

[Dockerfile](/Dockerfile) contains the docker buidling definitions. You can build the container using 
```
docker build -t tft_pytorch
```
This creates a container with name tag tft_pytorch. The run our scripts inside the container.

## Google Colab

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Add the following installation commands in the code. Upload the `TFT-pytorch` folder in your drive and set that filepaths accordingly.

```python
!pip install pytorch_lightning==1.8.1
!pip install pytorch_forecasting==0.10.3
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the paths accordingly in the notebook. 

Load your drive and `cd` to the folder,
```
from google.colab import drive

drive.mount('/content/drive')
 %cd /content/drive/My Drive/gpce-covid
```

# How to Reproduce

Given you have the environment ready, you can use the [Related Works](/Related%20Works/) folder to reproduce the baseline models we compared our TFT-model performance with. Use the [TFT-pytorch](/TFT-pytorch/) folder to reproduce the TFT-model training and interpretation.

**Note: For scripts, run them from the same directory they are in.** This is due to using relative imports for the other modules in the project. Notebooks are run from the same folder anyway, so this won't be an issue for them.

## Data Collection

To update static and dynamic data read the instructions in the following notebooks
1.  [Update static features.ipynb](/dataset_raw/Update%20static%20features.ipynb) 
2. [Update dynamic features.ipynb](/dataset_raw/Update%20dynamic%20features.ipynb)

These will create the single csv files in the [dataset_raw](/dataset_raw/) folder with some initial processing. For example, data updated till May 17, 2022 is in the [CovidMay17-2022](/dataset_raw/CovidMay17-2022/) folder. The [Support files](/dataset_raw/Support%20files/) are static, hence doesn't require frequent updates.

## Configuration

Experiment configurations are decided by the `.json` files in the [configurations](/TFT-pytorch/configurations/) folder.

## Data Processing

To merge the features file from [CovidMay17-2022](/dataset_raw/CovidMay17-2022/) into a single csv file, ready for the deep learning models to run with, use the `prepare_data.py` script.

```
gpce-covid\TFT-pytorch\script> python .\prepare_data.py --help
usage: prepare_data.py [-h] [--config CONFIG] [--input INPUT] [--output OUTPUT] [--replace] [--support SUPPORT]

Prepare Dataset

options:
  -h, --help         show this help message and exit
  --config CONFIG    configuration file path (default: ../configurations/baseline.json)
  --input INPUT      input folder of raw feature files (default: ../../dataset_raw/CovidMay17-2022)
  --output OUTPUT    output folder for the merged feature file (default: ../2022_May_cleaned)
  --replace          whether to replace the existing features files (default: False)
  --support SUPPORT  folder of support files (e.g. Population.csv) (default: ../../dataset_raw/Support files)
```

The default configuration will use the data section in configuration file to create a dataset from start to end time. `You don't need to rerun the preparation for other splits` since they are already within the start and end time. They will be `filtered by train, validation, test period before train or inference.`


## TFT-pytorch

### Runtime Requirement 
You need at least `32 GB RAM memory` to train the full dataset (`Total.csv`, which isn't uploaded in git, but you can create using the previous step). Each epoch will take approx 40-50 minutes. 

But top 100 and 500 counties (`Top_100.csv`, `Top_500.csv` in the `2022_May_cleaned` folder) will run fine within `16 GB memory`. Each epoch takes 2-4 minutes for top 100 and 5-10 minutes for top 500. 

### Training and Interpretation

The training can be done using either `train.py` or ``train_simple.py` script. `train.py` interprets the trained model, when `train_simple.py` only does the training. Move to the scripts folder and run `python train.py` to run the tft training. 

```
gpce-covid\TFT-pytorch\script> python .\train.py --help
Using cuda backend.
usage: train.py [-h] [--config CONFIG] [--input_file INPUT_FILE] [--output OUTPUT] [--show-progress]

Train TFT model

options:
  -h, --help            show this help message and exit
  --config CONFIG       config filename in the configurations folder (default: baseline.json)
  --input_file INPUT_FILE
                        path of the input feature file (default: ../2022_May_cleaned/Top_100.csv)
  --output OUTPUT       output result folder. Anything written in the scratch folder will be ignored by Git. (default: ../scratch/TFT_baseline)
  --show-progress       show the progress bar. (default: False)
```

Similary, the `inference.py` can be used to make model predictions and interpretation on the predicted values. The scripts also have their notebook counterparts in the `notebook` folder.

The `.sh` files are for submitting job requests on remove servers (`Rivanna` or `CS`). If you are submitting on Rivanna, make sure you are added to the accounts. Model `tuning` was done by manually changing the configurations and submitting jobs.

Visualzing the predictions and interpretations are partially done by some of the other notebooks in the notebooks folder.

## Related Works

We compared our works with four other deep learning models named `LSTM`, `BiLSTM`, `NBEATS`, and `NHiTS`. [Related Works](/Related%20Works/) folder has four types of files each of them  

* `Model_train.ipynb` or `.py` for training the model.
* `Model_tuning` to tune the model parameters
* `.db` files to save the tuning results
* `LSTM` and `BiLSTM` both can be trained and tested using the `train.py` and `test.py`. Other two models have their own scripts. 
* The best configurations for the models are saved in `best_config.py`. Unfortunately, the feature columns and some other configurations are still hard coded and have to be changed from the code.
* `splits.py` defines the dataset split (primary, split 1-3).
* Check the `results` and `outputs` folders for details of the results and execution outputs.