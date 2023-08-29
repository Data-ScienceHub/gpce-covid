# Preparing the Environment
For the environment you can either prepare it from scratch or use the containers (both `docker` and `singularity`) we have already created.

## 1. Virtual Environment for Python
First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`.

## 1.1 Option A: Anaconda
If you have `Anaconda` installed locally, follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). An example code,

```
conda create -n ml python=3.10 pip
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
```

On windows :
```bash
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv ml
.\env\Scripts\activate
```

Follow the instructions [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to make sure you have the `pip` and `virtualenv` installed and working. Then create a virtual environement (e.g. name ml) or install required libraries in the default env. 

## 1.2 Install Required Libraries
Once you have the virtual environment created and running, you can download the libraries using, the [requirement.txt](/requirements.txt) file. If you have trouble installing from the file, try installing each of them manually.

### 1.2.1 PyTorch with CUDA
The default versions installed with `pytorch-forecasting` might not work and print cpu instead for the following code. Since it doesn't install CUDA with pytorch.

1. Check if PyTorch is properly installed with CUDA or not. Running the following from the command like should show True.

  ```bash
  python -c "import torch;print(torch.cuda.is_available())"
  ```
  You can also check using the following python code,
  
  ```python
  import torch
  torch.cuda.is_available()
  ```
2. If the previous step returns False, install the following version. Anything [newer](https://pytorch.org/get-started/locally) doesn't work with the `pytorch_forecasting` framework for now.

```bash
# source https://pytorch.org/get-started/previous-versions/#v1131
# option 1. using anaconda
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# option 2. using pip
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Recheck whether PyTorch and CUDA are properly installed using the step 1.

### 1.2.2 CuDNN

This library is required for tensorflow to run on GPU. It can be manually installed following the instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). But not using `pip`. A more convenient way is to use anaconda:

```
conda install -c conda-forge cudnn=8.1.0
```

### 1.2.3 Tensorflow 

The detailed instructions for all OS and versions are [here](https://www.tensorflow.org/install/pip).For me a windows user, `Tensorflow` 2.10.0 is the last version installable by `windows native`. Later versions have to be installed by `wsl`. On `linux`, there is no major change. The following is for windows.

```bash
pip install tensorflow==2.10.*
```

If you are running on `CPU`, move onto the next section. If using `GPU`, check if the `CuDNN` is properly installed. Tensorflow GPU is used in the related work benchmarking with LSTM and BiLSTM only. It isn't required for the TFT. You can check using python,

```{code-block} python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

Or using command line,
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If this returns an empty list or 0, you need to install it. Installing `cudnn` doesn't work with pip, but `anaconda` works. It can also be manually installed following the instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

```
conda install -c conda-forge cudnn=8.1.0
set CUDA_VISIBLE_DEVICES=1
```

Recheck again if `tf.config.list_physical_devices('GPU')` returns a list of GPU devices.

### 1.2.4 Other libraries

Install the rest of the libraries using `pip` and [requirements.txt](/requirements.txt) file. The requirement file has been updated to support the latest versions. The versions used during the study in the paper is in [requirements_old.txt](/requirements_old.txt) file.

```{warning}
Using `pip` inside conda virtual environment actually installs the libraries on the global pip. Check this blog for more info https://www.anaconda.com/blog/using-pip-in-a-conda-environment. This can be unintended. But conda install fails to resolve the dependency conflicts. Hence couldn't be used. 
```

To install the other libraries from `pip` run the following commands. You can test whether the environment has been installed properly using a small dataset in the [`train.py`](/TFT-pytorch/script/train.py) file. The default configuration runs the training on top 100 US counties by population.

```bash
# On linux/macOS
python3 -m pip install -r requirements.txt

# On windows
py -m pip install -r requirements.txt
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

## 2.3 Google Colab

```{note}
Recent trials (June, 23) with colab has shown it keeps failing with the version changes. And doesn't install the pytorch_forecasting properly.
```

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Add the following installation commands in the code. Upload the `TFT-pytorch` folder in your drive and set that filepaths accordingly.

```python
!pip install pytorch_lightning==1.8.6
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

If the environment is ready, you can use the [Related Works](/Related%20Works/) folder to reproduce the baseline models we compared our TFT-model performance with. And the [TFT-pytorch](/TFT-pytorch/) folder to reproduce the TFT-model training and interpretation.

```{note}
In case of scripts, run them from the same directory they are in. This is due to using relative imports for the other modules in the project.
```

Running from the same directory helps python finding the modules paths in parent folders. Notebooks are run from the same folder anyway, so this won't be an issue for them.

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