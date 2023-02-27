# %% [markdown]
# # Introduction
# This file converts the cleaned raw dataset into a single merged file that the TFTModel can work on. The script version available at [prepare_data.py](../script/prepare_data.py).
# 
# If you need to change the input feature set, only add that info in the `"data"` section of the json configuration  file. This notebook will update the rest (at least feature column mappings and locations) . If you have pivoted dynamic feature and need to melt that date columns, make sure to keep the feature name as `string` in `"dynamic_features_map"`. If it is already melted and your dynamic file has a `Date` column, `list` or `string` format both is fine.
# 
# In the final output all null values are replaced with 0. If you don't want that, comment that out.

# %% [markdown]
# # Import libraries

# %%
import sys
sys.path.append( '..' )

# %% [markdown]
# # Setup storage
# 
# You would need the `CovidOct-2022` and `Support files` folders for the dateset. And the `TFT-pytorch` folder for the codes. Upload both of them in the place where you are running the code from. My folder structure looks like this
# * dataset_raw
#     * CovidMay17-2022
#     * Support files
# * TFT-pytorch

# %% [markdown]
# ## Googe drive
# Not needed, since you can run this on CPU. But set `running_on_colab = True` if using. Also update the `cd` path so that it points to the notebook folder in your drive.

# %%
running_on_colab = False

# if running_on_colab:
#     from google.colab import drive
#     drive.mount('/content/drive')

#     %cd /content/drive/My Drive/Projects/Covid/TFT-pytorch/notebooks

# %% [markdown]
# ## Input
# If running on colab, modify the below paths accordingly. Note that this config.json is different from the config.json in TF2 folder as that is for the old dataset.

# %%
from dataclasses import dataclass
from Class.DataMerger import *

@dataclass
class args:
    # folder where the cleaned feature file are at
    dataPath = '../../dataset_raw/CovidMay17-2022'
    supportPath = '../../dataset_raw/Support files'
    configPath = '../configurations/age_groups.json'
    cachePath = None # '../2022_May_cleaned/Total.csv'

    # choose this carefully
    outputPath = '../2022_May_age_groups/'

# %%
# create output path if it doesn't exist
if not os.path.exists(args.outputPath):
    print(f'Creating output directory {args.outputPath}')
    os.makedirs(args.outputPath, exist_ok=True)

import json

# load config file
with open(args.configPath) as inputFile:
    config = json.load(inputFile)
    print(f'Config file loaded from {args.configPath}')
    inputFile.close()

# %% [markdown]
# # Data merger

# %% [markdown]
# ## Total features

# %%
# get merger class
dataMerger = DataMerger(config, args.dataPath, args.supportPath)

# %%
# if you have already created the total df one, and now just want to 
# reuse it to create different population cut
if args.cachePath:
    total_df = pd.read_csv(args.cachePath)
else:
    total_df = dataMerger.get_all_features()
    
    output_path_total = os.path.join(args.outputPath, 'Total.csv') 
    print(f'Writing total data to {output_path_total}\n')
    
    # rounding up to reduce the file size
    total_df.round(4).to_csv(output_path_total, index=False)

# %% [markdown]
# ## Population cut

# %%
# you can define 'Population cut' in 'data'->'support'
# this means how many of top counties you want to keep

if dataMerger.need_population_cut():
    population_cuts = dataMerger.population_cut(total_df)
    for index, population_cut in enumerate(population_cuts):
        top_counties = dataMerger.data_config.population_cut[index]
        filename = f"Top_{top_counties}.csv"

        output_path_population_cut = os.path.join(args.outputPath, filename)

        print(f'Writing top {top_counties} populated counties data to {output_path_population_cut}.')
        population_cuts[index].round(4).to_csv(output_path_population_cut, index=False)