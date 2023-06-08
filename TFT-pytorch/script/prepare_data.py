# %% [markdown]
# # Introduction
# This file converts the cleaned raw dataset into a single merged file that the TFT model can work on. 
# %% [markdown]
# # Import libraries

# %%
import sys
sys.path.append( '..' )

# %% [markdown]
# ## Input
# If running on colab, modify the below paths accordingly. Note that this config.json is different from the config.json in TF2 folder as that is for the old dataset.

# %%
from Class.DataMerger import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(
    description='Prepare Dataset',
    formatter_class=ArgumentDefaultsHelpFormatter
)

parser.add_argument(
   '--config', default='../configurations/baseline.json',
   help='configuration file path'
)

parser.add_argument(
   '--input', help='input folder of raw feature files',
   default='../../dataset_raw/CovidMay17-2022'
)

parser.add_argument(
   '--output', default='../2022_May_cleaned',
   help='output folder for the merged feature file'
)
parser.add_argument(
   '--replace', help='whether to replace the existing features files',
   action='store_true'
)
parser.add_argument(
   '--support', help='folder of support files (e.g. Population.csv)',
   default='../../dataset_raw/Support files'
)

args = parser.parse_args()

# %%
# create output path if it doesn't exist
if not os.path.exists(args.output):
    print(f'Creating output directory {args.output}')
    os.makedirs(args.output, exist_ok=True)

import json

# load config file
with open(args.config) as inputFile:
    config = json.load(inputFile)
    print(f'Config file loaded from {args.config}')
    inputFile.close()

# %% [markdown]
# # Data merger

# %% [markdown]
# ## Total features

# %%
# get merger class
dataMerger = DataMerger(config, args.input, args.support)

# %%
# if you have already created the total df one, and now just want to 
# reuse it to create different population cut
output_path_total = os.path.join(args.output, 'Total.csv') 

# whether to use the cached file
if (not args.replace) and os.path.exists(output_path_total):
    total_df = pd.read_csv(output_path_total)
    print(f'Total.csv already exists in path {output_path_total}. Skipping...')
else:
    total_df = dataMerger.get_all_features()
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

        output_path_population_cut = os.path.join(args.output, filename)

        if (not args.replace) and os.path.exists(output_path_population_cut):
            print(f'{filename} already exists at {output_path_population_cut}. Skipped.')
            continue

        print(f'Writing top {top_counties} populated counties data to {output_path_population_cut}.')
        population_cuts[index].round(4).to_csv(output_path_population_cut, index=False)