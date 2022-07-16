import argparse
import sys, json
sys.path.append( '..' )
from Class.DataMerger import *


"""
Example usage 

python prepare_data.py -c '../config_2021_Nov.json' -d '../../dataset_raw/CovidDecember12-2021' -o '../2021_Nov/'
python prepare_data.py -c '../config_2022_May.json' -d '../../dataset_raw/CovidMay17-2022' -o '../2022_May/'
"""

parser = argparse.ArgumentParser(description='Create merged feature file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-c', '--configPath',help='Path to the json config file', 
    type=str, default='../config_2022_May.json'
)
parser.add_argument(
    '-d', '--dataPath', help='Directory where raw input feature files are located', 
    type=str, default='../../dataset_raw/CovidMay17-2022'
)
parser.add_argument(
    '-o', '--outputPath', help='Directory where outputs will be saved. This path will be created if it does not exist.',
    type=str, default='../2022_May/'
)

parser.add_argument(
    '-ch', '--cachePath', help='Cached location of the Tota.csv file. This can be reused to create the population or rurality cut.',
    type=str, default=None # '../2022_May/Total.csv'
)

parser.add_argument(
    '-s', '--supportPath', help='Directory where input support files (e.g. population, rurality) are located',
    type=str, default='../../dataset_raw/Support files'
)

args = parser.parse_args()

# create output path if it doesn't exist
if not os.path.exists(args.outputPath):
    print(f'Creating output directory {args.outputPath}')
    os.makedirs(args.outputPath, exist_ok=True)

# load config file
with open(args.configPath) as inputFile:
    config = json.load(inputFile)
    print(f'Config file loaded from {args.configPath}')
    inputFile.close()

# get merger class
dataMerger = DataMerger(config, args.dataPath, args.supportPath)
# if you have already created the total df one, and now just want to 
# reuse it to create different population or rurality cut
if args.cachePath:
    total_df = pd.read_csv(args.cachePath)
else:
    total_df = dataMerger.get_all_features(
        remove_input_outliers=True, 
        # check whether you want to remove target outliers or not 
        remove_target_outliers=True
    )

output_path_total = os.path.join(args.outputPath, 'Total.csv') 
print(f'Writing total data to {output_path_total}\n')
total_df.round(4).to_csv(output_path_total, index=False)

# you can define Rurality cut in 'data'->'support'
# Rurality cut has to be set true. and also set lower and upper limit in RuralityRange and/or MADRange
# having -1 in either of these two will result in ignoring that key
if dataMerger.need_rurality_cut():
    rurality_df = dataMerger.rurality_cut(total_df)

    output_path_rurality_cut = os.path.join(args.outputPath, 'Rurality_cut.csv')
    print(f'Writing rurality cut data to {output_path_rurality_cut}\n')
    rurality_df.round(4).to_csv(output_path_rurality_cut, index=False)

# you can define 'Population cut' in 'data'->'support'
# this means how many of top counties you want to keep
if dataMerger.need_population_cut():
    top_df = dataMerger.population_cut(total_df)
    filename = f"Top_{dataMerger.support_config['Population cut']}.csv"

    output_path_population_cut = os.path.join(args.outputPath, filename)
    print(f'Writing population cut data to {output_path_population_cut}\n')
    top_df.round(4).to_csv(output_path_population_cut, index=False)