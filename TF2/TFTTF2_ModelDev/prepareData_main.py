from data_preparation import DataPrep
import json
import argparse

def main():

    parser = argparse.ArgumentParser(description='Create TFT data file', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', '--configPath',
                        help='Path the the .config file to read in paremeters for TFT', type=str, default=None)
    parser.add_argument('-f', '--dataPath',
                        help='Path to input files for the aggregation', type=str, default=None)
    parser.add_argument('-o', '--outputPath',
                        help='Save file Path')

    args = parser.parse_args()
    if args.configPath is None:
        raise ValueError('You did not pass a path to a configuration file.')
    if args.dataPath is None:
        raise ValueError('You did not pass an input path argument')
    if args.outputPath is None:
        raise ValueError('You did not pass an output path argument.')

    f = open(args.configPath)
    config = json.load(f)

    dp = DataPrep(config['TFTparams']['data'], args.dataPath)

    TOTALDF = dp.scale_data()

    print('Final Columns in prepared data')
    print(TOTALDF.columns)

    TOTALDF.to_csv(args.outputPath)

if __name__ =='__main__':
    main()