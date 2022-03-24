from data_prep import DataPreparation

import argparse
import json
import datetime


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

    data_config = config['TFTparams']['data']
    data_dir = args.dataPath#'/Users/andrejerkelens/Desktop/HiDT/git_repo/HiDimensionalTransformer/COVIDJuly2020/CovidDecember12-2021'
    output_dir = args.outputPath
    dataPrep = DataPreparation(data_config, data_dir)

    # TODO: Add these dates to the config file
    init_date = datetime.datetime(2020, 2, 29)
    fin_dat = datetime.datetime(2021, 6, 30)
    data = dataPrep.read_data(initial_date=init_date, end_date=fin_dat)

    print('DATA TFTdfTOTAL')
    print(data.columns)

    data.to_csv(output_dir + 'TFTdfCurrent.csv')


if __name__ == '__main__':
    main()