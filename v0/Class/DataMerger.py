import pandas as pd
from pandas import DataFrame
from functools import reduce
import numpy as np
import os, json
import math

from Class.ParameterManager import ParameterManager

import sys
sys.path.append('..')
from script.utils import *

"""

Args:


Returns:
    
"""

class Embedding:
    """
    Utility class to add embedding features based on time and locations
    """
    def LinearLocationEncoding(TotalLoc):
        linear = np.empty(TotalLoc, dtype=float)
        for i in range(0, TotalLoc):
            linear[i] = float(i) / float(TotalLoc)
        return linear

    def LinearTimeEncoding(Dateslisted):
        Firstdate = Dateslisted[0]
        numtofind = len(Dateslisted)
        dayrange = (Dateslisted[numtofind - 1] - Firstdate).days + 1
        linear = np.empty(numtofind, dtype=float)
        for i in range(0, numtofind):
            linear[i] = float((Dateslisted[i] - Firstdate).days) / float(dayrange)
        return linear

    def P2TimeEncoding(numtofind):
        P2 = np.empty(numtofind, dtype=float)
        for i in range(0, numtofind):
            x = -1 + 2.0 * i / (numtofind - 1)
            P2[i] = 0.5 * (3 * x * x - 1)
        return P2

    def P3TimeEncoding(numtofind):
        P3 = np.empty(numtofind, dtype=float)
        for i in range(0, numtofind):
            x = -1 + 2.0 * i / (numtofind - 1)
            P3[i] = 0.5 * (5 * x * x - 3) * x
        return P3

    def P4TimeEncoding(numtofind):
        P4 = np.empty(numtofind, dtype=float)
        for i in range(0, numtofind):
            x = -1 + 2.0 * i / (numtofind - 1)
            P4[i] = 0.125 * (35 * x * x * x * x - 30 * x * x + 3)
        return P4

    def WeeklyTimeEncoding(Dateslisted):
        numtofind = len(Dateslisted)
        costheta = np.empty(numtofind, dtype=float)
        sintheta = np.empty(numtofind, dtype=float)
        for i in range(0, numtofind):
            j = Dateslisted[i].date().weekday()
            theta = float(j) * 2.0 * math.pi / 7.0
            costheta[i] = math.cos(theta)
            sintheta[i] = math.sin(theta)
        return costheta, sintheta
    
    def add_embedding(data):
        data['TimeFromStart'] = (data['Date'] - data['Date'].min()).dt.days

        # Set up linear location encoding for all of the data
        LLE = Embedding.LinearLocationEncoding(data['FIPS'].nunique())

        for idx, i in enumerate(data['FIPS'].unique()):
            data.loc[data['FIPS'] == i, 'LinearSpace'] = LLE[idx]

        # Set up constant encoding
        data['Constant'] = 0.5

        # Set up linear time encoding
        dates = pd.to_datetime(data['Date'].unique())

        LTE = Embedding.LinearTimeEncoding(dates)
        P2E = Embedding.P2TimeEncoding(len(dates))
        P3E = Embedding.P3TimeEncoding(len(dates))
        P4E = Embedding.P4TimeEncoding(len(dates))

        CosWeeklyTE, SinWeeklyTE = Embedding.WeeklyTimeEncoding(dates)

        for idx, i in enumerate(dates):
            data.loc[data['Date'] == i, 'LinearTime'] = LTE[idx]
            data.loc[data['Date'] == i, 'P2Time'] = P2E[idx]
            data.loc[data['Date'] == i, 'P3Time'] = P3E[idx]
            data.loc[data['Date'] == i, 'P4Time'] = P4E[idx]
            data.loc[data['Date'] == i, 'CosWeekly'] = CosWeeklyTE[idx]
            data.loc[data['Date'] == i, 'SinWeekly'] = SinWeeklyTE[idx]

        # return data

class DataMerger:
    def __init__(self, config:dict, dataPath:str, support_path:str):
        self.parameters = ParameterManager(config)

        self.dataPath = dataPath
        self.support_path = support_path

    @property
    def first_date(self):
        return self.parameters.first_date
    
    @property
    def last_date(self):
        return self.parameters.last_date

    @property
    def data_config(self):
        return self.parameters.data_config

    @property
    def support_config(self):
        return self.parameters.support_config

    @property
    def static_feature_list(self) -> list:
        """Generates the list of static features

        Returns:
            list: feature names
        """

        return self.parameters.static_features

    @property
    def dynamic_feature_list(self) -> list:
        """Generates the list of dynamic features

        Returns:
            list: feature names
        """

        return self.parameters.dynamic_features

    @property
    def known_future_feature_list(self) -> list:
        """
        Dummy retrieval of embeddings to get column list.

        """
        df = pd.DataFrame({'Date':[pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-02')], 'FIPS':[1001, 1001]})
        Embedding.add_embedding(df)
        return [col for col in df.columns if col not in ['Date', 'FIPS']+self.parameters.col_mappings['Time']]

    @property
    def target_feature_list(self) -> list:

        return self.parameters.target_features

    def get_static_features(self) -> DataFrame:
        """Loads and merges the static features

        Returns:
            DataFrame: static features
        """

        id_columns = ['FIPS']

        static_df = self.get_population()[id_columns]
        locs = static_df['FIPS'].nunique()
        print(f'Unique counties present {locs}')

        """## Merge"""

        static_features_map = self.data_config['static_features_map']
        for file_name in static_features_map.keys():
            feature_df = read_feature_file(self.dataPath, file_name)
            print(f'Merging feature {file_name} with length {feature_df.shape[0]}')

            has_date_columns = False
            for column in feature_df.columns:
                if valid_date(column):
                    has_date_columns = True
                    break

            # if static feature has date column, convert the first date column into feature of that name
            # this is for PVI data, and in that case static_features_map[file_name] is a single value
            if has_date_columns:
                feature_column = static_features_map[file_name]
                feature_df.rename({column: feature_column}, axis=1, inplace=True)
                feature_df = feature_df[['FIPS', feature_column]]
            else: 
                feature_columns = static_features_map[file_name]
                if type(feature_columns) == list:
                    feature_df = feature_df[['FIPS'] + feature_columns]
                else:
                    feature_df = feature_df[['FIPS', feature_columns]]

            static_df = static_df.merge(feature_df, how='inner', on='FIPS')

        print(f"\nMerged static features have {static_df['FIPS'].nunique()} counties")
        print(static_df.head())
        
        return static_df

    def get_dynamic_features(self) -> DataFrame:
        """Loads and merges the dynamic features

        Returns:
            DataFrame: dynamic features
        """

        dynamic_features_map = self.data_config['dynamic_features_map']

        dynamic_df = None
        merge_keys = ['FIPS', 'Date']

        for file_name in dynamic_features_map.keys():
            print(f'Reading {file_name}')
            df = read_feature_file(self.dataPath, file_name)
            
            # check whether the Date column has been pivoted
            if 'Date' not in df.columns:
                # technically this should be set of common columns
                id_vars = [col for col in df.columns if not valid_date(col)]
                df = df.melt(
                    id_vars= id_vars,
                    var_name='Date', value_name=dynamic_features_map[file_name]
                ).reset_index(drop=True)

            # can be needed as some feature files may have different date format
            df['Date'] = pd.to_datetime(df['Date'])
            print(f'Min date {df["Date"].min()}, max date {df["Date"].max()}')
            df = df[(df['Date'] >= self.first_date) & (df['Date'] <= self.last_date)]

            print(f'Length {df.shape[0]}.')

            if dynamic_df is None: dynamic_df = df
            else:
                # if a single file has multiple features
                if type(dynamic_features_map[file_name]) == list:
                    selected_columns = merge_keys + dynamic_features_map[file_name]
                else:
                    selected_columns = merge_keys + [dynamic_features_map[file_name]]

                # using outer to keep the union of dates 
                # as vaccination dates are not available before late in 2020
                dynamic_df = dynamic_df.merge(df[selected_columns], how='outer',on=merge_keys)

                # however, we don't need to keep mismatch of FIPS
                dynamic_df = dynamic_df[~dynamic_df['FIPS'].isna()]
            print()

        print(f'Total dynamic feature shape {dynamic_df.shape}')
        print(dynamic_df.head())
        
        return dynamic_df

    def get_target_feature(self) -> DataFrame:
        """Loads and converts the target feature

        Returns:
            DataFrame: daily covid cases for each county
        """

        target_file_names = list(self.data_config['targets'].keys())
        target_dfs = []

        for target in target_file_names:
            target_column = self.data_config['targets'][target]
            target_df = read_feature_file(self.dataPath, target)

            if 'Date' not in target_df.columns:
                target_df = convert_cumulative_to_daily(target_df)
                target_df.fillna(0, inplace=True)

            target_df = fix_outliers(target_df, verbose=False)

            """## Melt columns"""

            target_df = target_df.melt(
                id_vars= ['FIPS'],
                var_name='Date', value_name=target_column
            ).reset_index(drop=True)
            target_df = target_df.fillna(0)
            target_df['Date'] = pd.to_datetime(target_df['Date'])

            # some days had old covid cases fixed by adding neg values
            target_df.loc[target_df[target_column]<0, target_column] = 0

            target_df = target_df[(target_df['Date'] >= self.first_date) & (target_df['Date'] <= self.last_date)]
            target_dfs.append(target_df)

        target_df = reduce(lambda x, y: pd.merge(x, y, on=['FIPS','Date']), target_dfs)

        return target_df

    def get_all_features(self) -> DataFrame:
        """Loads and merges all features

        Returns:
            DataFrame: the merged file of all features 
        """

        static_df = self.get_static_features()
        dynamic_df = self.get_dynamic_features()
        target_df = self.get_target_feature()

        # the joint types should be inner for consistency
        print('Merging all features')
        total_df = dynamic_df.merge(target_df, how='outer', on=['FIPS', 'Date'])
        total_df = static_df.merge(total_df, how='inner', on='FIPS')
        total_df = total_df.reset_index(drop=True)

        print(f'Total merged data shape {total_df.shape}')
        print('Missing percentage in total data')
        print(missing_percentage(total_df))
        
        print('Filling null values with 0')
        total_df.fillna(0, inplace=True)

        Embedding.add_embedding(total_df)
        return total_df

    def need_rurality_cut(self) -> bool:
        """Whether the configuration mentions to perform a cut based on rurality
        
        Returns:
            bool: yes or no 
        """
        return self.support_config['Rurality cut']

    def need_population_cut(self) -> bool:
        """Whether the configuration mentions to perform a cut based on population
        
        Returns:
            bool: yes or no 
        """
        return self.support_config['Population cut'] > 0

    def rurality_cut(self, total_df:DataFrame) -> DataFrame:
        """Slices the total features based on rurality cut defined by RuralityRange and MADRange 
        
        Args:
            total_df: total feature file

        Returns:
            DataFrame: sliced feature file
        """

        MADRANGE = self.support_config['MADRange']
        RURRANGE = self.support_config['RuralityRange']

        # fails to read on unicode
        rurality = pd.read_csv(os.path.join(self.support_path, self.support_config["Rurality"]), encoding = 'latin1')

        locs = rurality.FIPS

        if -1 in RURRANGE:
            print('No Median Rurality Cut')
            lost = []
        else:
            locs = rurality[(rurality['Median'] >= RURRANGE[0]) & (rurality['Median'] <= RURRANGE[1])].FIPS
            lost = rurality[~((rurality['Median'] >= RURRANGE[0]) & (rurality['Median'] <= RURRANGE[1]))].FIPS
            rurality = rurality[rurality['FIPS'].isin(locs)]

        print('Lost number of locations from median cut ' + str(len(lost)))
        print('Remaining number of locations from median cut ' + str(len(locs)))

        if -1 in MADRANGE:
            print('No MAD cut')
            lost = []
        else:
            locs = rurality[(rurality['MAD'] >= MADRANGE[0]) & (rurality['MAD'] < MADRANGE[1])].FIPS
            lost = rurality[~((rurality['MAD'] >= MADRANGE[0]) & (rurality['MAD'] < MADRANGE[1]))].FIPS

        print('Lost Num Locations from MAD Cut ' + str(len(lost)))
        print('Remaining Num Locations from MAD Cut ' + str(len(locs)))

        print('#' * 50)
        print('Final Location Count: ' + str(len(locs)))

        # only keep the selected counties
        df = total_df[total_df['FIPS'].isin(locs)].reset_index(drop=True)
        print(f'Rurality cut dataset shape {df.shape}')

        return df

    def get_population(self) -> DataFrame :
        """Loads the population file

        Returns:
            DataFrame: population file
        """
        support_file = self.support_config['Population']
        population = pd.read_csv(os.path.join(self.support_path, f'{support_file}'))
        
        return population

    def population_cut(self, total_df:DataFrame) -> DataFrame:
        """Slices the total feature file based on number of top counties by population, 
        mentioned in `Population cut`

        Args:
            total_df: total feature file

        Returns:
            DataFrame: selected feature files
        """
        # number of top counties (by population) to keep
        top_counties = self.support_config['Population cut']
        print(f'Slicing based on top {top_counties} counties by population')
        
        population = self.get_population()
        sorted_fips = population.sort_values(by=['POPESTIMATE2019'], ascending=False)['FIPS'].values

        df = total_df[total_df['FIPS'].isin(sorted_fips[:top_counties])]

        return df

    def update_config(self, configPath:str) -> None:
        """Updates the config file bases on calculated features

        Args:
            configPath: config file path
        """
        counter = 0
        static_locs = [i for i in range(len(self.static_feature_list))]
        counter += len(static_locs)
        #TODO:  skip explicitly labeling the dynamic feature list --> change this later
        counter += len(self.dynamic_feature_list)
        print(f'static locs: {static_locs}')

        future_locs = [i for i in range(counter, counter + len(self.known_future_feature_list))]
        counter += len(future_locs)
        print(f'future locs: {future_locs}')

        target_locs = [i for i in range(counter, counter + len(self.target_feature_list))]
        counter += len(target_locs)
        print(f'target loc: {target_locs}. total inputs {counter}')

        print(f'col_mappings: Static {self.static_feature_list}')
        print(f'col_mappings: Future {self.known_future_feature_list}')
        print(f'col_mappings: Known Regular  {self.static_feature_list + self.dynamic_feature_list}')

        # read the config file again
        with open(configPath) as inputFile:
            config = json.load(inputFile)
            inputFile.close()

        config["TFTparams"]["static_locs"] = static_locs
        config["TFTparams"]["future_locs"] = future_locs
        config["TFTparams"]["target_loc"] = target_locs
        config["TFTparams"]["total_inputs"] = counter

        config["col_mappings"]["Static"] = self.static_feature_list

        config["col_mappings"]["Target"] = self.target_feature_list
        config["col_mappings"]["Future"] = self.known_future_feature_list
        config["col_mappings"]["Known Regular"] = self.static_feature_list + self.dynamic_feature_list
        # dump the json config
        with open(configPath, 'w') as outputFile:
            json.dump(config, outputFile, indent=4)
            outputFile.close()