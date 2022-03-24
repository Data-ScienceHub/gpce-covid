import pandas as pd
import numpy as np
import datetime
import os
from functools import reduce


class CustomScaler:
    def __init__(self, targ=False, **kwargs):

        self.targ = targ

    def fit(self, X):

        X = X.copy()
        X = X.astype(np.float32)
        if self.targ:
            X = np.sqrt(X)
        self.max = np.nanmax(X, axis=0)
        self.min = np.nanmin(X, axis=0)
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.rec = np.reciprocal(np.subtract(self.max, self.min), dtype=np.float32)
        self.multi_s1 = np.multiply(self.rec, np.subtract(self.mean, self.min))
        self.multi_s2 = np.multiply(self.rec, self.std)

        return self

    def transform(self, X):

        X = X.astype(np.float32)
        if self.targ:
            X = np.sqrt(X, dtype=np.float32)
        return np.multiply(X - self.min, self.rec)

    def inverse_transform(self, X):

        X = X / self.rec + self.min
        if self.targ:
            return np.square(X)
        else:
            return X

class DataPrep:

    def __init__(self, config, datadir):
        self.config = config
        
        self.data = None
        self.locs = None
        self.fdate = None
        self.cut_locs = None

        self.DATADIR = datadir

        self.first_date = datetime.datetime(config['support']['FirstDate'][0],config['support']['FirstDate'][1],config['support']['FirstDate'][2])
        self.last_date = datetime.datetime(config['support']['LastDate'][0],config['support']['LastDate'][1],config['support']['LastDate'][2])

        self.inputs = config['inputs_use']
        self.input_files = [config['input_files'][file] for file in config['input_files'] if file in self.inputs]

        self.target = [file for file in config['targets']][0]
        self.target_file = [config['targets'][file] for file in config['targets']][0]

        self.support = [file for file in config['support'] if isinstance(config['support'][file],str)]
        self.support_files = [config['support'][file] for file in config['support'] if isinstance(config['support'][file],str)]

        self.feature_scaler = CustomScaler()
        self.target_scaler = CustomScaler(targ=True)

        self.MADRANGE = config['support']['MADRange']
        self.RURRANGE = config['support']['RuralityRange']


    def prepare_target(self):

        base_feature = self.target
        base_data = pd.read_csv(os.path.join(self.DATADIR, self.target_file))

        locs = base_data['FIPS'].values
        self.locs = locs
        print(str(locs) + ' number of counties in the target file.')
        nfips = self.config['support']['NFIPS']
        if len(locs) != nfips:
            raise ValueError('Number of locations in target file does not match number of locations parameter:' + str(
                locs) + ' in file vs. ' + str(nfips))

        base_data = base_data.melt('FIPS', var_name='Date', value_name=self.target)
        base_data['Date'] = pd.to_datetime(base_data['Date'])

        base_data = base_data[(base_data['Date'] >= self.first_date) & (base_data['Date'] <= self.last_date)]

        if (base_data.groupby('FIPS').count()['Date'] == base_data.groupby('FIPS').count()['Date'].iloc[0]).all():
            print('All FIPS code for target ' + str(base_feature) + ' have same length of ' + str(base_data.groupby('FIPS').count()['Date'].iloc[0]))
        else:
            # Add more lines in here to debug potential mismatch
            print('All series do not have the same length')

        return base_data

    def prepare_features(self):

        fdf = []
        for idx, i in enumerate(self.inputs):

            feature_df = pd.read_csv(os.path.join(self.DATADIR, self.input_files[idx]))

            if 'Name' in feature_df.columns:
                feature_df = feature_df.melt(['Name', 'FIPS'], var_name='Date', value_name=i)
            else:
                feature_df = feature_df.melt(['FIPS'], var_name='Date', value_name=i)

            feature_df['Date'] = pd.to_datetime(feature_df['Date'])

            feature_df = feature_df[(feature_df['Date'] >= self.first_date) & (feature_df['Date'] <= self.last_date)]
            if (feature_df.groupby('FIPS').count()['Date'] == feature_df.groupby('FIPS').count()['Date'].iloc[0]).all():
                print('All FIPS code for feature ' + str(i) + ' have the same length of ' + str(feature_df.groupby('FIPS').count()['Date'].iloc[0]))
            else:
                print('All feature series do not have the same length')
            feature_df = feature_df.sort_values(['FIPS','Date'])
            feature_df = feature_df.reset_index(drop=True)

            fdf.append(feature_df)

        return fdf

    def make_cut(self):

        rur = pd.read_csv(os.path.join(self.DATADIR, self.support_files[-1]))

        locs = rur.FIPS

        if -1 in self.RURRANGE:
            print('No Median Rurality Cut')
            lost = []
        else:
            locs = rur[(rur['Median'] >= self.RURRANGE[0]) & (rur['Median'] <= self.RURRANGE[1])].FIPS
            lost = rur[~((rur['Median'] >= self.RURRANGE[0]) & (rur['Median'] <= self.RURRANGE[1]))].FIPS
            rur = rur[rur['FIPS'].isin(locs)]

        print('Lost number of locations from median cut ' + str(len(lost)))
        print('Remaining number of locations from median cut ' + str(len(locs)))

        if -1 in self.MADRANGE:
            print('No MAD cut')
            lost = []
        else:
            locs = rur[(rur['MAD'] >= self.MADRANGE[0]) & (rur['MAD'] < self.MADRANGE[1])].FIPS
            lost = rur[~((rur['MAD'] >= self.MADRANGE[0]) & (rur['MAD'] < self.MADRANGE[1]))].FIPS

        print('Lost Num Locations from MAD Cut ' + str(len(lost)))
        print('Remaining Num Locations from MAD Cut ' + str(len(locs)))

        print('#' * 50)
        print('Final Location Count: ' + str(len(locs)))

        self.locs = locs.values

    def prepare_data(self):

        TFdfBase = self.prepare_target()
        TFdfFeatures = self.prepare_features()
        self.make_cut()

        for idx, i in enumerate(TFdfFeatures):
            if 'Name' in i.columns:
                init_df = TFdfFeatures.pop(idx)
                break

        for i in TFdfFeatures:
            if 'Name' in i.columns:
                i = i.drop(columns=['Name'], axis=1)
            init_df = pd.merge(init_df, i, on=['FIPS', 'Date'], how='left')

        TFdfBase = TFdfBase[TFdfBase['FIPS'].isin(self.locs)]
        init_df = init_df[init_df['FIPS'].isin(self.locs)]

        return TFdfBase, init_df

    def scale_data(self):

        targ, feat = self.prepare_data()

        self.target_scaler = self.target_scaler.fit(targ[self.target].values)
        print(self.target_scaler.mean)
        targ[self.target] = self.target_scaler.transform(targ[self.target])

        self.feature_scaler = self.feature_scaler.fit(feat[self.inputs])
        print(self.feature_scaler.mean)
        feat[self.inputs] = self.feature_scaler.transform(feat[self.inputs])

        TotDF = pd.merge(targ, feat, how='left', on=['FIPS', 'Date'])

        TotDF['TimeFromStart'] = (TotDF['Date'] - self.first_date).dt.days

        return TotDF





















