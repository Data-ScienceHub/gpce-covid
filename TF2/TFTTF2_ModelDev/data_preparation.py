import pandas as pd
import numpy as np
import datetime
import os
from functools import reduce
import math


class CustomScaler:
    def __init__(self, target=False, **kwargs):

        self.target = target

    def fit(self, X):

        X = X.copy()
        X = X.astype(np.float32)
        if self.target:
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
        if self.target:
            X = np.sqrt(X, dtype=np.float32)
            X[np.isnan(X)] = 0
        return np.multiply(X - self.min, self.rec)

    def inverse_transform(self, X):

        X = X / self.rec + self.min
        if self.target:
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

        self.first_date = pd.to_datetime(config['support']['FirstDate'])
        self.last_date = pd.to_datetime(config['support']['LastDate'])

        self.inputs = config['inputs_use']
        self.input_files = [config['input_files'][file] for file in config['input_files'] if file in self.inputs]

        self.target = [file for file in config['targets']][0]
        self.target_file = [config['targets'][file] for file in config['targets']][0]

        #self.support = [file for file in config['support'] if isinstance(config['support'][file],str)]
        #self.support_files = [config['support'][file] for file in config['support'] if isinstance(config['support'][file],str)]

        self.feature_scaler = CustomScaler()
        self.target_scaler = CustomScaler(target=True)

        self.MADRANGE = config['support']['MADRange']
        self.RURRANGE = config['support']['RuralityRange']


    def prepare_target(self):

        base_feature = self.target
        base_data = pd.read_csv(os.path.join(self.DATADIR, self.target_file))

        locs = base_data['FIPS'].values
        self.locs = locs
        print(str(len(locs)) + ' number of counties in the target file.')
        nfips = self.config['support']['NFIPS']
        if len(locs) != nfips:
            raise ValueError('Number of locations in target file does not match number of locations parameter:' + str(
                locs) + ' in file vs. ' + str(nfips))
        
        # drop unnamed index column
        base_data = base_data.loc[:, ~base_data.columns.str.contains('^Unnamed')]
        
        base_data = base_data.T
        head = base_data.iloc[0]
        base_data = base_data.iloc[1:]
        base_data.columns = head
        base_data = base_data.diff()
        base_data = base_data.fillna(0)
        base_data = base_data.T.reset_index()
        base_data = base_data.melt('FIPS', var_name='Date', value_name=self.target)
        base_data['Date'] = pd.to_datetime(base_data['Date'])

        base_data.loc[base_data[self.target]<0, self.target] = 0

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

            # some csv files had index columns when they were dumped, now the appear as Unnamed when read
            feature_df = feature_df.loc[:, ~feature_df.columns.str.contains('^Unnamed')]

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

        rur = pd.read_csv(os.path.join(self.DATADIR, self.config['support']['Rurality']), encoding = 'latin1')

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



    def add_embeddings(self, data):

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

        # Set up linear location encoding for all of the data
        LLE = LinearLocationEncoding(self.config['support']['NFIPS'])

        for idx, i in enumerate(data['Name'].unique()):
            data.loc[data['Name'] == i, 'LinearSpace'] = LLE[idx]

        # Set up constant encoding
        data['Constant'] = 0.5

        # Set up linear time encoding
        dates = []
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        for i in data['Date'].unique():
            i = datetime.datetime.strptime(i, '%Y-%m-%d')
            dates.append(i)

        LTE = LinearTimeEncoding(dates)
        P2E = P2TimeEncoding(len(dates))
        P3E = P3TimeEncoding(len(dates))
        P4E = P4TimeEncoding(len(dates))

        CosWeeklyTE, SinWeeklyTE = WeeklyTimeEncoding(dates)

        for idx, i in enumerate(data['Date'].unique()):
            data.loc[data['Date'] == i, 'LinearTime'] = LTE[idx]
            data.loc[data['Date'] == i, 'P2Time'] = P2E[idx]
            data.loc[data['Date'] == i, 'P3Time'] = P3E[idx]
            data.loc[data['Date'] == i, 'P4Time'] = P4E[idx]
            data.loc[data['Date'] == i, 'CosWeekly'] = CosWeeklyTE[idx]
            data.loc[data['Date'] == i, 'SinWeekly'] = SinWeeklyTE[idx]

        return data

    def scale_data(self):

        targ, feat = self.prepare_data()

        self.target_scaler = self.target_scaler.fit(targ[self.target].values)
        targ[self.target] = self.target_scaler.transform(targ.loc[:,self.target])

        self.feature_scaler = self.feature_scaler.fit(feat[self.inputs])
        feat[self.inputs] = self.feature_scaler.transform(feat[self.inputs])

        TotDF = pd.merge(targ, feat, how='left', on=['FIPS', 'Date'])

        TotDF['TimeFromStart'] = (TotDF['Date'] - self.first_date).dt.days

        TotDF = self.add_embeddings(TotDF)

        # replace any remaining null values with 0
        TotDF = TotDF.fillna(0)

        return TotDF





















