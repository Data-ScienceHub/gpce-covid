import numpy as np
import pandas as pd
import tensorflow as tf


class DataManager:

    def __init__(self, data_path, total_seq_len, col_mappings, data_params):

        self.training = pd.read_csv(data_path)
        
        # Temporary overwrite
        # self.training = self.training.dropna()

        # fill na values 0 instead of dropping them
        self.training = self.training.fillna(0)

        if self.training.isna().sum().any():
            raise ValueError('Null values found in your training dataset')

        self.tseq_len = total_seq_len
        self.col_mappings = col_mappings

        self.batch_size = data_params['batch_size']
        self.buffer_size = data_params['buffer_size']

        self.num_samples = None
        self.inference_data = None
        self.training_data = None
        self.np_inference = None

    def batch_data(self, data):

        if self.training is None:
            return None

        def _batch_single_entity(input_data, tseq):
            time_steps = len(input_data)
            lags = tseq
            x = input_data.values
            if time_steps >= lags:
                return np.stack([x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)
            else:
                return None

        id_col = self.col_mappings['ID']
        time_col = self.col_mappings['Time']
        target_col = self.col_mappings['Target']
        input_cols = self.col_mappings['Known Regular'] + self.col_mappings['Future']

        data_map = {}
        for _, sliced in data.groupby(id_col):

            for k in self.col_mappings:
                cols = self.col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy(), self.tseq_len)

                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        data_map['TargetAsInput'] = data_map['Target']
        data_map['Target'] = data_map['Target'][:, 13:, :]

        active_entries = np.ones_like(data_map['Target'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)

        return data_map

    def createTFData(self):

        batched_data = self.batch_data(self.training)

        # TODO fix this line below
        self.num_samples = batched_data['Future'].shape[0]

        all_inputs = np.concatenate(
            (batched_data['Known Regular'], batched_data['Future'], batched_data['TargetAsInput']), axis=2)
        tf_data = tf.data.Dataset.from_tensor_slices((all_inputs, batched_data['Target']))

        def make_inference_batches(ds):
            return ds.cache().batch(self.batch_size)

        def make_train_batches(ds):
            return ds.cache().shuffle(self.buffer_size).batch(self.batch_size)

        self.inference_data = make_inference_batches(tf_data)
        self.np_inference = batched_data
        self.training_data = make_train_batches(tf_data)
