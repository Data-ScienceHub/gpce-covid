import numpy as np
import tensorflow as tf

class DataProcessor:
    def __init__(self, total_seq_len, col_mappings, data_params):
        self.total_seq_length = total_seq_len
        self.col_mappings = col_mappings

        self.batch_size = data_params['batch_size']
        self.buffer_size = data_params['buffer_size']

    def batch_data(self, data):
        def _batch_single_entity(input_data, tseq):
            time_steps = len(input_data)
            lags = tseq
            x = input_data.values
            if time_steps >= lags:
                return np.stack([x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)
            else:
                return None

        id_col = self.col_mappings['ID']
        data_map = {}
        for _, sliced in data.groupby(id_col):

            for k in self.col_mappings:
                cols = self.col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy(), self.total_seq_length)

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

    def prepare_batch(self, df, train=False):
        if df.isna().sum().any() :
            raise ValueError('Null values found in your dataset. Try data.fillna(0) before passing the data here.')
            return None, 0

        data = self.batch_data(df)

        all_inputs = np.concatenate(
            (data['Known Regular'], data['Future'], data['TargetAsInput']), axis=2)
        data = tf.data.Dataset.from_tensor_slices((all_inputs, data['Target']))

        if train:
            data = data.cache().shuffle(self.buffer_size).batch(self.batch_size)
        else:
            data = data.cache().batch(self.batch_size)
        return data
