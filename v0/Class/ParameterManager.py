import pandas as pd

class ParameterManager:
    def __init__(self, config):
        self.config = config
        self.tft_params = config['TFTparams']
        self.attn_params = self.tft_params['attn']
        self.optimizer_params = self.tft_params['optimizer']
        self.col_mappings = config['col_mappings']

        self.data_config = config['data']
        self.data_params = self.data_config['params']
        self.support_config = self.data_config['support']

        self.target_column = list(self.data_config['targets'].values())[0]
        self.first_date = pd.to_datetime(self.data_config['support']['FirstDate'])
        self.last_date = pd.to_datetime(self.data_config['support']['LastDate'])

        self.epochs = self.tft_params['epochs']
        self.batch_size = self.data_params['batch_size']

        split = self.data_config['split']
        self.train_start = pd.to_datetime(split['train_start'])
        self.validation_start = pd.to_datetime(split['validation_start'])
        self.test_start = pd.to_datetime(split['test_start'])
        self.test_end = pd.to_datetime(split['test_end'])

        self.target_sequence_length = self.tft_params['target_sequence_length']
        self.input_sequence_length = self.tft_params['input_sequence_length']
        self.total_sequence_length = self.target_sequence_length + self.input_sequence_length
        self.early_stopping_patience = self.tft_params['early_stopping_patience']

    @property
    def static_features(self) -> list:
        """Generates the list of static features

        Returns:
            list: feature names
        """

        static_features_map = self.data_config['static_features_map']
        static_feature_list = []
        for value in static_features_map.values():
            if type(value)==list:
                static_feature_list.extend(value)
            else:
                static_feature_list.append(value)

        return static_feature_list

    @property
    def target_features(self) -> list:
        """Generates the list of target features

        Returns:
            list: feature names
        """
        target_features_map = self.data_config['targets']
        target_list = []
        for value in target_features_map.values():
            if type(value) == list:
                target_list.extend(value)
            else:
                target_list.append(value)

        return target_list


    @property
    def dynamic_features(self) -> list:
        """Generates the list of dynamic features

        Returns:
            list: feature names
        """

        dynamic_features_map = self.data_config['dynamic_features_map']
        dynamic_feature_list = []
        for value in dynamic_features_map.values():
            if type(value)==list:
                dynamic_feature_list.extend(value)
            else:
                dynamic_feature_list.append(value)
        # print(f'Dynamic features {dynamic_feature_list}')
        return dynamic_feature_list

    @property
    def unknown_inputs(self):
        """
        number of input futures that are unknown in future (e.g. transmissible cases)
        """
        return self.tft_params['total_inputs'] - len(self.tft_params['static_locs']) - len(self.tft_params['future_locs'])

    def print_params(self):
        print('TFT Regular Parameters\nAll loc parameters below indicate matrix location (column) in dataframe')
        for i in self.tft_params:
            if i != 'attn':
                print(i + ': ' + str(self.tft_params[i]))
    
        print('\nTFT Attention Parameters')
        for i in self.attn_params:
            print(i + ': ' + str(self.attn_params[i]))