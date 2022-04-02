import json


class ParamManager:

    def __init__(self, config_path):
        f = open(config_path)
        self.param_dict = json.load(f)

        try:
            self.tft_params = self.param_dict['TFTparams']
        except:
            raise ValueError('There are no TFT params key in the config file')

        if self.tft_params:
            self.attn_params = self.tft_params['attn']
            self.optimizer_params = self.tft_params['optimizer']
            self.col_mappings = self.tft_params['col_mappings']
            self.data_params = self.tft_params['data']['params']
            self.support_params = self.tft_params['data']['support']

    def print_params(self):

        print('TFT Regular Parameters\nAll loc parameters below indicate matrix location (column) in dataframe')
        for i in self.tft_params:
            if i != 'attn':
                print(i + ': ' + str(self.tft_params[i]))
        print('\nTFT Attention Parameters')
        for i in self.attn_params:
            print(i + ': ' + str(self.attn_params[i]))

