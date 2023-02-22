import os
from os.path import join
from dataclasses import dataclass

CONFIG_DIR = '../configurations'
BASELINE_DATA_DIR = '../2022_May_cleaned'
INDUSTRY_DATA_DIR = '../2022_May_industry_groups'
AGE_DATA_DIR = '../2022_May_age_groups'
ROOT_RESULTS_DIR = '../results'

class Arguments:
    def __init__(
        self, result_folder, figure_folder='figures', 
        input_filename='Total.csv', 
        config_filename='baseline.json',
        show_progress_bar = False
    ) -> None:
        self.result_folder = join(ROOT_RESULTS_DIR, result_folder) 
        
        self.input_filename = input_filename
        self.config_filename = config_filename

        # set this to false when submitting batch script, otherwise it prints a lot of lines
        self.show_progress_bar = show_progress_bar

    @property
    def figure_folder(self):
        return join(self.result_folder, 'figures')

    @property
    def input_filePath(self):
        return join(BASELINE_DATA_DIR, self.input_filename)

    @property
    def config_folder(self):
        return join(CONFIG_DIR, self.config_filename)

    @property
    def checkpoint_folder(self):
        return join(self.result_folder, 'checkpoints')

    def __get_best_model(self, prefix='best-epoch='):
        for item in os.listdir(self.checkpoint_folder):
            if item.startswith(prefix):
                return join(self.checkpoint_folder, item)

        raise FileNotFoundError(f"Couldn't find the best model in {self.checkpoint_folder}")

@dataclass
class BaseArguments:
    result_folder = join(ROOT_RESULTS_DIR, 'TFT_baseline') 
    figPath = join(result_folder, 'figures')
    checkpoint_folder = join(result_folder, 'checkpoints')
    input_filePath = join(BASELINE_DATA_DIR, 'Total.csv')

    configPath = join(CONFIG_DIR, 'baseline.json')

    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = False

    def __get_best_model(checkpoint_folder):
        for item in os.listdir(checkpoint_folder):
            if item.startswith('best-epoch='):
                return join(checkpoint_folder, item)

        raise FileNotFoundError(f"Couldn't find the best model in {checkpoint_folder}")

    

    model_path = __get_best_model(checkpoint_folder)

@dataclass
class IndustryBaseArguments(BaseArguments):
    input_filePath = join(INDUSTRY_DATA_DIR, 'Total.csv')
    configPath = join(CONFIG_DIR, 'industry_groups.json')
    
    # additional
    static_index = 0

print(BaseArguments.model_path, IndustryBaseArguments.static_index)