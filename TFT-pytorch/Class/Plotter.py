"""
Done following
https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/base_model.html#BaseModel.plot_prediction
"""

import os, sys
import numpy as np
from pandas import DataFrame, to_timedelta
from typing import List, Dict, Union

sys.path.append('..')
from script.utils import calculate_result
from Class.PredictionProcessor import *
from Class.Parameters import Parameters

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
# Apply the default theme
sns.set_theme()
sns.set(font_scale = 2)


plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 12

markers = ['s', 'x',  '.', '+',  'h', 'D', '^', '>', 'p', '<', '*', 'P', 'v']

DPI = 300
ONE_DAY = to_timedelta(1, unit='D')

class PlotResults:
    def __init__(self, figpath:str, targets:List[str], figsize=(18,10), show=True) -> None:
        self.figpath = figpath
        if not os.path.exists(figpath):
            print(f'Creating folder {figpath}')
            os.makedirs(figpath, exist_ok=True)

        self.figsize = figsize
        self.show = show
        self.targets = targets
    
    def plot(self, df:DataFrame, target:str, title:str=None, unit=1, figure_name:str=None, base:int=7):
        fig, ax = plt.subplots(figsize=self.figsize)
        if title is not None: plt.title(title)
        x_column = 'Date'

        plt.plot(df[x_column], df[target], color='blue', label='Observation')
        plt.plot(df[x_column], df[f'Predicted_{target}'], color='green', label='Prediction')

        # TODO: the error line makes the plot look very messy, should remove later
        # plt.plot(df[x_column], abs(df[target] - df[f'Predicted_{target}']), color='red', label='Error')
        plt.xlim(left=df[x_column].min() - ONE_DAY, right=df[x_column].max() + ONE_DAY)
        plt.ylim(bottom=0)

        if base is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=base))

        plt.xticks(rotation = 45)

        if unit>1:
            label_text   = [f'{int(loc/unit)}k' for loc in plt.yticks()[0]]
            ax.set_yticklabels(label_text)
        # plt.xlabel(x_column)
        plt.ylabel(f'Daily {target}')
        plt.legend(loc='upper right')
        fig.tight_layout()

        if figure_name is not None:
            plt.savefig(os.path.join(self.figpath, figure_name), dpi=DPI)
        if self.show:
            plt.show()

    def summed_plot(self, merged_df:DataFrame, type:str='', save:bool=True, base:int=7):
        """
        Plots summation of prediction and observation from all counties

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """

        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = merged_df[target].values, merged_df[predicted_column].values
            
            mae, rmse, msle, smape, nnse = calculate_result(y_true, y_pred)
            title = f'Summed plot: {target} {type} MAE {mae:0.4g}, RMSE {rmse:0.4g}, MSLE {msle:0.4g}, SMAPE {smape:0.4g}, NNSE {nnse:0.4g}'
            unit = 1
            if target=='Cases':
                unit = 1000
            summed_df = PredictionProcessor.makeSummed(merged_df, self.targets)

            target_figure_name = None
            if save: target_figure_name = f'Summed_plot_{target}_{type}.jpg'

            self.plot(summed_df, target, title, unit, target_figure_name, base)

    def individualPlot(self, df:DataFrame, fips:str, type:str='', save:bool=True, base:int=7):
        """
        Plots the prediction and observation for this specific county

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """

        assert fips in df['FIPS'].values, f'Provided FIPS code {fips} does not exist in the dataframe.'
        df = df[df['FIPS']==fips]

        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = df[target].values, df[predicted_column].values
            
            mae, rmse, msle, smape, nnse = calculate_result(y_true, y_pred)
            if target=='Cases': unit = 1000
            else: unit = 1

            target_figure_name = None
            if save: target_figure_name = f'Individual_plot_{target}_{type}_FIPS_{fips}.jpg'
            
            title = f'{target} {type} MAE {mae:0.4g}, RMSE {rmse:0.4g}, MSLE {msle:0.4g}, SMAPE {smape:0.4g}, NNSE {nnse:0.4g}'
            self.plot(df, target, title, unit, target_figure_name, base)

class PlotWeights:
    def __init__(self, figpath:str, parameters:Parameters, show:bool=True):
        self.figpath = figpath
        if not os.path.exists(figpath):
            print(f'Creating folder {figpath}')
            os.makedirs(figpath, exist_ok=True)

        self.static_variables = parameters.data.static_features
        self.encoder_variables = parameters.data.time_varying_unknown_features + parameters.data.time_varying_known_features
        self.decoder_variables = parameters.data.time_varying_known_features
        self.max_encoder_length = parameters.model_parameters.input_sequence_length
        self.show = show
        self.weight_formatter = StrMethodFormatter('{x:,.2f}')

    # variable selection
    def make_selection_plot(
        self, title:str, values, labels, figsize=(10,6)
    ):
        fig, ax = plt.subplots(figsize=figsize)
        order = np.argsort(values)
        values = values / values.sum(-1).unsqueeze(-1)
        ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
        ax.set_title(title)

        ax.set_xlabel("Importance in %")
        plt.tight_layout()

        if self.show:
            plt.show()
        return fig

    def plot_interpretation(self, interpretation) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figures = {}
        figures['attention'] = self.plot_summed_attention(
            interpretation, figsize=(12, 8)
        )
        figures["static_variables"] = self.make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), 
            self.static_variables, (10, 4)
        )

        # Dynamic variables
        figures["encoder_variables"] = self.make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), 
            self.encoder_variables, (10, 8)
        )

        # time unknown variable
        figures["decoder_variables"] = self.make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), 
            self.decoder_variables, (10, 6)
        )

        return figures

    def plot_summed_attention(
        self, interpretation, title:str=None, figure_name:str=None, figsize=(10, 6)
    ):
        fig, ax = plt.subplots(figsize=figsize)
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.max_encoder_length, attention.size(0) - self.max_encoder_length), attention
        )
        plt.ylim(bottom=0)
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention weight")

        if title is not None: ax.set_title(title)
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))

        if figure_name is not None:
            plt.savefig(os.path.join(self.figpath, figure_name), dpi=DPI)

        if self.show: 
            plt.show()
        return fig

    def plot_attention(
        self, attention_mean:DataFrame, title:str='Attention comparison on different time index',
        figsize=(18, 10), step_size:int=1, figure_name:str=None, base:int=7, target_day:int=None, 
        limit:int=3, enable_markers=True
    ):
        """
        Plots attention weights by weekdays.

        Args:
            attention_mean: average attention per date
            title: if none, no title will be set
            figure_name: filename the figure will be saved with. if none it won't be saved. E.g.: "weekly_attention.jpg"
            base: period along the x-axis.
            target_day: Days of the week to annotate in the plot.
            limit: maximum encoder lengths to be plotted. If None, plots all lengths
        """

        max_encoder_length = self.max_encoder_length

        fig, ax = plt.subplots(figsize=figsize)
        if title is not None: plt.title(title)
        x_column = 'Date'

        if limit is None: limit = max_encoder_length
        for encoder_length in range(-1, -1-max_encoder_length, -step_size):
            if limit < 1: break

            if enable_markers:
                plt.plot(
                    attention_mean[x_column], attention_mean.loc[:, max_encoder_length + encoder_length], 
                    label=f'Time index {encoder_length}', 
                    marker=markers[max_encoder_length + encoder_length]
                )
            else:
                plt.plot(
                    attention_mean[x_column], attention_mean.loc[:, max_encoder_length + encoder_length], 
                    label=f'Time index {encoder_length}'
                )
            limit -= 1

        for index, column in enumerate(['mean', 'median']):
            if enable_markers: plt.plot(
                attention_mean[x_column], attention_mean.loc[:, column], 
                label=column.capitalize(), 
                marker=markers[index]
            )
            else: plt.plot(
                attention_mean[x_column], attention_mean.loc[:, column], 
                label=column.capitalize()
            )
            break
 
        _, ymax = plt.gca().get_ylim()

        weeks = attention_mean[x_column].dt.weekday.values

        if target_day is not None:
            plt.vlines(
                    x=attention_mean[weeks==target_day][x_column], ymin=0,
                    ymax=ymax, label=weekdays[target_day], color='black'
                )
        
        plt.xlim(attention_mean[x_column].min() - ONE_DAY, attention_mean[x_column].max() + ONE_DAY)
        plt.ylim(bottom=0)
        if base is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=base))
        
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        plt.xticks(rotation = 45)

        plt.ylabel('Attention weight')
        plt.legend(loc='upper right')

        fig.tight_layout()

        if figure_name is not None: 
            plt.savefig(os.path.join(self.figpath, f'{figure_name}.jpg'), dpi=DPI)
            
        if self.show: plt.show()
        
    def plot_weekly_attention(
        self, attention_weekly:DataFrame, title:str= 'Attention comparison on different days of the week',
        figsize=(18, 10), step_size=1, limit:int=3, figure_name:str=None
    ):
        """
        Plots attention weights by weekdays.

        Args:
            attention_weekly: average attention per weekday
            title: if none, no title will be set
            step_size: the increment in index when plotting attention for different encoder lengths
            figure_name: filename the figure will be saved with. if none it won't be saved. E.g.: "weekly_attention.jpg"
            limit: maximum encoder lengths to be plotted. If None, plots all lengths
        """
        fig, ax = plt.subplots(figsize=figsize)
        if title is not None: plt.title(title)

        if limit is None: limit = self.max_encoder_length
        for encoder_length in range(-1, -1-self.max_encoder_length, -step_size):
            if limit < 1: break

            plt.plot(
                attention_weekly['weekday'], attention_weekly.loc[:, self.max_encoder_length + encoder_length], 
                label=f'Time index {encoder_length}', 
                marker=markers[encoder_length]
            )
            limit -= 1

        for index, column in enumerate(['mean', 'median']):
            plt.plot(
                attention_weekly['weekday'], attention_weekly.loc[:, column], 
                label=column.capitalize(), 
                marker=markers[index]
            )
            break
        
        plt.ylim(bottom=0)

        plt.ylabel('Attention weight')
        # plt.xlabel('Weekday')
        
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        plt.legend(loc='upper right')

        fig.tight_layout()

        if figure_name is not None:
            plt.savefig(os.path.join(self.figpath, f'{figure_name}.jpg'), dpi=DPI)
            
        if self.show: plt.show()