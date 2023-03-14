"""
Done following
https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/base_model.html#BaseModel.plot_prediction
"""

import os, sys
import numpy as np
from pandas import DataFrame, to_timedelta
from typing import List, Dict
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import matplotlib.pyplot as plt

sys.path.append('..')
from script.utils import calculate_result
from Class.PredictionProcessor import *
from Class.PlotConfig import *

from matplotlib.ticker import StrMethodFormatter, MultipleLocator

class PlotResults:
    def __init__(self, figPath:str, targets:List[str], figsize=FIGSIZE, show=True) -> None:
        self.figPath = figPath
        if not os.path.exists(figPath):
            print(f'Creating folder {figPath}')
            os.makedirs(figPath, exist_ok=True)

        self.figsize = figsize
        self.show = show
        self.targets = targets
    
    def plot(
        self, df:DataFrame, target:str, title:str=None, scale=1, 
        base:int=None, figure_name:str=None, plot_error:bool=False
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        if title is not None: plt.title(title)
        x_column = 'Date'

        plt.plot(df[x_column], df[target], color='blue', label='Ground Truth')
        plt.plot(df[x_column], df[f'Predicted_{target}'], color='green', label='Prediction')

        if plot_error:
            plt.plot(df[x_column], abs(df[target] - df[f'Predicted_{target}']), color='red', label='Error')
        _, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max*1.1)
        
        if base is None:
            x_first_tick = df[x_column].min()
            x_last_tick = df[x_column].max()
            x_major_ticks = DATE_TICKS
            ax.set_xticks(
                [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
            )
        else:
            # plt.xlim(left=df[x_column].min() - ONE_DAY, right=df[x_column].max() + ONE_DAY)
            ax.xaxis.set_major_locator(MultipleLocator(base=base))
        
        # plt.xticks(rotation = 15)
        # plt.xlabel(x_column)

        if scale>1:
            if scale==1e3 or scale==1e6:
                label_text = [] 
                if scale ==1e3: unit = 'K'
                else: unit = 'M'

                for loc in plt.yticks()[0]:
                    if loc == 0:
                        label_text.append('0')
                    else:
                        label_text.append(f'{loc/scale:0.5g}{unit}') 

                ax.set_yticklabels(label_text)
                plt.ylabel(f'Daily {target}')
            else:
                ax.yaxis.set_major_formatter(get_formatter(scale))
                if scale==1e3: unit = 'in thousands'
                elif scale==1e6: unit = 'in millions'
                else: unit = f'x {scale:.0e}'

                plt.ylabel(f'Daily {target} ({unit})')
        else:
            plt.ylabel(f'Daily {target}')
            
        if plot_error:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=3, loc='best')
        else:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=2, loc='best')
            
        # fig.tight_layout() # might change y axis values

        if figure_name is not None:
            plt.savefig(os.path.join(self.figPath, figure_name), dpi=DPI)
        if self.show:
            plt.show()
        return fig

    def summed_plot(
        self, merged_df:DataFrame, type:str='', save:bool=True, 
        base:int=None, plot_error:bool=False
    ):
        """
        Plots summation of prediction and observation from all counties

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        summed_df = PredictionProcessor.makeSummed(merged_df, self.targets)
        figures = []
        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = merged_df[target].values, merged_df[predicted_column].values
            
            mae, rmse, rmsle, smape, nnse = calculate_result(y_true, y_pred)
            title = f'MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, SMAPE {smape:0.3g}, NNSE {nnse:0.3g}'
            
            if (summed_df[target].max() - summed_df[target].min()) >= 1e3:
                scale = 1e3
            else: scale = 1

            target_figure_name = None
            if save: target_figure_name = f'Summed_plot_{target}_{type}.jpg'

            fig = self.plot(
                summed_df, target, title, scale, base, target_figure_name, plot_error
            )
            figures.append(fig)
        
        return figures

    def individual_plot(
        self, df:DataFrame, fips:str, type:str='', save:bool=True, 
        base:int=None, plot_error:bool=False
    ):
        """
        Plots the prediction and observation for this specific county

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """

        assert fips in df['FIPS'].values, f'Provided FIPS code {fips} does not exist in the dataframe.'
        df = df[df['FIPS']==fips]

        figures = []
        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = df[target].values, df[predicted_column].values
            
            mae, rmse, rmsle, smape, nnse = calculate_result(y_true, y_pred)
            if (df[target].max() - df[target].min())>=2e3: scale = 1e3
            else: scale = 1

            target_figure_name = None
            if save: target_figure_name = f'Individual_plot_{target}_{type}_FIPS_{fips}.jpg'
            
            title = f'MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, SMAPE {smape:0.3g}, NNSE {nnse:0.3g}'
            fig = self.plot(df, target, title, scale, base, target_figure_name, plot_error)
            figures.append(fig)

        return figures

class PlotWeights:
    def __init__(
        self, figPath:str, max_encoder_length:int, 
        model:TemporalFusionTransformer, show:bool=True
    ):
        self.figPath = figPath
        if not os.path.exists(figPath):
            print(f'Creating folder {figPath}')
            os.makedirs(figPath, exist_ok=True)

        self.static_variables = model.static_variables
        self.encoder_variables = model.encoder_variables 
        self.decoder_variables = model.decoder_variables

        # print(f"Variables:\nStatic {self.static_variables} \nEncoder {self.encoder_variables} \nDecoder {self.decoder_variables}.")
        self.max_encoder_length = max_encoder_length
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
        fig.tight_layout() # might change y axis values

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
            interpretation
        )

        variables = interpretation["static_variables"].detach().cpu()
        figures["static_variables"] = self.make_selection_plot(
            "Static variables importance", variables, 
            self.static_variables, (10, 1.5*len(variables)+0.5)
        )

        # Dynamic variables
        variables = interpretation["encoder_variables"].detach().cpu()
        figures["encoder_variables"] = self.make_selection_plot(
            "Encoder variables importance", variables, 
            self.encoder_variables, (10, 1*len(variables) + 0.5)
        )

        # time unknown variable
        variables = interpretation["decoder_variables"].detach().cpu()
        figures["decoder_variables"] = self.make_selection_plot(
            "Decoder variables importance", variables, 
            self.decoder_variables, (10, 1.5*len(variables)+ 0.5)
        )

        return figures

    def plot_summed_attention(
        self, interpretation, title:str=None, figure_name:str=None, figsize=FIGSIZE
    ):
        fig, ax = plt.subplots(figsize=figsize)
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.max_encoder_length, attention.size(0) - self.max_encoder_length), attention
        )
        # plt.ylim(attention.min())
        ax.set_ylim(0)
        ax.set_xlabel("Position Index (n)")
        ax.set_ylabel("Attention Weight")

        if title is not None: ax.set_title(title)
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        ax.xaxis.set_major_locator(MultipleLocator(base=1))
        fig.tight_layout() # might change y axis values

        if figure_name is not None:
            plt.savefig(os.path.join(self.figPath, figure_name), dpi=DPI)

        if self.show: 
            plt.show()
        return fig

    def plot_attention(
        self, attention_mean:DataFrame, title:str='Attention comparison on different position index',
        figsize=(14, 8), step_size:int=1, figure_name:str=None, base:int=None, target_day:int=None, 
        limit:int=0, enable_markers=True
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

        if limit is None: count = max_encoder_length
        else: count = limit
        
        for encoder_length in range(-1, -1-max_encoder_length, -step_size):
            if count < 1: break

            if enable_markers:
                plt.plot(
                    attention_mean[x_column], attention_mean.loc[:, max_encoder_length + encoder_length], 
                    label=f'Position Index {encoder_length}', 
                    marker=markers[max_encoder_length + encoder_length]
                )
            else:
                plt.plot(
                    attention_mean[x_column], attention_mean.loc[:, max_encoder_length + encoder_length], 
                    label=f'Position Index {encoder_length}'
                )
            count -= 1

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
 
        ymin, ymax = plt.gca().get_ylim()
        weeks = attention_mean[x_column].dt.weekday.values

        if target_day is not None:
            plt.vlines(
                    x=attention_mean[weeks==target_day][x_column], ymin=ymin,
                    ymax=ymax, label=weekdays[target_day], color='olive'
                )
        
        plt.ylim(ymin)
        if base is None:
            x_first_tick = attention_mean[x_column].min()
            x_last_tick = attention_mean[x_column].max()
            x_major_ticks = DATE_TICKS
            ax.set_xticks(
                [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
            )
        else:
            # plt.xlim(attention_mean[x_column].min() - ONE_DAY, attention_mean[x_column].max() + ONE_DAY)
            ax.xaxis.set_major_locator(MultipleLocator(base=base))
        
        # plt.xticks(rotation = 15)
        
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        plt.ylabel('Attention Weight')
        if limit != 0:
            plt.legend()

        # fig.tight_layout() # might change y axis values

        if figure_name is not None: 
            plt.savefig(os.path.join(self.figPath, f'{figure_name}.jpg'), dpi=DPI)
            
        if self.show: plt.show()
        
    def plot_weekly_attention(
        self, attention_weekly:DataFrame, title:str= 'Attention comparison on different days of the week',
        figsize=(14, 8), step_size=1, limit:int=3, figure_name:str=None
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
                label=f'Position Index {encoder_length}', 
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
        plt.ylabel('Attention Weight')
        # plt.xlabel('Weekday')
        
        plt.gca().yaxis.set_major_formatter(self.weight_formatter)
        plt.legend()

        fig.tight_layout() # might change y axis values

        if figure_name is not None:
            plt.savefig(os.path.join(self.figPath, f'{figure_name}.jpg'), dpi=DPI)
            
        if self.show: plt.show()