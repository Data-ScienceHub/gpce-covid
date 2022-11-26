import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
from utils import calculate_result

# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlepad=15)

# set tick width
plt.rcParams['xtick.major.size'] = 15 # default 3.5
plt.rcParams['xtick.major.width'] = 2 # default 0.8 

plt.rcParams['ytick.major.size'] = 15 # default 3.5
plt.rcParams['ytick.major.width'] = 2 # 0.8 

plt.rcParams['lines.linewidth'] = 2.5

DPI = 200
FIGSIZE = (12, 7)
DATE_TICKS = 5

def plot_train_history(
    history, title:str, figure_path:str=None, 
    figsize=FIGSIZE, base:int=None, show_image:bool=True
    ):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure(figsize=figsize)

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    loss_formatter = StrMethodFormatter('{x:,.3f}')
    plt.gca().yaxis.set_major_formatter(loss_formatter)

    if base is not None:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=base))

    plt.title(title)
    plt.legend()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=DPI)

    if show_image:
        plt.show()

def plot_predition(
    df:DataFrame, target:str, plot_error:bool=False, 
    figure_path:str=None, figsize=FIGSIZE, show_image:bool=True
):
    x_major_ticks = DATE_TICKS
    predicted = 'Predicted_'+ target

    # make sure to do this before the aggregation
    mae, rmse, rmsle, smape, nnse = calculate_result(df[target].values, df[predicted].values)
    title = f'{target} MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, SMAPE {smape:0.3g}, NNSE {nnse:0.3g}'

    df = df.groupby('Date')[
        [target, predicted]
    ].aggregate('sum').reset_index()
    
    _, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.plot(df['Date'], df[target], color='blue', label='Ground Truth')
    plt.plot(df['Date'], df[predicted], color='green', label='Predicted')
    if plot_error:
        plt.plot(df['Date'], abs(df[target] - df[predicted]), color='red', label='Error')
    ax.set_ylim(0, ax.get_ylim()[-1]*1.05)

    label_text, scale, unit = [], 1e3, 'K'
    for loc in plt.yticks()[0]:
        if loc == 0:
            label_text.append('0')
        else:
            label_text.append(f'{loc/scale:0.5g}{unit}')
        
    ax.set_yticks(plt.yticks()[0])
    ax.set_yticklabels(label_text)
    
    plt.ylabel(f'Daily {target}') 

    x_first_tick = df['Date'].min()
    x_last_tick = df['Date'].max()
    ax.set_xticks(
        [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
    )

    if plot_error:
        plt.legend(framealpha=0.3, edgecolor="black", ncol=3)
    else:
        plt.legend(framealpha=0.3, edgecolor="black", ncol=2)
    
    if figure_path is not None:
        plt.savefig(figure_path, dpi=200)

    if show_image:
        plt.show()