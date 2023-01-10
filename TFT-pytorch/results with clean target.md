# Experiment Results with Targets Cleaned of Outliers

The following experiments are on inputs and outputs where outliers were removed. The raw target has very high spikes during covid peak times, specially during late 2021 and early 2022 due to dominant mutants. Removing the outliers show improved loss metrics.

## All counties
Model is trained for max 60 epochs and the best model by validation loss is used for results.

### Train
![daily-cases](/TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Train_error.jpg)

### Validation

![daily-cases](/TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Validation.jpg)

### Test
![daily-cases](/TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Test.jpg)

### Attention on prior days (train data)

The closer the past day is to the present day, the more attention weight it has. Also the same weekday in the previous week (Time index -7), has similary high weight as the previous day (Time index -1). So yesterday's data and same weekday's data from previous week are most important for model prediction.

![train-attention](/TFT-pytorch/results/TFT_baseline/figures/Train_attention.jpg)

### Weekly attention

Attention weights mostly peak on Friday, as seen from the mean values. And it is lowest on Saturday/Sunday. This is because covid cases often peaked at Friday. On weekends less cases and deaths are reported, so eventually they have less impact on the model attention.

* Train ![train-weekly-attention](/TFT-pytorch/results/TFT_baseline/figures/Train_weekly_attention.jpg)

### Attention over time

Attentison peaks are shown mostly on wednesady and thursday in the validation (e.g. 2022-01-06) and test data (e.g. 2022-03-31).

* Train ![train-mean-attention](/TFT-pytorch/results/TFT_baseline/figures/Train_daily_attention.jpg)

### Variable importance (Train data)

Weeekly patterns are very important as seen from the attentions. That is also evident from the variable importance, as weekly features get most importance. Then past observations of the target variables. Since covid cases/deaths in the past weeks can help learn the trend and predict the future better.

* Static variables ![Train_static_variables](/TFT-pytorch/results/TFT_baseline/figures/Train_static_variables.jpg)
* Encoder variables ![Train_encoder_variables](/TFT-pytorch/results/TFT_baseline/figures/Train_encoder_variables.jpg)
* Decoder variables ![Train_decoder_variables](/TFT-pytorch/results/TFT_baseline/figures/Train_decoder_variables.jpg)

## Top 500 counties

The final results are result received from model early stopped with patience 3.

### Train

![daily-cases](/TFT-pytorch/results/TFT_top_500/figures/Summed_plot_Cases_Train_error.jpg)

### Validation

![daily-cases](/TFT-pytorch/results/TFT_top_500/figures/Summed_plot_Cases_Validation.jpg)

### Test

![daily-cases](/TFT-pytorch/results/TFT_top_500/figures/Summed_plot_Cases_Test.jpg)

### Attention on prior days (train data)

The closer the past day is to the present day, the more attention weight it has. Also the same weekday in the previous week (Time index -7), has similary high weight as the previous day (Time index -1). So yesterday's data and same weekday's data from previous week are most important for model prediction.

![train-attention](/TFT-pytorch/results/TFT_top_500/figures/Train_attention.jpg)

### Weekly attention

Attention weights mostly peak on Friday, as seen from the mean values. And it is lowest on Saturday/Sunday. This is because covid cases often peaked at Friday. On weekends less cases and deaths are reported, so eventually they have less impact on the model attention.

Train ![train-weekly-attention](/TFT-pytorch/results/TFT_top_500/figures/Train_weekly_attention.jpg)

### Attention over time

Attentison peaks are shown mostly on wednesady and thursday in the validation (e.g. 2022-01-06) and test data (e.g. 2022-03-31).

* Train ![train-mean-attention](/TFT-pytorch/results/TFT_top_500/figures/Train_daily_attention.jpg)

### Variable importance (Train data)

Weeekly patterns are very important as seen from the attentions. That is also evident from the variable importance, as weekly features get most importance. Then past observations of the target variables. Since covid cases/deaths in the past weeks can help learn the trend and predict the future better.

* Static variables ![Train_static_variables](/TFT-pytorch/results/TFT_top_500/figures/Train_static_variables.jpg)
* Encoder variables ![Train_encoder_variables](/TFT-pytorch/results/TFT_top_500/figures/Train_encoder_variables.jpg)
* Decoder variables ![Train_decoder_variables](/TFT-pytorch/results/TFT_top_500/figures/Train_decoder_variables.jpg)