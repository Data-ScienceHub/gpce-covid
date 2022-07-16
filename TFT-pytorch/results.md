# Experiment Results

## Top 500 counties

The training was kept running for 60 epochs. Outliers removed from both inputs and targets.

### Train

* Cases
![daily-cases](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Cases_Train_avg.jpg)

* Deaths
![daily-deaths](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Deaths_Train_avg.jpg)

### Validation

* Cases
![daily-cases](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Cases_Validation_avg.jpg)

* Deaths
![daily-deaths](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Deaths_Validation_avg.jpg)

### Test

* Cases
![daily-cases](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Cases_Test_avg.jpg)

* Deaths
![daily-deaths](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Summed_plot_Deaths_Test_avg.jpg)

### Attention on prior days (train data)

The closer the past day is to the present day, the more attention weight it has. Also the same weekday in the previous week (Time index -7), has similary high weight as the previous day (Time index -1). So yesterday's data and same weekday's data from previous week are most important for model prediction.

![train-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_attention.jpg)

### Weekly attention

Most attention seems to be on thursday and sunday. And it is lowest on Saturday. This is because covid cases often peaks at Friday. Sometimes on Monday. So the data from the prior day will have get more attention.

* Train ![train-weekly-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_weekly_attention.jpg)
* Validation ![validation-weekly-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Validation_weekly_attention.jpg)
* Test ![test-weekly-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Test_weekly_attention.jpg)

### Attention over time

Attentison peaks are shown mostly on thursday in the validation (e.g. 2022-01-06) and test data (e.g. 2022-03-31).

* Train ![train-mean-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_mean_attention.jpg)
* Validation ![validation-mean-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Validation_mean_attention.jpg)
* Test ![test-mean-attention](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Test_mean_attention.jpg)

### Vaiable importance (train data)

Weeekyl patterns are very important as seen from the attentions. That is also evident from the variable importance, as weekly features get most importance. Then past observations of the target variables. Since covid cases/deaths in the past weeks can help learn the trend and predict the future better.

* Static variables ![Train_static_variables](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_static_variables.jpg)
* Encoder variables ![Train_encoder_variables](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_encoder_variables.jpg)
* Decoder variables ![Train_decoder_variables](/TFT-pytorch/results/top_500_target_cleaned_unscaled/figures/Train_decoder_variables.jpg)