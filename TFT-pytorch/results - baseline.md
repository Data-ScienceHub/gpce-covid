# Experiment with Baseline model

The [Baseline model](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.baseline.Baseline.html?highlight=baseline) works by simply repeating the last known observation from the past. It gives a very simple but strong baseline to compare with our model.

Since the model last repeats last know value, there is no training. And features other than the known observations are not used. The results here are still shown in train/validation/test split to be comparable to other models.

The results are on all 3,142 counties.

## Train

![daily-cases](/TFT-pytorch/results/Naive%20Model/figures/Summed_plot_Cases_Train.jpg)

## Validation

![daily-cases](/TFT-pytorch/results/Naive%20Model/figures/Summed_plot_Cases_Validation.jpg)

## Test

![daily-cases](/TFT-pytorch/results/Naive%20Model/figures/Summed_plot_Cases_Test.jpg)