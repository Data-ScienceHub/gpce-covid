# Summary of related works
Title | Pubished on | Model
--- | --- | --- 
AICov: An Integrative Deep Learning Framework forCOVID-19 Forecasting with Population Covariates | Journal of Data Science, 2021 | LSTM
Inter-Series Attention Model for COVID-19 Forecasting | SIAM International Conference on Data Mining 2021 | Attention Crossing Time Series
Interpretable Sequence Learning for COVID-19 Forecasting | arxiv 2021 | SEIR + Encoder
Temporal Fusion Transformersfor Interpretable Multi-horizon Time Series Forecasting | Journal of Forecasting, 2021 | Temporal Fusion Transformer (TFT)
Predictions for COVID-19 with deep learning models of LSTM, GRU and Bi-LSTM | Chaos, Solitons & Fractals, 2020 | ARIMA, SVR, LSTM, GRU and Bi-LSTM
Time Series Forecasting With Deep Learning A Survey | Philosophical Transactions of the Royal Society A, 2021 | Survey
Deep learning methods for forecasting COVID-19 time-Series data | Chaos, Solitons & Fractals, 2020 | RNN,LSTM,Bi-LSTM,GRUs, and VAE 
COVID-19 predictability in the United States using Google Trends time series | Nature, Scientific report2 2020 | Regression
Time series forecasting of COVID-19 transmission in Canada using LSTM networks | Chaos, Solitons & Fractals, 2020 | LSTM 
Multi-Horizon Time Series Forecasting with Temporal Attention | KDD, 2019 | Encoder-Decoder
Exploring Interpretable LSTM Neural Networks over Multi-Variable Data | ICML, 2019 | IMV-LSTM

# Related works
## [AICov: An Integrative Deep Learning Framework forCOVID-19 Forecasting with Population Covariates](https://jds-online.org/journal/JDS/article/124/info)
Model  COVID-19  and  other  pandemics  in  terms  of  the  broader social contexts: population’s socioeconomic, health, and behavioralrisk factors at their specific locations.
Use  deep  learning  strategies  based  on  LSTM  and event modeling. Proposed AICov, provides  an  integrative  deep  learningframework for COVID-19 forecasting with population covariates.

## [Inter-Series Attention Model for COVID-19 Forecasting](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.56)

Can a direct data-driven approach without modeling the disease spreading dynamics outperform the well referred compartmental models? </br>Neural forecasting model:ttention Crossing Time Series (ACTS). makes forecasts via comparing patterns across time series obtained from multiple regions.

ACTS mainly relies on matching patterns similar to the current input from the training data to make predictions. </br> Detrending: adopt a learnable Holt smoothing model to remove long-term trends. </br> Segment Embedding: apply min-max normalization to the cumulative sum of incidence time series.instantiate the function  using a convo-lution layer with d feature maps to the scaled segment and time-dependent features. </br> Inter-series Attention: use dot-product attention to compare segments and combine the values.

## [Interpretable Sequence Learning for COVID-19 Forecasting](https://arxiv.org/abs/2008.00646)
Integrates machine learning into compartmental disease modeling (e.g., SEIR). Use interpretable encoders to incorporate covariates and improve performance model can be applied at different geographic resolutions. 

Adapt the standard SEIR model with some major changes: </br> 1. Undocumented infected and recovered compartments; </br> 2. Hospitalized, ICU and ventilator compartments; </br> 3.Partial immunity; </br> 4. Other Assumptions:published COVID-19 death counts are coming from documented cases, not undocumented.  entire population is invariant assume a fixed sampling interval of 1 day. </br> When encoding covariates, the Interpretable encoder architecture is adopted.

## [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637)
 Propose an attention based architecture to solve the problem of Multi-horizon forecasting.
 
Major constituents of TFT:

1. Gating mechanisms to skip over any unused components of the architecture.
2. Variable selection networks to select relevant input variables at each time step. 
3. Static covariate encoders. 
4. Temporal processing.

TFT uses specialized components to process all inputs (ie, static covariates, a priori known inputs, and observation inputs) that usually appear in multilevel prediction problems. Specifically, these include:

1. Time processing components based on sequence-to-sequence and attention, capturing time-varying relationships at different time scales.
2. Static covariate encoders, which enable the network to predict time in static metadata conditions. 
3. Control components to make Skip any unnecessary parts of the network. For a given data set.
4. Variable selection network, select relevant input characteristics at each time step. 
5. Quantile prediction to obtain a data that spans all prediction layers Output interval.

## [Predictions for COVID-19 with deep learning models of LSTM, GRU and Bi-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0960077920306081)

Compare autoregressive integrated moving average (ARIMA), support vector regression (SVR), long shot term memory (LSTM), bidirectional long short term memory (Bi-LSTM). The quality of this paper is relatively low.

## [Time Series Forecasting With Deep Learning A Survey](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209)
Summarizes the existing newer forecasting methods, Basic Building Blocks, Multi-horizon Forecasting, Incorporating Domain Knowledge with Hybrid Models

The time series mentioned in the paper is a fairly universal model, taking into account exogenous variables and static metadata
Different encoder architectures include CNN Model, RNN Model, Attention-based Model.


## [Deep learning methods for forecasting COVID-19 time-Series data](https://www.sciencedirect.com/science/article/abs/pii/S096007792030518X)
Comparative study of five deep learning methods to forecast the number of new cases and recovered cases. RNN,LSTM,Bi-LSTM,GRUs, and VAE  algorithms have been applied for global forecasting of COVID-19 cases based on a small volume of data. 

Deep learning models are able to capture time-variant properties and relevant patterns of past data and forecast the future tendency of COVID-19 time-series data. The forecasting results show the VAE model by achieving higher accuracy compared to the other models for one-step forecasting.

## [COVID-19 predictability in the United States using Google Trends time series](https://www.nature.com/articles/s41598-020-77275-9)
Google Trends time series predictability of COVID-19. Find a robust regression analysis that is the appropriate statistical approach to taking against the presence of outliers in the sample while also mitigating small sample estimation bias. 

Results indicate that there are statistically significant correlations between Google Trends and COVID-19 data, while the estimated models exhibit strong COVID-19 predictability.

## [Time series forecasting of COVID-19 transmission in Canada using LSTM networks]()

LSTM approach. 2, 4, 6, 8, 10, 12 and 14 th day predictions
evaluate the key features to predict the trends and possible stopping time of the current COVID-19 outbreak.
Data set: Johns Hopkins University

This paper is an example of directly using LSTM to predict COVID19.
This article also performed a statistical analysis of the input data. performed Augmented Dickey Fuller (ADF) test on the input data. ADF is the stan- dard unit root test to find the impact of trends on the data and its results are interpreted by observing p-values of the test.
Policies or decisions taken by government will greatly af- fect the current outbreak.

## [Multi-Horizon Time Series Forecasting with Temporal Attention](https://dl.acm.org/doi/abs/10.1145/3292500.3330662) 
Multi-horizon forecasting (forecasting on multiple steps in future time) learn constructing hidden patterns’ representations with deep neural networks and attending to different parts of the history for forecasting the future. 

Adopt sequence-to-sequence learning pipeline to encode historical (and future) input variables and decode to future predictions. the encoder part is a two-layer LSTM which maps the history of sequences to latent representations.

The decoder is another recurrent network which takes the encoded history as its initial state, and the future information as inputs to generate the future sequence as outputs. 

## [Exploring Interpretable LSTM Neural Networks over Multi-Variable Data](http://proceedings.mlr.press/v97/guo19b/guo19b.pdf) 
Capture different dynamics in multi-variable time series and distinguish the contribution of variables to the prediction Interpretable Multi-Variable LSTM (IMV-LSTM)

The structure of IMV-LSTM is slightly different from ordinary LSTM. </br> Mixture attention mechanism: Temporal attention is first applied to the sequence of hidden states corresponding to each variable, to obtain the summarized history of each variable. Then by using the history enriched hidden state, variable attention is derived to merge variable-wise states. </br>Datasets: PM2.5(41700 multi-variable sequences); PLANT; SML.IMV-LSTM family outperforms baselines by around 80% at most.