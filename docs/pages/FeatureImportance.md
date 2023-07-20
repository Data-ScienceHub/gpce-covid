# Feature Importance

| Feature | Static | Observed | Known
|---------|--------|----------| -----|
|  Cases       |  |   **38.26%**  |      |
| Age Distribution | **54.45%** | |      |
| Health Disparities | 45.55% | |    |
| Vaccination |    | 11.28%   |      |
| Disease Spread | | 16.32%   |      |
| Transmissble Cases| | 3.26% |      |
| Social Distancing|  | 3.35% |
| SinWeekly|       |          |**72.85%**|
| CosWeekly|       |          |27.15%|

The above table is calculated from Variable Selection Network across the training set. We first computed the sum of weights assigned to each variable and then normalized the weights of each feature. According to the table, we found that Age Distribution has the highest weights among Static featuers and Cases among Observed features. 