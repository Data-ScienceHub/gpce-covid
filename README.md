# Expedition_COVID


<b> Andrej Notes: </b>

In /code/code_andrej you can find an example of the derivative rurality notebook in rurality_derivate and an example of the the regular cases and deaths split by rurality in rurality_original 

---

The strata are as follows:

{counties |  3 <= median rurality <= 5} - This gives us 702 counties.

| Median Rurality | Count Counties |
| -------------   | -------------  |
|  3.0            | 45             |
|  3.5            | 7              |
|  4.0            | 397            |
|  4.5            | 71             |
|  5.0            | 182            |
| --------------- | -------------- |
| Total           | 702            |

---
 
{counties subset | MAD < 1} - this is 434 counties

| Median Rurality | Count Counties |
| -------------   | -------------  |
|  3.0            | 9              |
|  4.0            | 307            |
|  4.5            | 43             |
|  5.0            | 75             |
| --------------- | -------------- |
| Total           | 434            |

---
 
{counties subset | 1 <= MAD < 2} - 188 counties

| Median Rurality | Count Counties |
| -------------   | -------------  |
|  3.0            | 33             |
|  3.5            | 6              |
|  4.0            | 63             |
|  4.5            | 7              |
|  5.0            | 79             |
| --------------- | -------------- |
| Total           | 188            |
 
---
 
{counties subset | 2 <= MAD < 3} - 69 counties

| Median Rurality | Count Counties |
| -------------   | -------------  |
|  3.0            | 3              |
|  3.5            | 1              |
|  4.0            | 26             |
|  4.5            | 21             |
|  5.0            | 18             |
| --------------- | -------------- |
| Total           | 69             |

---
 
<b> This one is no longer in use as the sample is too small </b>
{counties subset | 3 <= MAD < 4 } - 11 counties

I have also added differenced and derivate data to dataset_raw/

<b> I am currently working on automating pdf saving in notebook code so that it is much more painless to organize all of the results. 
