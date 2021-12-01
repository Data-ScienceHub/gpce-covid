# Expedition_COVID


<b> Andrej Notes: </b>

In /code/code_andrej you can find an example of the derivative rurality notebook in rurality_derivate and an example of the the regular cases and deaths split by rurality in rurality_original 
The strata are as follows:

{counties |  3 <= median rurality <= 5} - This gives us 624 counties.
 
{counties subset | MAD < 1} - this is 391 counties
 
{counties subset | 1 <= MAD < 2} - 175 counties
 
{counties subset | 2 <= MAD < 3} - 47 counties
 
<b> This one is no longer in use as the sample is too small </b>
{counties subset | 3 <= MAD < 4 } - 11 counties

I have also added differenced and derivate data to dataset_raw/

<b> I am currently working on automating pdf saving in notebook code so that it is much more painless to organize all of the results. 
