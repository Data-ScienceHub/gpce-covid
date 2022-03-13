All changes to model must be made withing `config.json`

<b> Notes on `config.json` </b>
+ `static_locs`, `future_locs` are the dataframe (column) locations of the inputs.
+ The locations of the inputs will be the first portion of the dataframe and the target(s) will be the last column of the dataframe
+ <b> Categorical Features currently NOT supported </b>


```
# to process and save data
python data_saveMain.py -p <PATH-TO-CONFIG.JSON> -f <PATH-TO-RAW-DATA-CSVS> -o <DIR-TO-SAVE-TO>
```

```
# to train TFT
python main.py -p <PATH-TO-CONFIG.JSON> -c <CHECKPOINT-DIR> -d <PATH-TO-DATA>
```
