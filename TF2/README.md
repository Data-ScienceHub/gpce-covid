All changes to model must be made withing `config.json`


```
# to process and save data
python data_saveMain.py -p <PATH-TO-CONFIG.JSON> -f <PATH-TO-RAW-DATA-CSVS> -o <DIR-TO-SAVE-TO>
```

```
# to train TFT
python main.py -p <PATH-TO-CONFIG.JSON> -c <CHECKPOINT-DIR> -d <PATH-TO-DATA>
```
