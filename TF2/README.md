All changes to model must be made withing `config.json`

<b> Notes on `config.json` </b>
+ `static_locs`, `future_locs` are the dataframe (column) locations of the inputs.
+ The locations of the inputs will be the first portion of the dataframe and the target(s) will be the last column of the dataframe
+ <b> Categorical Features currently NOT supported </b>


```
# to process and save data
python prepareData_main.py -p <CONFIG PATH> -f <RAW CSV FILE DIR PATH> -o <OUTPUT DATA DIR AND NAME>
```

```
# to train TFT
python main.py -p <PATH-TO-CONFIG.JSON> -c <CHECKPOINT-DIR> -d <PATH-TO-DATA>
```

<b> Inference: </b>

This piece is still a work in progress. The current functionality is to load the model weights from the checkpoint and get predictions on a dataset and visualize this as the aggregated (summed across all counties for each day) covid cases.

```
python inference.py -p <CONFIG PATH> -c <CHECKPOINT-DIR> -d <DATH PATH> -f <DIR TO SAVE FIGURES>
```

Current directory structure (Note the checkpoints are my specific checkpoints, but this is what a general structure may look like):

├── TFTModel.py
├── __pycache__
│   ├── TFTModel.cpython-38.pyc
│   ├── data_manager.cpython-38.pyc
│   ├── data_prep.cpython-38.pyc
│   └── param_manager.cpython-38.pyc
├── checkpoints
│   ├── TFT-stratum2
│   │   ├── checkpoint
│   │   ├── ckpt-10.data-00000-of-00001
│   │   ├── ckpt-10.index
│   │   ├── ckpt-11.data-00000-of-00001
│   │   ├── ckpt-11.index
│   │   ├── ckpt-12.data-00000-of-00001
│   │   ├── ckpt-12.index
│   │   ├── ckpt-8.data-00000-of-00001
│   │   ├── ckpt-8.index
│   │   ├── ckpt-9.data-00000-of-00001
│   │   └── ckpt-9.index
│   ├── TFT-stratum2-April2
│   │   ├── checkpoint
│   │   ├── ckpt-2.data-00000-of-00001
│   │   ├── ckpt-2.index
│   │   ├── ckpt-3.data-00000-of-00001
│   │   ├── ckpt-3.index
│   │   ├── ckpt-4.data-00000-of-00001
│   │   ├── ckpt-4.index
│   │   ├── ckpt-5.data-00000-of-00001
│   │   ├── ckpt-5.index
│   │   ├── ckpt-6.data-00000-of-00001
│   │   └── ckpt-6.index
│   └── TFT-stratum2-test
│       ├── checkpoint
│       ├── ckpt-10.data-00000-of-00001
│       ├── ckpt-10.index
│       ├── ckpt-11.data-00000-of-00001
│       ├── ckpt-11.index
│       ├── ckpt-12.data-00000-of-00001
│       ├── ckpt-12.index
│       ├── ckpt-8.data-00000-of-00001
│       ├── ckpt-8.index
│       ├── ckpt-9.data-00000-of-00001
│       └── ckpt-9.index
├── config.json
├── data
│   ├── TFTdfCurrent.csv
│   ├── TFTdfNew.csv
│   └── TFTdfTotal.csv
├── data_manager.py
├── data_preparation.py
├── figures
│   └── CovidA21-TFT2-Feb2020toJuly2021-Iteration.png
├── inference.py
├── main.py
├── param_manager.py
├── prepareData_main.py
└── requirements.txt


There may be issues with this code, but I have trained several runs of models on it no problem. Any issue please let me know or open an issue.

The remaining pieces to address are:
+ county level plots
+ clean up additional code
