cd ../v0/script

python prepare_data.py -c '../../reproduce/rurality_cut/1/config.json' -d '../../dataset_raw/CovidMay17-2022' -o '../../reproduce/rurality_cut/1/'
python prepare_data.py -c '../../reproduce/rurality_cut/2/config.json' -o '../../reproduce/rurality_cut/2/' -ch '../../reproduce/Total.csv'
python prepare_data.py -c '../../reproduce/rurality_cut/3/config.json' -o '../../reproduce/rurality_cut/3/' -ch '../../reproduce/Total.csv'
python prepare_data.py -c '../../reproduce/population_cut/config.json' -o '../../reproduce/population_cut/' -ch '../../reproduce/Total.csv'

python train.py -c '../../reproduce/population_cut/config.json' -d '../../reproduce/population_cut/Population_cut.csv' -o '../../reproduce/population_cut/output' -p '../../reproduce/population_cut/output/checkpoint'