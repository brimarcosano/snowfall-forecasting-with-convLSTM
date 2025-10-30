# LSTM Only Documentation


### To run:
- LSTM/main_lstm_only.py


### Data files:

- LSTM Folder > Data/temporal_into_df.py is first run to generate CSV of weekly or daily data, but the weekly averaged data is already saved as "temporal_data_weekly.csv" and used in the code. Raw data files are under Data > Raw and Avg > Raw Tabular.


### Breakdown of process and main modules called from main file:

- main_lstm_only.py > Data/preprocess_data.py (all data cleaning and train/val splitting) > Data/LSTM_CustomDataset.py > Scripts/train_lstm_only.py (train, validation, test, prediction loops) > Data/preprocess_data.py (for inversion of normalized data before plotting) > Scripts/plot.py


### Other notes:

- Code is currently set up to train on ONE New Hampshire county (set in preprocess_data.py), however I recieved similar results including multiple counties.
