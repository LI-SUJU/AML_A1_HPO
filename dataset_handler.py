import pandas as pd
import os

# Define the paths to the CSV files
parent_dir = os.path.dirname(os.path.abspath(__file__))
lcdb_config_path = os.path.join(parent_dir, 'lcdb_configs.csv')
config_performances_dir = os.path.join(parent_dir, 'config-performances')

# Read the CSV files
lcdb_config_df = pd.read_csv(lcdb_config_path)

# Assuming the other three CSV files are named config1.csv, config2.csv, config3.csv
config1_path = os.path.join(config_performances_dir, 'config_performances_dataset-6.csv')
config2_path = os.path.join(config_performances_dir, 'config_performances_dataset-11.csv')
config3_path = os.path.join(config_performances_dir, 'config_performances_dataset-1457.csv')

# Read the other CSV files
config1_df = pd.read_csv(config1_path)
config2_df = pd.read_csv(config2_path)
config3_df = pd.read_csv(config3_path)

# Create a dictionary to hold the DataFrames
dataframes_dict = {
    'lcdb': lcdb_config_df,
    'dataset-6': config1_df,
    'dataset-11': config2_df,
    'dataset-1457': config3_df
}

# Export the dictionary
def get_dataframes():
    return dataframes_dict