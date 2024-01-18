import pandas as pd
import os

directory = "results"
output_file = "all_results.pkl"

pkl_files = [file for file in os.listdir(directory) if file.endswith('.pkl')]
df_list = [pd.read_pickle(os.path.join(directory, file)) for file in pkl_files]
big_dataframe = pd.concat(df_list, ignore_index=True)
big_dataframe.to_pickle(directory + '/' + output_file)
