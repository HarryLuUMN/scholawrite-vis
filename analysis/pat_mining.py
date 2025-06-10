#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%
data1 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00000-of-00002.parquet', engine='pyarrow')
data2 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00001-of-00002.parquet', engine='pyarrow')
df = pd.concat([data1, data2], axis=0, ignore_index=True)
df.head()

#%%
project_tables = {project: group for project, group in df.groupby('project')}
for project, table in project_tables.items():
    project_tables[project] = table.sort_values(by='timestamp', ascending=True)
project_tables[1].head()

#%%
exp_data = pd.read_csv("/kaggle/input/experiment-data/model_outputs (1).csv")
exp_data.head()

#%%
model_tables = {}

for model_name, group_df in exp_data.groupby('model'):
    sorted_df = group_df.sort_values(by='index_in_file').reset_index(drop=True)
    model_tables[model_name] = sorted_df

for model_name, df in model_tables.items():
    print(f"Model: {model_name}")

#%%
def longest_contiguous_label_run_np(label_series):
    arr = np.array(label_series)
    change_points = np.where(arr[1:] != arr[:-1])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(arr)]))
    run_lengths = np.diff(change_points)
    run_labels = arr[change_points[:-1]]
    max_idx = np.argmax(run_lengths)
    return run_labels[max_idx], run_lengths[max_idx]


