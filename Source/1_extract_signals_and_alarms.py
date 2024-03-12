import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys
import zipfile_deflate64 as zipfile
from tqdm import tqdm
sys.path.append('..')

# Variables
alarm_vars = ['ShortYellowAlarm', 'YellowAlarm', 'RedAlarm', 'HardInop', 'SoftInop', 'SevereInop']
signal_vars = ['HF', 'RF', 'SpO2', 'ABPs', 'ABPd', 'ABPm', 'NiBDs', 'NiBDd', 'NiBDm',  'etCO2']

LOCATION_ZIP = '../../210111343819.zip'
LOCATION_DATA = '../data/'

s               = 30 # seconds
S               = str(s)+'S'

# Helpers
# def extract_signals_parallel(i):
#     zip_files = zipfile.ZipFile(LOCATION_ZIP)
#     name_files = zip_files.namelist()
#     len_files = len(name_files)

#     batches = []
#     idx_start = int((i)*(len_files/4))
#     idx_end = int((i+1)*(len_files/4))

#     for f in name_files[idx_start:idx_end]:
#         df = pd.read_csv(zip_files.open(f), sep=',', low_memory=False, compression='infer', on_bad_lines='skip')
#         df = df.loc[:, ['MRN', 'measurementime','ParameterText','ParameterValue']]
#         df = df[~df.ParameterText.isin(signal_vars)]
#         batches.append(df)

#     df = pd.concat(batches, axis=0, ignore_index=True)
#     df.to_csv('batch_SIGNALS'+str(i)+'.csv')
#     del df
    
def extract_signals(i):
    
    zip_files = zipfile.ZipFile(LOCATION_ZIP)
    name_files = zip_files.namelist()
    len_files = len(name_files)
    batches = []
    idx_start = int((i)*(len_files/4))
    idx_end = int((i+1)*(len_files/4))

    for f in tqdm(name_files[idx_start:idx_end]):
        df = pd.read_csv(zip_files.open(f), sep=',', low_memory=False, compression='infer', on_bad_lines='skip')
        df = df.loc[:, ['MRN', 'measurementime','ParameterText','ParameterValue']]
        df = df[df.ParameterText.isin(signal_vars)]
        df.loc[:, 'measurementime'] =  pd.to_datetime(df.loc[:, 'measurementime'])
        df.loc[:, 'ParameterValue'] = df.loc[:, 'ParameterValue'].astype(float)
        df = df.groupby(['MRN','ParameterText',  pd.Grouper(key = 'measurementime', freq=S)]).agg([np.nanmean]).reset_index()
        df.columns = df.columns.droplevel(1)

        # # Separate PVC from other measurements
        # pvc_df = df[df.ParameterText == 'PVC']
        # non_pvc_df = df[df.ParameterText != 'PVC']

        # # Aggregate PVC using sum
        # pvc_agg = pvc_df.groupby(['MRN', 'ParameterText', pd.Grouper(key='measurementime', freq=S)]).agg(np.nansum).reset_index()

        # # Aggregate non-PVC using mean
        # non_pvc_agg = non_pvc_df.groupby(['MRN', 'ParameterText', pd.Grouper(key='measurementime', freq=S)]).agg(np.nanmean).reset_index()

        # # Combine the results
        # final_df = pd.concat([pvc_agg, non_pvc_agg])

        # Sort by MRN and measurementime if needed to maintain order
        df = df.sort_values(by=['MRN', 'measurementime'])
        df = df.loc[:, ['MRN', 'measurementime', 'ParameterText', 'ParameterValue']]


        batches.append(df)
        del df

    df = pd.concat(batches, axis=0)
    df.to_csv(LOCATION_DATA+'batch_SIGNALS'+str(i)+'.csv')
    del df

def extract_alarms_parallel(i):
    zip_files = zipfile.ZipFile(LOCATION_ZIP)
    name_files = zip_files.namelist()
    len_files = len(name_files)

    batches = []
    idx_start = int((i)*(len_files/4))
    idx_end = int((i+1)*(len_files/4))

    for f in name_files[idx_start:idx_end]:
        df = pd.read_csv(zip_files.open(f), sep=',', low_memory=False, compression='infer', on_bad_lines='skip')
        df = df.loc[:, ['MRN', 'measurementime','ParameterText','ParameterValue']]
        df = df[df.ParameterText.isin(alarm_vars)]
        df.loc[:, 'measurementime'] =  pd.to_datetime(df.loc[:, 'measurementime'])
        df.loc[:, 'count'] = np.ones((df.shape[0], ))
        df = df.groupby(['MRN','ParameterText', 'ParameterValue', pd.Grouper(key = 'measurementime', freq=S)]).agg(np.nansum).reset_index()
        df.columns = df.columns.droplevel(1)
        batches.append(df)
        del df

    df = pd.concat(batches, axis=0, ignore_index=True)
    df.to_csv(LOCATION_DATA+'batch_ALARMS'+str(i)+'.csv')
    del df

# 1. Extract SIGNALS and ALARMS of interest

print('Extract signal...')
[extract_signals(i) for i in tqdm(range(4))]

batches_signal  = []
for i in tqdm(range(4)):
    df = pd.read_csv(LOCATION_DATA+'batch_SIGNALS'+str(i)+'.csv', low_memory=False, index_col=0, on_bad_lines='skip')
    batches_signal.append(df)
    del df

data_signals = pd.concat(batches_signal, axis=0)
data_signals.to_csv(LOCATION_DATA+'1_all_signals.csv')

del data_signals

print('Extract alarms...')
Parallel(n_jobs=4)(delayed(extract_alarms_parallel)(i) for i in range(4))

batches_alarm  = []
for i in tqdm(range(4)):
    df = pd.read_csv(LOCATION_DATA+'batch_ALARMS'+str(i)+'.csv', low_memory=False, index_col=0, on_bad_lines='skip')
    batches_alarm.append(df)
    del df

data_alarms = pd.concat(batches_alarm, axis=0)
data_alarms.to_csv(LOCATION_DATA+'1_all_alarms.csv')

del data_alarms

print('End :)')