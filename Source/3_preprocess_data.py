import pandas as pd
import numpy as np

LOCATION_DATA   = 'data/'
FREQ = '1Min'

cols_to_check = ['SI_SpO2_zoekend', 'HI_ABP_Geen_transdcr',
       'HI_ABP_buiten_bereik', 'SI_SpO2_ctrl_sensor', 
       'HI_SpO2_geen_sensor', 'SI_ABP_artefact', 
       'HI_ABP_meting_misl.', 'SI_SpO2_zwak_signaal',
       'SI_Som_ECG_alrmn_uit',
       'HI_ABP_onderbroken', 'SI_ECG/arr_alrmn_uit', 'HI_ECG_afl_los',
       'SI_SpO2_lage_PFI', 'HI_Niet_te_leren_ECG',
       'HI_SpO2_geen_puls', 'SI_ABP_Wijzig_schl',
       'HI_SpO2_sensor_los', 'HI_Resp_afl_los', ' HI_SpO2_grillig', 'HI_ABP_contr_mancht', 'HI_ABP_defect',
       'HI_ABP_geen_puls', 'HI_ABP_slecht_signaal', 'HI_Ctrlr_ECG_kabel',
       'HI_ECG_contr_kabel', 'HI_EcgOutput_defect', 'HI_SpO2_defect',
       'HI_SpO2_intrfrentie', 'HI_SpO2_onbk_sensor', 'HI_SpO2_sens_defect',
       'HI_V_Afl_los', 'SEI_ABP_manch_ovrdruk', 'SI_ABP_contr_nulling', 'SI_ECG_contr_kabel',
       'SI_SpO2_trageUpdate']

sig_cols = ['ABPd', 'ABPm', 'ABPs', 'HF', 'PVC', 'RF', 'SpO2', 'etCO2', 'etCO2_bin']

al_cols = ['RA_ VTach', 'RA_ABP_losgeraakt', 'RA_ABPm_high',
       'RA_ABPm_low', 'RA_ABPs_high', 'RA_ABPs_low', 'RA_Apneu', 'RA_Brady',
       'RA_Brady/P_low', 'RA_Desaturatie', 'RA_Extreme_brady',
       'RA_Extreme_tachy', 'RA_Tachy', 'RA_Tachy/P_high', 'RA_Vent_fibr/tach',
       'RA_asystolie', 'RA_xBrady_low', 'RA_xTachy_high',
       'SYA_AFIB', 'SYA_Doublet_PVCs',
       'SYA_Einde onreg_HF', 'SYA_HF_high', 'SYA_HF_low',
       'SYA_HF_onregelmatig', 'SYA_Multivorm_PVCs', 'SYA_PVCs/min_high',
       'SYA_Pauze', 'SYA_R-op-T_PVCs', 'SYA_Run_PVCs_high', 'SYA_Ventr_ritme',
       'YA_ABPm_high', 'YA_ABPm_low', 'YA_ABPs_high', 'YA_ABPs_low',
       'YA_ABPs_low ', 'YA_ECG_afl_los', 'YA_HF_high',
       'YA_HF_low', 'YA_Pols_high', 'YA_RF_high', 'YA_RF_low',
       'YA_SpO2_gn_puls', 'YA_SpO2_high', 'YA_SpO2_low', 'YA_etCO2_high',
       'YA_etCO2_low']


alarms                = pd.read_csv(LOCATION_DATA+'2_alarms_clean.csv', low_memory=False)
alarms.loc[:, 'Time'] = pd.to_datetime(alarms.loc[:, 'Time'])
alarms.loc[:, 'MRN']  = alarms.loc[:, 'MRN'].astype(str)
alarms                = alarms.sort_values(['MRN', 'Time'])
alarms                = alarms.groupby(['MRN', pd.Grouper(key = 'Time', freq=FREQ)]).agg([np.nansum])

data                  = pd.read_csv(LOCATION_DATA+'2_signal_clean.csv', low_memory=False)
data.loc[:, 'Time']   = pd.to_datetime(data.loc[:, 'Time'])
data.loc[:, 'MRN']    = data.loc[:, 'MRN'].astype(str)
data                  = data.sort_values(['MRN', 'Time'])
data                  = data.groupby(['MRN', pd.Grouper(key = 'Time', freq=FREQ)]).agg([np.nanmean])

# THRESH = 0.003
# h = alarms.sum().sort_values(ascending=True)/alarms.sum().sum()
# alarms_to_keep = h[h>= THRESH].index
# alarms = alarms.loc[:, alarms_to_keep]

data.columns         = data.columns.droplevel(1)
data                 = data.drop(columns='Unnamed: 0')

alarms.columns       = alarms.columns.droplevel(1)

data                 = data.merge(alarms, on=['MRN', 'Time'], how='left')

data_no_tech_alarms = data #[data.loc[:, cols_to_check].sum(axis=1) == 0]
# data_no_tech_alarms = data_no_tech_alarms.drop(columns=cols_to_check)

data_no_tech_alarms.loc[:, al_cols] = data_no_tech_alarms.loc[:, al_cols].fillna(0)
data_no_tech_alarms.loc[:, sig_cols] = data_no_tech_alarms.loc[:, sig_cols].fillna(data_no_tech_alarms.loc[:, sig_cols].mean())

data_no_tech_alarms = data_no_tech_alarms.drop(columns='etCO2')

# to_drop = ['RA_Brady/P_low', 'RA_Tachy', 'RA_Tachy/P_high', 'SYA_Multivorm_PVCs', 'YA_SpO2_gn_puls']

# data_no_tech_alarms = data_no_tech_alarms.drop(columns=to_drop)

data_no_tech_alarms.to_csv('data/full_data_and_alarms_with_tech.csv')