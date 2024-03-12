import pandas as pd
import numpy as np
import warnings
import re
import sys
sys.path.append('..')

# Parameters

LOCATION_DATA   = '../data/'
# VARS_SIG        = ['HF', 'RF', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'etCO2']
HOURS_MIN       = 6
VARS_MIN        = 4
s               = 30

# SIGNAL
print('Preprocess signal...')

signal                         = pd.read_csv(LOCATION_DATA+'1_all_signals.csv', low_memory=False, index_col=0)

signal.columns                 = ['MRN', 'Time', 'ParameterName', 'ParameterValue']
signal                         = signal.iloc[2:, :]

signal.loc[:, 'Time']          = pd.to_datetime(signal.loc[:, 'Time'],  format='%Y-%m-%d %H:%M:%S', errors='coerce') #infer_datetime_format=True)
signal = signal[~signal['Time'].isna()]

print(signal)

signal.loc[:, 'MRN']           = signal.loc[:, 'MRN'].astype(str)
signal.loc[:, 'ParameterName'] = signal.loc[:, 'ParameterName'].str.replace('NiBD', 'ABP')
signal                         = signal.drop_duplicates(['MRN', 'Time', 'ParameterName'])

signal_pivoted             = signal.pivot(index=['MRN', 'Time'], columns='ParameterName', values='ParameterValue').reset_index()

del signal

# Filter patients with at lest "HOURS_MIN" hours of recording
count_hours                   = signal_pivoted.groupby('MRN').count().iloc[:, 0]*s/60/60
pats_to_keep                  = count_hours[count_hours >= HOURS_MIN].index
signal_pivoted                = signal_pivoted[signal_pivoted.MRN.isin(pats_to_keep)]
    
# Keep rows with at least "VARS_MIN" columns present
signal_pivoted                = signal_pivoted[signal_pivoted.
                                               loc[:, ['ABPd', 'ABPm', 'ABPs', 'HF', 'RF', 'SpO2', 'etCO2']].
                                               isna().sum(axis=1) <= (7-VARS_MIN)]

# Add binary features
signal_pivoted['etCO2_bin']   = np.where(~signal_pivoted['etCO2'].isna(), 1, 0).tolist()

# store patients population
pop_id                        = signal_pivoted.MRN.unique()

pd.DataFrame(pop_id).to_csv(LOCATION_DATA+'pop_id.csv')
signal_pivoted.to_csv(LOCATION_DATA+'2_signal_clean.csv')

del signal_pivoted


# ALARMS
print('Preprocess alarm...')

alarms                         = pd.read_csv(LOCATION_DATA+'1_all_alarms.csv', low_memory=False, index_col=0)
pop_id                         = np.asarray(pd.read_csv(LOCATION_DATA+'pop_id.csv', index_col=0).values).flatten().astype(str)

# filter by population
alarms                         = alarms[alarms.MRN.astype(str).isin(pop_id)]

# improve alarm  descr.
alarms.loc[:, 'ParameterText']  = alarms.loc[:, 'ParameterText'].str.replace("ShortYellowAlarm", "SYA")
alarms.loc[:, 'ParameterText']  = alarms.loc[:, 'ParameterText'].str.replace("YellowAlarm ", "YA")
alarms.loc[:, 'ParameterText']  = alarms.loc[:, 'ParameterText'].str.replace("RedAlarm ", "RA")

alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("NiBD", "ABP")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("hoog", "high")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("laag", "low")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("*", "", regex=False)
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("!", "")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("<", "low") 
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace(">", "high")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace(r'\d', "", regex=True)

alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("           ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("          ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("         ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("        ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("       ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("      ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("     ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("    ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("   ", " ")
alarms.loc[:, 'ParameterValue'] = alarms.loc[:, 'ParameterValue'].str.replace("  ", " ")

substrings = ['ABP geen puls', 'ECG afl. los',
 'Niet te leren ECG', 'Resp afl. los', 'SpO geen puls', 'SpO geen sensor',
 'SpO sensor los', ' ABP losgeraakt', ' ABPs low', ' Apneu',
 ' HF low', 'ABP Wijzig schl', 'ABP contr.nulling', 'ECG/arr-alrmn uit',
 'SpO lage PFI', 'SpO trageUpdate', 'SpO zoekend', ' ECG-afl. los', ' RF low',
 'ABP Geen transdcr', 'ABP buiten bereik', 'ABP artefact',
 ' SpO low', ' VTach', ' HF high',
 'ABP-meting misl.',  'SpO intrfrentie', ' Desaturatie', ' AFIB',
 'Som ECG alrmn uit',
 'SpO zwak signaal', ' RF high', ' Pauze', ' SpO gn puls',
 'ABP onderbroken', ' Extreme tachy', ' Extreme brady',
 ' asystolie', 'ABP defect', 'SpO grillig' ,
 ' Vent fibr/tach', ' ABPs high',
 'ABP slecht signaal', ' Ventr.ritme', 
 ' Tachy (pols)', ' Pols high', ' Desat low ', 'ABPs low ', 'ABPs high ',
  'SpO low ', 'ABPs - low ', ' HF low ', 'RF high ',
 'xBrady low ', 'xTachy high', ' HF high ', 'etCO low ',
 'etCO high ', 'RF low ', ' Apneu : ', 'SpO sens.defect',
 'V Afl. los', 'EcgOutput defect',
 ' Brady (pols)',
 'ECG contr kabel', ' etCO high', ' etCO low',
 'SpO onbk.sensor', 'Brady/P low ',
 'HF high ', 'ABPm high ', 'ABPm low ', ' ABPm high', ' ABPm low', ' SpO high',
 'Tachy/P high', 'HF low ', 'SpO ctrl sensor', 'ABPm - low ',
 'ABP contr. mancht' , ' HF onregelmatig', ' Einde onreg. HF' ,
 'ABP manch.ovrdruk', 'Ctrlr ECG-kabel', 'SpO high ', 'xTachy high ',
 'Tachy/P high ', 'Contr.ECG bron', 'SpO defect']

alarms = alarms[alarms['ParameterValue'].isin(substrings)]

# Format variables
alarms.loc[:, 'ParameterName'] = alarms.loc[:, 'ParameterText'] + '_' + alarms.loc[:, 'ParameterValue']
alarms                         = alarms.loc[:, ['MRN', 'measurementime', 'ParameterName', 'count']]
alarms.columns                 = ['MRN', 'Time', 'ParameterName', 'ParameterValue']

alarms.loc[:, 'Time']          = pd.to_datetime(alarms.loc[:, 'Time'])
alarms.loc[:, 'MRN']           = alarms.loc[:, 'MRN'].astype(str)
alarms                         = alarms.drop_duplicates(['MRN', 'Time', 'ParameterName'])

alarms_pivoted                 = alarms.pivot(index=['MRN', 'Time'], columns='ParameterName', values='ParameterValue')

del alarms

pd.DataFrame(alarms_pivoted.columns).to_csv('cols_alarms.txt')

new_cols = ['HI_ABP_Geen_transdcr', 'HI_ABP_buiten_bereik', 'HI_ABP_contr_mancht',
            'HI_ABP_defect', 'HI_ABP_geen_puls', 'HI_ABP_onderbroken',  'HI_ABP_slecht_signaal',
            'HI_ABP_meting_misl.', 'HI_Ctrlr_ECG_kabel', 'HI_ECG_afl_los', 'HI_ECG_contr_kabel',
            'HI_EcgOutput_defect', 'HI_Niet_te_leren_ECG', 'HI_Resp_afl_los', 'HI_SpO2_defect',
            'HI_SpO2_geen_puls', 'HI_SpO2_geen_sensor',' HI_SpO2_grillig','HI_SpO2_intrfrentie','HI_SpO2_onbk_sensor','HI_SpO2_sens_defect',
            'HI_SpO2_sensor_los','HI_V_Afl_los','RA_ABP_losgeraakt','RA_ABPm_high','RA_ABPm_low','RA_ABPs_high','RA_ABPs_low',
            'RA_Apneu','RA_Apneu','RA_Brady','RA_Desaturatie' ,'RA_Desaturatie','RA_Extreme_brady','RA_Extreme_tachy',
            'RA_Tachy','RA_ VTach','RA_Vent_fibr/tach','RA_asystolie','RA_ABPm_low','RA_ABPm_high',
            'RA_ABPm_low','RA_ABPs_low','RA_ABPs_high' ,'RA_ABPs_low','RA_Brady/P_low' ,'RA_Tachy/P_high','RA_Tachy/P_high',
            'RA_xBrady_low' ,'RA_xTachy_high','RA_xTachy_high','SYA_AFIB','SYA_Einde onreg_HF',
            'SYA_HF_high','SYA_HF_high' ,'SYA_HF_low','SYA_HF_low' ,'SYA_HF_onregelmatig',
            'SYA_Pauze','SYA_Ventr_ritme',
            'SEI_ABP_manch_ovrdruk','SI_ABP_Wijzig_schl','SI_ABP_artefact','SI_ABP_contr_nulling','SI_ECG_contr_kabel',
            'SI_ECG/arr_alrmn_uit','SI_Som_ECG_alrmn_uit','SI_SpO2_ctrl_sensor','SI_SpO2_lage_PFI','SI_SpO2_trageUpdate',
            'SI_SpO2_zoekend','SI_SpO2_zwak_signaal','YA_ABPm_high','YA_ABPm_low','YA_ABPs_high',
            'YA_ABPs_low','YA_ECG_afl_los','YA_HF_high','YA_HF_low','YA_Pols_high',
            'YA_RF_high','YA_RF_low','YA_SpO2_gn_puls','YA_SpO2_high','YA_SpO2_low',
            'YA_etCO2_high','YA_etCO2_low','YA_ABPm_low' ,'YA_ABPm_high' ,'YA_ABPm_low',
            'YA_ABPs_low ','YA_ABPs_high' ,'YA_ABPs_low' ,'YA_HF_high',
            'YA_HF_low' ,'YA_RF_high' ,'YA_RF_low' ,'YA_SpO2_high' ,'YA_SpO2_low',
            'YA_etCO2_high' ,'YA_etCO2_low']

alarms_pivoted.columns = new_cols
alarms_pivoted = alarms_pivoted.groupby(axis=1, level=0).agg(np.nansum)

alarms_pivoted.to_csv(LOCATION_DATA+'2_alarms_clean.csv')

print('End :)')