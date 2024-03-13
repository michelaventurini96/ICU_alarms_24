import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import time
from spmf import Spmf
sys.path.append('..')

def to_transaction_db(data, var_name, s):
    
    f = open(LOCATION_RESULTS+var_name+"_input_spmf.txt","w+")

    first_time_pat       = data.index.get_level_values(1)[0]
    current_t            = data.index.get_level_values(1)[0]
    current_pat          = data.index.get_level_values(0)[0]
    prev_pat             = data.index.get_level_values(0)[0]
    
    pats = []

    i = 0
    for index, row in data.iterrows():
        
        current_pat = index[0]
        current_t = index[1]
        
        if (current_pat != prev_pat):
            
            first_time_pat = current_t
            prev_pat       = current_pat
            
            pats.append(current_pat)
            f.write("-2 \n")
            
        elif ((current_t-first_time_pat).total_seconds()/60 == s) and (current_pat == prev_pat):
            
            first_time_pat = current_t
            prev_pat       = current_pat
            
            pats.append(current_pat)
            f.write("-2 \n")
            
        if ((current_t-first_time_pat).total_seconds()/60 > s) and (current_pat == prev_pat):
            # first_time_pat = current_t
            # hoffset = int(((current_t-first_time_pat).total_seconds()/(60*60))/(W/60))*int(W/60)
            # first_time_pat = first_time_pat + pd.DateOffset(hours=hoffset)
            moffset = int(((current_t-first_time_pat).total_seconds()/(60))/(s))*int(s)
            first_time_pat = first_time_pat + pd.DateOffset(minutes=moffset)
            prev_pat = current_pat
            pats.append(current_pat)
            f.write("-2 \n")
            
        else:
            if i != 0:
                f.write("-1 ") 
            
        prev_pat = current_pat
        
        for j in row.index:
            if row[j] != 0:
                    f.write(str(j) + " ")
        
        i+=1
                    
    pats.append(current_pat)
    f.write("-2")
    f.close()
    pd.DataFrame(np.asarray(pats)).to_csv(LOCATION_RESULTS+"_PATS.csv")
           
def run_spmf_rules(var_name, algo, arg, s, w, NO_BINS):
  
    input_file = LOCATION_RESULTS+var_name+"_input_spmf.txt"
    arguments = arg
    spmf = Spmf(algo, input_filename=input_file,
            output_filename = LOCATION_RESULTS+var_name+'_'+str(s)+'_'+str(w)+'_'+str(NO_BINS)+"_rules_output.txt", 
            spmf_bin_location_dir="/home/michelav/.local/lib/python3.8/site-packages/spmf/",
            arguments=arguments)
    spmf.run()    

def find_rules(formatted_data, dict_hierc, s, algo, arg,  w, NO_BINS):
    
    for k, l in tqdm(dict_hierc.items()):
        
        data_tmp = formatted_data.iloc[:, l]
        
        to_transaction_db(data_tmp, var_name=k, s=s)
        run_spmf_rules(var_name=k, algo=algo, arg=arg, s=s, w=w, NO_BINS=NO_BINS)

def get_rules(data_cluster, dict_hierc, s):
    
    st = time.time()
    find_rules(data_cluster, dict_hierc, s, 'RuleGrowth', [.01, .1, 3, 1])
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 's')
    
def get_k_nonredundant_rules(data_cluster, dict_hierc, s, k,  w, NO_BINS):
    
    st = time.time()
    find_rules(data_cluster, dict_hierc, s, 'TNS', [k, .1, 2],  w, NO_BINS) # K, MIN CONF, DELTA
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 's')

def end_to_end_rules(data, w, s, NO_BINS, USE_SIGNAL, k):
    
    data_segm = []
    for pat in data.MRN.unique():
        
        tmp = data[data.MRN == pat].groupby(['MRN', pd.Grouper(key='Time', freq=w)]).agg([np.nanmean])
        data_segm.append(tmp)
    
    data = pd.concat(data_segm)
    data.columns = data.columns.droplevel(1)
    
    data[data.loc[:, COLS_ALARMS]>0] = 1
    
    # data.to_csv(LOCATION_RESULTS+'data_to_descr.csv')
    vars_to_disc = ['ABPd', 'ABPm', 'ABPs', 'HF', 'RF', 'SpO2']

    if USE_SIGNAL:

        # SAX
        
        tmp          = data.loc[:, vars_to_disc]
        disc_data    = (tmp - tmp.mean())/tmp.std()
        
        for var in disc_data.columns:
            disc_data.loc[:, 'qvar_'+var] = pd.qcut(disc_data[var].values, q=NO_BINS, precision=3, duplicates='drop')
            
        disc_data_cat = disc_data.filter(regex='qvar_')
            
        for col in disc_data_cat.columns:
            disc_data_cat.loc[:, col] = pd.Categorical(disc_data_cat.loc[:, col])
            disc_data_cat.loc[:, col] = disc_data_cat.loc[:, col].cat.codes
            disc_data_cat.loc[:, col] = pd.Categorical(disc_data_cat.loc[:, col])

        disc_data_cat                = pd.get_dummies(disc_data_cat, dummy_na=False)
        cols_to_drop                 = [col for col in disc_data_cat.columns if '-1' in col]
        disc_data_cat                = disc_data_cat.drop(cols_to_drop, axis=1)
        disc_data_cat.index          = data.index

        disc_data_alarms = data.loc[:, COLS_ALARMS].join(disc_data_cat)
        # disc_data_alarms = disc_data_alarms.join(data.loc[:, 'etCO2_bin'])
        
    else:
        
        disc_data_alarms = data.drop(columns=vars_to_disc) # data.loc[:, COLS_ALARMS]
        disc_data_alarms = disc_data_alarms[disc_data_alarms.sum(axis=1) >0]
        data[data > 0] = 1

    # Change column names 
    NAMES_TO_IDXS                 = pd.DataFrame()
    NAMES_TO_IDXS.loc[:, 'name']  = disc_data_alarms.columns
    NAMES_TO_IDXS.loc[:, 'code']  = np.arange(0, disc_data_alarms.shape[1])
    disc_data_alarms.columns      = NAMES_TO_IDXS.code

    ALL = NAMES_TO_IDXS.code.values

    DICT_HIERC = {'ALL':ALL}
    
    NAMES_TO_IDXS.to_csv(LOCATION_RESULTS+'NAMES_TO_IDXS_rules_'+str(NO_BINS)+'_.csv')

    if k:
        get_k_nonredundant_rules(disc_data_alarms, DICT_HIERC, s, k, w, NO_BINS)
    else:
        get_rules(disc_data_alarms, DICT_HIERC, s)

    
    rules = pd.read_csv(LOCATION_RESULTS+'ALL_'+str(s)+'_'+str(w)+'_'+str(NO_BINS)+'_rules_output.txt', sep='#', header=None)
    rules.columns = ['Rule', 'Support', 'Confidence']
    rules.Support = rules.Support.str.replace('SUP: ', "").astype(float)
    rules.Confidence = rules.Confidence.str.replace('CONF: ', "").astype(float)
    rules = rules.sort_values(['Confidence'], ascending=False)
    
    if k:
        rules = rules.iloc[:k, :]
    
    else:
        rules = rules.iloc[:1000, :]
    
    
    return rules.Support.mean(), rules.Confidence.mean()    

def mark_surrounding_alarms(sub_df):
    # Initialize a column to mark rows to keep, default to False
    sub_df['Keep'] = False
    
    # Identify rows where an alarm happened
    alarm_indices = sub_df[(sub_df.loc[:, COLS_ALARMS].sum(axis=1) > 0)].index
    
    # For each alarm, mark the surrounding 5 minutes
    for idx in alarm_indices:
        alarm_time = sub_df.loc[idx, 'Time']
        before_time = alarm_time - pd.Timedelta(minutes=5)
        after_time = alarm_time + pd.Timedelta(minutes=5)
        
        # Mark rows within 5 minutes before and after the alarm
        sub_df.loc[(sub_df['Time'] >= before_time) & (sub_df['Time'] <= after_time), 'Keep'] = True
    
    return sub_df

if __name__ == "__main__":
    
    # Parameters - FIXED
    LOCATION_RESULTS = '../results/resk1000/'

    VARS            = ['HF', 'PVC', 'RF', 'SpO2', 'ABP']

    # COLS_ALARMS     = ['RA_ VTach', 'RA_Extreme_tachy', 'RA_xTachy_high', 'YA_etCO2_low',
    #                     'RA_Apneu', 'YA_RF_low', 'YA_HF_high', 'YA_HF_low', 'YA_RF_high',
    #                     'RA_ABPs_high', 'SYA_AFIB', 'RA_ABPs_low', 'YA_ABPs_high', 'SYA_PVCs/min_high', 'SYA_HF_high',
    #                     'YA_ABPs_low', 'YA_SpO2_low', 'SYA_HF_low', 'RA_Desaturatie']
    
    # COLS_ALARMS     = ['RA_ VTach', 'RA_ABP_losgeraakt', 'RA_ABPm_high',
    #    'RA_ABPm_low', 'RA_ABPs_high', 'RA_ABPs_low', 'RA_Apneu', 'RA_Brady',
    #    'RA_Brady/P_low', 'RA_Desaturatie', 'RA_Extreme_brady',
    #    'RA_Extreme_tachy', 'RA_Tachy', 'RA_Tachy/P_high', 'RA_Vent_fibr/tach',
    #    'RA_asystolie', 'RA_xBrady_low', 'RA_xTachy_high',
    #    'SYA_AFIB', 'SYA_Doublet_PVCs',
    #    'SYA_Einde onreg_HF', 'SYA_HF_high', 'SYA_HF_low',
    #    'SYA_HF_onregelmatig', 'SYA_Multivorm_PVCs', 'SYA_PVCs/min_high',
    #    'SYA_Pauze', 'SYA_R-op-T_PVCs', 'SYA_Run_PVCs_high', 'SYA_Ventr_ritme',
    #    'YA_ABPm_high', 'YA_ABPm_low', 'YA_ABPs_high', 'YA_ABPs_low',
    #    'YA_ABPs_low ', 'YA_Contr_ECG_bron', 'YA_ECG_afl_los', 'YA_HF_high',
    #    'YA_HF_low', 'YA_Pols_high', 'YA_RF_high', 'YA_RF_low',
    #    'YA_SpO2_gn_puls', 'YA_SpO2_high', 'YA_SpO2_low', 'YA_etCO2_high',
    #    'YA_etCO2_low']
    
    # COLS_ALARMS_tmp = ['MRN', 'Time', 'ABPd', 'ABPm', 'ABPs', 'HF', 'RF', 'SpO2',
    #                    'etCO2_bin', 'VTach', 'ABP_losgeraakt', 'ABPm_high', 'ABPm_low',
    #                'ABPs_high', 'ABPs_low', 'Apneu', 'Brady', 'Desaturatie',
    #                'Brady', 'Tachy', 'Vent_fibr/tach', 'asystolie',
    #                'Brady', 'Tachy', 'AFIB', 'Einde onreg_HF', 
    #                'HF_high', 'HF_low', 'HF_onregelmatig', 'Pauze', 
    #                'Ventr_ritme', 'ABPm_high', 'ABPm_low', 
    #                'ABPs_high', 'ABPs_low', 'ABPs_low', 'ECG_afl_los', 'HF_high', 'HF_low', 
    #                'Pols_high', 'RF_high', 'RF_low', 'SpO2_high', 'SpO2_low'] 
    
    COLS_ALARMS_tmp = ['MRN', 'Time', 'ABPd', 'ABPm', 'ABPs', 'HF', 'RF', 'SpO2', 'etCO2_bin',
       'VTach', 'ABP_losgeraakt', 'ABPm_high', 'ABPm_low',
       'ABPs_high', 'ABPs_low', 'Apneu', 'Brady', 'Brady/P_low',
       'Desaturatie', 'Extreme_brady', 'Extreme_tachy', 'Tachy',
       'Tachy/P_high', 'Vent_fibr/tach', 'asystolie', 'xBrady_low',
       'xTachy_high', 'AFIB', 'Einde onreg_HF', 'HF_high',
       'HF_low', 'HF_onregelmatig', 'Pauze', 'Ventr_ritme',
       'ABPm_high', 'ABPm_low', 'ABPs_high', 'ABPs_low',
       'ABPs_low ', 'ECG_afl_los', 'HF_high', 'HF_low',
       'Pols_high', 'RF_high', 'RF_low', 'SpO2_gn_puls',
       'SpO2_high', 'SpO2_low']     
    
    # COLS_ALARMS = ['etCO2_bin', 'RA_ VTach', 'RA_ABP_losgeraakt', 'RA_ABPm_high', 'RA_ABPm_low',
    #                'RA_ABPs_high', 'RA_ABPs_low', 'RA_Apneu', 'RA_Brady', 'RA_Desaturatie',
    #                'RA_Extreme_brady', 'RA_Extreme_tachy', 'RA_Vent_fibr/tach', 'RA_asystolie',
    #                'RA_xBrady_low', 'RA_xTachy_high', 'SYA_AFIB', 'SYA_Doublet_PVCs', 'SYA_Einde onreg_HF', 
    #                'SYA_HF_high', 'SYA_HF_low', 'SYA_HF_onregelmatig', 'SYA_PVCs/min_high', 'SYA_Pauze', 
    #                'SYA_R-op-T_PVCs', 'SYA_Run_PVCs_high', 'SYA_Ventr_ritme', 'YA_ABPm_high', 'YA_ABPm_low', 
    #                'YA_ABPs_high', 'YA_ABPs_low', 'YA_ECG_afl_los', 'YA_HF_high', 'YA_HF_low', 
    #                'YA_Pols_high', 'YA_RF_high', 'YA_RF_low', 'YA_SpO2_high', 'YA_SpO2_low', 'YA_etCO2_high', 'YA_etCO2_low']   
    
    # COLS_ALARMS = ['etCO2_bin', 'VTach', 'ABP_losgeraakt', 'ABPm_high', 'ABPm_low',
    #                'ABPs_high', 'ABPs_low', 'Apneu', 'Brady', 'Desaturatie',
    #                'Tachy', 'Vent_fibr/tach', 'asystolie',
    #                'AFIB', 'Einde onreg_HF', 
    #                'HF_high', 'HF_low', 'HF_onregelmatig', 'Pauze', 
    #                'Ventr_ritme', 
    #                'ECG_afl_los',  
    #                'Pols_high', 'RF_high', 'RF_low', 'SpO2_high', 
    #                'SpO2_low'] 
    
    COLS_ALARMS = ['ABP_losgeraakt', 'ABPm_high', 'ABPm_low',
       'ABPs_high', 'ABPs_low', 'ABPs_low ', 'AFIB', 'Apneu', 'Brady',
       'Brady/P_low', 'Desaturatie', 'ECG_afl_los', 'Einde onreg_HF',
       'Extreme_brady', 'Extreme_tachy', 'HF_high', 'HF_low',
       'HF_onregelmatig', 'Pauze', 'Pols_high', 'RF_high', 'RF_low',
        'SpO2_gn_puls', 'SpO2_high', 'SpO2_low', 'Tachy',
       'Tachy/P_high', 'VTach', 'Vent_fibr/tach', 'Ventr_ritme', 'asystolie',
       'etCO2_bin', 'xBrady_low', 'xTachy_high']
    
    # Parameters - TO OPTIMIZE
    W               = ['30S', '1Min', '3Min', '5Min'] # segmentation window
    S               = [20, 40, 60] # MINUTES, sequence length
    NO_BINS         = [3, 5, 10, 15] # alphabet 
    USE_SIGNAL      = True
    K               = 1000

    # # Load data ---ALREADY DONE
    # data            = pd.read_csv('../data/full_data_and_alarms_no_tech.csv', low_memory=False)
    # data.MRN        = data.MRN.astype(str)
    # data.Time       = pd.to_datetime(data.Time)
    # data            = data.sort_values(['MRN', 'Time'])
    
    # data.columns = COLS_ALARMS_tmp
    
    # newd = data.iloc[:, 2:].astype(float).groupby(data.iloc[:, 2:].columns, axis=1).sum()
    # newd.loc[:, 'MRN'] = data.MRN
    # newd.loc[:, 'Time'] = data.Time
    
    # data = newd
    
    # # data = data[data.loc[:, COLS_ALARMS].sum(axis=1) > 0]
    
    # data = pd.concat([mark_surrounding_alarms(group) for _, group in data.groupby('MRN')])
    # data = data[data['Keep']]
    # data = data.drop(columns=['Keep'])
    
    # print(data.shape)
    # print(len(data.MRN.unique()))
    
    # data.to_csv(LOCATION_RESULTS+'ready_data.csv') ---END ALREADY DONE
    
    data = pd.read_csv(LOCATION_RESULTS+'ready_data.csv', low_memory=False)
    data.loc[:, 'Time'] = pd.to_datetime(data.loc[:, 'Time'])
    
    data = data.sort_values(['MRN', 'Time'])
    
    print('Start sensitivity analysis')
    
    for b in NO_BINS:

        avg_sup = np.zeros((len(W), len(S)))
        avg_conf = np.zeros((len(W), len(S)))

        for i, w in enumerate(W):
            for j, s in enumerate(S):
                avg_sup[i, j], avg_conf[i, j] = end_to_end_rules(data, w, s, NO_BINS=b, USE_SIGNAL=USE_SIGNAL, k=K)
                
        avg_sup = pd.DataFrame(avg_sup)
        avg_sup.columns = S
        avg_sup.index   = W
        
        avg_conf = pd.DataFrame(avg_conf)
        avg_conf.columns = S
        avg_conf.index   = W
        
        avg_sup.to_csv(LOCATION_RESULTS+str(b)+'_avg_sup_w30-5_s20-60.csv')
        avg_conf.to_csv(LOCATION_RESULTS+str(b)+'_avg_conf_w30-5_s20-60.csv')
        
    print('End :)')