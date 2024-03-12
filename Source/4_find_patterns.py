import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import time
from spmf import Spmf
sys.path.append('..')

def to_transaction_db(data, var_name, W):
    
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
            
        elif ((current_t-first_time_pat).total_seconds()/60 == W) and (current_pat == prev_pat):
            
            first_time_pat = current_t
            prev_pat       = current_pat
            
            pats.append(current_pat)
            f.write("-2 \n")
            
        if ((current_t-first_time_pat).total_seconds()/60 > W) and (current_pat == prev_pat):
            # first_time_pat = current_t
            # hoffset = int(((current_t-first_time_pat).total_seconds()/(60*60))/(W/60))*int(W/60)
            # first_time_pat = first_time_pat + pd.DateOffset(hours=hoffset)
            moffset = int(((current_t-first_time_pat).total_seconds()/(60))/(W))*int(W)
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
         
def run_spmf(var_name, algo, arg):
  
    input_file = LOCATION_RESULTS+var_name+"_input_spmf.txt"
    arguments = arg
    spmf = Spmf(algo, input_filename=input_file,
            output_filename = LOCATION_RESULTS+var_name+"_output.txt", 
            spmf_bin_location_dir="/home/michelav/.local/lib/python3.8/site-packages/spmf/",
            arguments=arguments)
    spmf.run()   
    
def find_patterns(formatted_data, dict_hierc, W, algo, arg):
    
    for k, l in tqdm(dict_hierc.items()):
        
        data_tmp = formatted_data.iloc[:, l]
        
        to_transaction_db(data_tmp, var_name=k, W=W)
        run_spmf(var_name=k, algo=algo, arg=arg)

def from_pattern_to_list(p): # used
    str_list = p.split('-1')
    str_list = [list(i.split(' ')) for i in str_list]

    str_list1 = []
    for l in str_list:
        l = [int(x) for x in l if x not in ['', ' ', '\n']]
        
        if len(l):
            str_list1.append(l)
        
    return str_list1

def find_index_pattern(var, pattern): # used
    
    f = open(LOCATION_RESULTS+var+"_input_spmf.txt","r")
    lines = f.readlines()

    index_pattern = []

    count = 0
    for line in lines:

        line = from_pattern_to_list(line)
        elems = set(np.concatenate(pattern).flat) 
        line = [list(set(l).intersection(elems)) for l in line]
        len_line = len(set(np.concatenate(line).flat))
        
        if len_line > 0:
            found = 0
            i = 0
            j = 0
            
            while (j < len(pattern)) & (i < len(line)) & (found < len(pattern)):
                
                if len(set(line[i]).intersection(set(pattern[j]))) == len(set(pattern[j])):
                    found +=1
                    j += 1
                
                    if found == len(pattern):
                        index_pattern.append(count)
                
                i += 1    
                
        count = count+1
        
        
    f.close()
    
    return index_pattern

def encode_pats_patterns(patterns, dict_hierc): # used
    
    index_p = []

    for pp in tqdm(patterns):
        pattern = from_pattern_to_list(pp)
        
        for k in dict_hierc.keys():
            tmp = find_index_pattern(k, pattern)
            
            if len(tmp):
                index_p.append(tmp)

            # else:
            #     index_p.append([])
    
    return index_p

def encode_pats_patterns_1(patterns, index): # used
    
    n_obs = 40000
    
    pats_pattern_encoding = np.zeros((n_obs, len(patterns)))

    for i in range(len(patterns)):
        try:
            if len(index[i]):
                for j in index[i]:
                    pats_pattern_encoding[j, i] = 1
        except Exception:
            pass
    
    return pats_pattern_encoding

# def filter_redundant_patterns(all_patterns): # used

#     all_patterns.loc[:, 'set_subp'] = pd.Series(dtype='int')

#     for i in range(all_patterns.shape[0]):
        
#         p1 = from_pattern_to_list(all_patterns.iloc[i, 0])
#         p1 = [item for sublist in p1 for item in sublist]

#         all_patterns.iloc[i, 3] = str(set(p1))
        
#     non_redundant_patts = all_patterns.drop_duplicates('set_subp', keep='first')

#     return non_redundant_patts

# def compute_cosine_sim_matrix(encoding_sequences):
    
#     n = encoding_sequences.shape[1]
    
#     cosine_sim_matrix = np.zeros((n, n))
    
#     for i in range(n):
#         for j in range(n):
#             cosine_sim_matrix[i, j] = distance.cosine(encoding_sequences.iloc[:, i].values, encoding_sequences.iloc[:, j].values)
    
#     return cosine_sim_matrix

def group_patterns(signals, c):
    
    patterns = []
    
    for s in signals:
        
        pattern_input = pd.read_csv(LOCATION_RESULTS+s+'_input_spmf.txt', sep=' -1', names=np.arange(c), header=None)
        
        tot = pattern_input.shape[0] # number of positive pats

        try: 
            pattern_output = pd.read_csv(LOCATION_RESULTS+s+'_output.txt', sep='#', header=None)
            pattern_output = pattern_output.sort_values(by=1, ascending=False)
            pattern_output.columns = ['rule', 'support']
            pattern_output.loc[:, 'support'] = pattern_output.support.str.extract('(\d+)').astype(int)

        except:
            pattern_output = pd.DataFrame(columns=['rule', 'support'])
      
        pattern_output.loc[:, 'coverage/rule_cond_support(%)']        = (pattern_output.loc[:, 'support']/tot)*100
        patterns.append(pattern_output)

    patterns = pd.concat(patterns, axis=0)
    patterns = patterns.drop_duplicates(['rule']).sort_values(by='support', ascending=False)
    
    return patterns

def extract_nonredundant_patterns(min_cov, dict_hierc, vars): # used

    # get all patterns
    all_patterns = group_patterns(vars, c=15)

    # filter based on coverage (in percentage)
    all_patterns = all_patterns[all_patterns['coverage/rule_cond_support(%)'] >= min_cov]

    # remove redundant patterns (filter co-occurring patterns if they only contains the same sub-patterns)
    non_redundant_patts = all_patterns # filter_redundant_patterns(all_patterns)

    # encode sequences in patterns
    index = encode_pats_patterns(non_redundant_patts.rule.values , dict_hierc)
    encoding_sequences = encode_pats_patterns_1(non_redundant_patts.rule.values, index)
    encoding_sequences = pd.DataFrame(encoding_sequences[encoding_sequences.sum(axis=1) > 0])
    
    encoding_sequences.columns = non_redundant_patts.rule.values
    
    return encoding_sequences, non_redundant_patts

def get_patterns_and_encoding(data_cluster, min_cov, dict_hierc, vars_grouped, W): #used
    
    st = time.time()
    find_patterns(data_cluster, dict_hierc, W, ALGOS[0], ARGSS[0])
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 's')
    
    pats_id = pd.read_csv(LOCATION_RESULTS+"_PATS.csv")
    
    encoded_pats, non_redundant_patts_cluster = extract_nonredundant_patterns(min_cov, dict_hierc, vars_grouped)
    
    return pats_id, encoded_pats, non_redundant_patts_cluster

################################################# 
################# FIND PATTERNS #################
#################################################

# Parameters
LOCATION_RESULTS = 'tmp/'
W               = 15 # MINUTES
NO_BINS         = 5
VARS            = ['HF', 'PVC', 'RF', 'SpO2', 'ABP']

COLS_ALARMS     = ['RA_ VTach', 'RA_Extreme_tachy', 'RA_xTachy_high', 'YA_etCO2_low',
                    'RA_Apneu', 'YA_RF_low', 'YA_HF_high', 'YA_HF_low', 'YA_RF_high',
                    'RA_ABPs_high', 'SYA_AFIB', 'RA_ABPs_low', 'YA_ABPs_high', 'SYA_PVCs/min_high', 'SYA_HF_high',
                    'YA_ABPs_low', 'YA_SpO2_low', 'SYA_HF_low', 'RA_Desaturatie']

# Find patterns 
ALGOS           = ['PrefixSpan'] 
MIN_COV_PERC    = .05
LEN_PATTS       = 3
ARGSS           = [[MIN_COV_PERC, LEN_PATTS]]

# Filter patterns
MIN_COV         = MIN_COV_PERC*100

# Load data 
data            = pd.read_csv('data/full_data_and_alarms_preproc.csv')
data.MRN        = data.MRN.astype(int)
data.Time       = pd.to_datetime(data.Time)
data            = data.sort_values(['MRN', 'Time'])
data            = data.set_index(['MRN', 'Time'])


# SAX
vars_to_disc = ['ABPd', 'ABPm', 'ABPs', 'HF', 'PVC', 'RF', 'SpO2']
tmp          = data.loc[:, vars_to_disc]
disc_data    = (tmp - tmp.min())/tmp.max()

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
disc_data_alarms = disc_data_alarms.join(data.loc[:, 'etCO2_bin'])

# Change column names 
NAMES_TO_IDXS                 = pd.DataFrame()
NAMES_TO_IDXS.loc[:, 'name']  = disc_data_alarms.columns
NAMES_TO_IDXS.loc[:, 'code']  = np.arange(0, disc_data_alarms.shape[1])
disc_data_alarms.columns                  = NAMES_TO_IDXS.code

# Create a dictionary of signals hierarchy
HF      = NAMES_TO_IDXS[ NAMES_TO_IDXS.name.str.contains("HF") | NAMES_TO_IDXS.name.str.contains("Tach") 
                        | NAMES_TO_IDXS.name.str.contains("tach") | NAMES_TO_IDXS.name.str.contains("AFIB")].code.values
RF      = NAMES_TO_IDXS[ NAMES_TO_IDXS.name.str.contains("RF")].code.values
PVC     = NAMES_TO_IDXS[ NAMES_TO_IDXS.name.str.contains("PVC")].code.values
SpO2    = NAMES_TO_IDXS[ NAMES_TO_IDXS.name.str.contains("SpO2") | NAMES_TO_IDXS.name.str.contains("etCO2") 
                        | NAMES_TO_IDXS.name.str.contains("Apneu") | NAMES_TO_IDXS.name.str.contains("Desaturatie")].code.values
ABP     = NAMES_TO_IDXS[ NAMES_TO_IDXS.name.str.contains("ABP")].code.values

DICT_HIERC = {'HF':HF, 'RF':RF, 'PVC':PVC, 'SpO2':SpO2, 'ABP': ABP}

pats_id, encoded_pats_all, non_redundant_patts_cluster_all = get_patterns_and_encoding(disc_data_alarms, MIN_COV, DICT_HIERC, VARS, W)

NAMES_TO_IDXS.to_csv('data/NAMES_TO_IDXS.csv')
non_redundant_patts_cluster_all.to_csv('data/non_redundant_patts.csv')
pats_id.to_csv('data/pats_id.csv')
encoded_pats_all.to_csv('data/patients_encoded.csv')