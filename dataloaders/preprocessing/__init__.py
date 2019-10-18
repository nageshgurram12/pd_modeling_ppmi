# -*- coding: utf-8 -*-

import pandas as pd

from mypath import Path
from custom_symbols import SYMBOLS

UPDRS2_DROP_COLS = SYMBOLS.UPDRS2_DROP_COLS
UPDRS3_DROP_COLS = SYMBOLS.UPDRS3_DROP_COLS

PAT_COL = SYMBOLS.PAT_COL # Patient col in data files
EVENT_COL = SYMBOLS.EVENT_COL # Event id col in data files (sc, bl, ..)

# If cohorts are not passed from input, then take these for analysis
COHORTS = SYMBOLS.COHORTS

# Encode events as a sequence of numbers
ENCODING = SYMBOLS.ENCODING

# Do one hot encoding for these variables
CATEGORICAL_VARS = SYMBOLS.CATEGORICAL_VARS

TOTAL_VISITS = SYMBOLS.TOTAL_VISITS
        
# TODO: check whether correct or not ?
UPDRS3_NA_COLS_FILL = SYMBOLS.UPDRS3_NA_COLS_FILL

#Fill other columns with some arbitrary number not in data
FILL_NA = SYMBOLS.FILL_NA

# Ambulatory score features
AMBUL_FEATURES = SYMBOLS.AMBUL_FEATURES

#default missing val strategy
DEFAULT_MISSING_VAL_STRATEGY = SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE

'''
If Baseline event is not present and Screening event exists, then 
change set SC as BL event
'''
def replace_sc_with_bl(df):
    for patient_id in df[PAT_COL].unique():
        # Check if BL(baseline) is empty and SC(Screening) record is there
        is_bl_cond = df[(df[PAT_COL] == patient_id) & \
                           (df[EVENT_COL] == "BL")]
        is_sc_cond = df[(df[PAT_COL] == patient_id) & \
                           (df[EVENT_COL] == "SC")]
        if not is_sc_cond.empty:
            if not is_bl_cond.empty:
                df.drop(is_sc_cond.index, axis=0, inplace=True)
            else:
                df.loc[is_sc_cond.index, EVENT_COL] = "BL"

def is_valid(date):
    try:
        pd.to_datetime(date)
        return True
    except ValueError:
        return False

'''
Add binary feature for whether patient is receiving symptomatic therapy
'''
def set_treated_flag(df):
    df.loc[:, "IS_TREATED"] = 0
    for patient in df[PAT_COL].unique():
        if not df[(df[PAT_COL] == patient) & (df["EVENT_ID"] == "ST")].empty:
            st_date = df.loc[(df[PAT_COL] == patient) & \
                       (df["EVENT_ID"] == "ST"), "INFODT"].values[0]
            if is_valid(st_date):
                st_date = pd.to_datetime(st_date)
                treated_index = df[(df[PAT_COL] == patient) & \
                (pd.to_datetime(df["INFODT"], errors='coerce') >= st_date)].index
                df.loc[treated_index,"IS_TREATED"] = 1      

def preprocess_updrs3():
    updrs3_file = Path.get_path('updrs3')
    updrs3_df = pd.read_csv(updrs3_file)
    
    # do this before dropping as its dependent on INFODT
    set_treated_flag(updrs3_df)
    
    # Drop unnecessary cols
    updrs3_df.drop(columns=UPDRS3_DROP_COLS, inplace=True)
    
    # Merge sc into bl
    replace_sc_with_bl(updrs3_df)
    
    # Fill NA cols with pre-defined missing values
    updrs3_df.fillna(UPDRS3_NA_COLS_FILL, inplace=True)
    
    return updrs3_df

def preprocess_updrs2():
    updrs2_file = Path.get_path('updrs2')
    updrs2_df = pd.read_csv(updrs2_file)

    # Drop unnecessary cols
    updrs2_df.drop(columns=UPDRS2_DROP_COLS, inplace=True)
    
    # Merge sc into bl
    replace_sc_with_bl(updrs2_df)
    
    return updrs2_df
    

def preprocess_patient_status(cohorts):
    patient_file =  Path.get_path('patient_status')
    patient_df = pd.read_csv(patient_file)
    
    # Get patients who has cohort name
    #patient_df['ENROLL_CAT'].notnull()
    pat_coh = patient_df[SYMBOLS.ENROLL_COL].isin(cohorts) 
    return patient_df.loc[pat_coh, (PAT_COL, SYMBOLS.ENROLL_COL)]
    
# Generate UPDRS_I, UPDRS_II, and UPDRS_III
def generate_updrs_subsets(data, features):
    # set features
    new_features = SYMBOLS.DERIVABLE_TOTALS
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Sum UPDRS subsets
    # data.loc[:, "UPDRS_I"] = data.filter(regex="NP1.*").sum(axis=1)
    data.loc[:, SYMBOLS.TOTAL_UPDRS2] = data.filter(regex="NP2.*").sum(axis=1)
    data.loc[:, SYMBOLS.TOTAL_UPDRS3] = data.filter(regex="NP3.*").sum(axis=1)
    data.loc[:, SYMBOLS.TOTAL_UPDRS2_3] = data[SYMBOLS.TOTAL_UPDRS2] + \
                                        data[SYMBOLS.TOTAL_UPDRS3]
    #data.loc[:, "TOTAL"] = data["UPDRS_II"] + data["UPDRS_III"]

    # Return new data
    return data

'''
Generate ambulatary score
'''
def generate_ambul_score(data):
    data.loc[:, SYMBOLS.AMBUL_SCORE] = data.filter(items=SYMBOLS.AMBUL_FEATURES). \
                                        sum(axis=1)
    return data

'''
Handle missing val by supplied strategy
'''
def handle_missing_vals(data, kwargs):
    miss_val_strategy = DEFAULT_MISSING_VAL_STRATEGY
    if SYMBOLS.MISSING_VAL_STRATEGY in kwargs:
        miss_val_strategy = kwargs[SYMBOLS.MISSING_VAL_STRATEGY]
    
    # change the filename of generated data based on missing val strategy
    final_data_type = ''
    
    # if missing val strategy is to remove the missig events,
    # then add a new column for gap between events
    if miss_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE:
        pat_ids = data[SYMBOLS.PAT_COL].unique()
        drop_pat_ids = []
        for pat_id in pat_ids:
            patient_data = data.loc[ data[SYMBOLS.PAT_COL] == pat_id ]
            event_ids = patient_data[SYMBOLS.EVENT_COL]
            # if patient doesn't have atleast some events, skip 
            if len(event_ids) < SYMBOLS.MIN_EVENTS:
                drop_pat_ids.append(pat_id)
            
            # create a new column for gap between events/visits
            event_ids = event_ids.sort_values()
            event_gaps = event_ids.diff()
            event_gaps.iloc[0] = 0 # dont keep NA 
            data.loc[ data[SYMBOLS.PAT_COL] == pat_id, \
            SYMBOLS.EVENT_GAP_COL] = event_gaps
        
        final_data_type += '_na_removed'
        data.drop(data.loc[data[SYMBOLS.PAT_COL].isin(drop_pat_ids)].index, \
                           inplace=True)
        
    # Pad missing visits sequences with fill_na value
    elif miss_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.PAD:
        multi_index = pd.MultiIndex.from_product([data[PAT_COL].unique(), \
        range(0, TOTAL_VISITS)], names=[PAT_COL, EVENT_COL])
        data = data.set_index([PAT_COL, EVENT_COL]).reindex(multi_index).reset_index()
        final_data_type = '_padded'
        
    return (final_data_type, data)
        
    
def preprocess_data(**kwargs):
    final_data_file_name = 'final_'
    
    if 'cohorts' in kwargs:
        cohorts = kwargs['cohorts']
    else:
        cohorts = COHORTS
    final_data_file_name += "_".join(cohorts)
    
    data_to_consider = kwargs['data_to_consider']
    
    # Preprocess the datafiles and merge into one
    data = preprocess_patient_status(cohorts)
    
    if SYMBOLS.UPDRS2 in data_to_consider:
        updrs2 = preprocess_updrs2()
        merge_on = [PAT_COL]
        if EVENT_COL in data.columns:
            merge_on.append(EVENT_COL)
        data = data.merge(updrs2, on=merge_on, how="outer")
        final_data_file_name += "_updrs2"
    
    if SYMBOLS.UPDRS3 in data_to_consider:
        updrs3 = preprocess_updrs3()
        merge_on = [PAT_COL]
        if EVENT_COL in data.columns:
            merge_on.append(EVENT_COL)
        data = data.merge(updrs3, on=merge_on, how="outer")
        final_data_file_name += "_updrs3"
    
    # filter patients based on passed cohorts
    data = data[data[SYMBOLS.ENROLL_COL].isin(cohorts)]
    
    # Drop duplicates
    data.drop_duplicates(subset=[PAT_COL, EVENT_COL], keep="first", inplace=True)
    
    # Drop rows if there is no event or Unscheduled vists (U01, U02..)
    data = data[data[EVENT_COL].notnull() & \
                data[EVENT_COL].isin(ENCODING[EVENT_COL].keys())]
    
    #TODO: Remove ST rows ?
    data = data[data[EVENT_COL] != 'ST']
    
    # Encode events with successive numbers from 0
    # BL-0, V01-1 ..
    data.replace(ENCODING, inplace=True)
    data[EVENT_COL] = pd.to_numeric(data[EVENT_COL])
    
    # Add new columns for sum of scores for updrs2,3 and combined
    data = generate_updrs_subsets(data=data, features=[])
    
    #if 'pred_types' in kwargs and SYMBOLS.AMBUL_SCORE in kwargs['pred_types']:
    data = generate_ambul_score(data)
    
    # Fill NA with passed param value
    if 'fill_na' in kwargs:
        fill_na = kwargs['fill_na']
    else:
        fill_na = FILL_NA
        
    # Do one hot encoding for categorical vars
    cat_vars = [cat_var for cat_var in CATEGORICAL_VARS if cat_var in data.columns]
    data[cat_vars] = data[cat_vars].astype('category')
    data = pd.get_dummies(data, columns=cat_vars)
        
    # handle missing data
    datatype, data = handle_missing_vals(data, kwargs)
    final_data_file_name += datatype
    
    data = data.fillna(fill_na)
    
    data.to_csv( Path.get_preprocessed(final_data_file_name), index=False)

if __name__ == "__main__":
    args = {
            SYMBOLS.MISSING_VAL_STRATEGY : SYMBOLS.MISSING_VAL_STRATEGIES.PAD,
            "cohorts" : SYMBOLS.COHORTS,
            "data_to_consider" : [SYMBOLS.UPDRS2, SYMBOLS.UPDRS3]
            }
    preprocess_data(**args)