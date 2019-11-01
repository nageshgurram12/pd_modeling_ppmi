# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

from mypath import Path
from custom_symbols import SYMBOLS

UPDRS2_DROP_COLS = SYMBOLS.UPDRS2_DROP_COLS
UPDRS3_DROP_COLS = SYMBOLS.UPDRS3_DROP_COLS
PD_FEAT_DROP_COLS = SYMBOLS.PD_FEAT_DROP_COLS

PAT_COL = SYMBOLS.PAT_COL # Patient col in data files
EVENT_COL = SYMBOLS.EVENT_COL # Event id col in data files (sc, bl, ..)
ENROLL_COL = SYMBOLS.ENROLL_COL
ON_OFF_DOSE = SYMBOLS.ON_OFF_DOSE
EVENT_DATE_COL = SYMBOLS.EVENT_DATE_COL
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
DEFAULT_MISSING_VAL_STRATEGY = SYMBOLS.MISSING_VAL_STRATEGIES.DN

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

'''
These below methods preprocess the data from files
'''
def preprocess_updrs3(**kwargs):
    updrs3_file = Path.get_path('updrs3')
    updrs3_df = pd.read_csv(updrs3_file, dtype = SYMBOLS.UPDRS3_COL_DATATYPES);
    #updrs3_df.astype(dtype = SYMBOLS.UPDRS3_COL_DATATYPES, errors="ignore")
    
    # do this before dropping as its dependent on INFODT
    set_treated_flag(updrs3_df)

    if ON_OFF_DOSE in kwargs and kwargs[ON_OFF_DOSE] == "off":
        updrs3_df = updrs3_df.loc[(updrs3_df['PAG_NAME'] != 'NUPDRS3A') & \
                              (updrs3_df[ON_OFF_DOSE] != 2), :]
    # Drop unnecessary cols
    updrs3_df.drop(columns=UPDRS3_DROP_COLS, inplace=True)
    # rename col 
    updrs3_df.rename(columns={EVENT_DATE_COL: EVENT_DATE_COL+"_U3"}, inplace=True)
    # Merge sc into bl
    replace_sc_with_bl(updrs3_df)
    
    # Fill NA cols with pre-defined missing values
    updrs3_df.fillna(UPDRS3_NA_COLS_FILL, inplace=True)
    
    return updrs3_df

def preprocess_updrs2():
    updrs2_file = Path.get_path('updrs2')
    updrs2_df = pd.read_csv(updrs2_file, dtype = SYMBOLS.UPDRS2_COL_DATATYPES)
    #updrs2_df.astype(dtype = SYMBOLS.UPDRS2_COL_DATATYPES, errors="ignore")
    
    # Drop unnecessary cols
    updrs2_df.drop(columns=UPDRS2_DROP_COLS, inplace=True)
    updrs2_df.rename(columns={EVENT_DATE_COL: EVENT_DATE_COL+"_U2"}, inplace=True)
    
    # Merge sc into bl
    replace_sc_with_bl(updrs2_df)
    
    return updrs2_df  

def preprocess_patient_status(cohorts):
    patient_file =  Path.get_path('patient_status')
    patient_df = pd.read_csv(patient_file)
    
    # Get patients who is in passed cohorts
    #patient_df['ENROLL_CAT'].notnull()
    pat_coh = patient_df[ENROLL_COL].isin(cohorts) 
    patient_df = patient_df.loc[pat_coh, (PAT_COL, ENROLL_COL)]
    
    #create has pd column
    patient_df.loc[:, "HAS_PD"] = 0
    patient_df.loc[(patient_df[ENROLL_COL] == "PD") | \
            (patient_df[ENROLL_COL] == "GENPD") | \
        (patient_df[ENROLL_COL] == "REGPD"), "HAS_PD"] = 1
    
    return patient_df

def preprocess_pd_features():
    pd_features_file =  Path.get_path('pd_features')
    pd_features_df = pd.read_csv(pd_features_file, \
                                 dtype=SYMBOLS.PD_FEAT_COL_DATATYPES)
    #pd_features_df.astype(dtype = SYMBOLS.PD_FEAT_COL_DATATYPES, errors="ignore")
    
    pd_features_df.drop(columns=SYMBOLS.PD_FEAT_DROP_COLS, inplace=True)
    pd_features_df.rename(columns={EVENT_DATE_COL: EVENT_DATE_COL+"_PF"}, \
                          inplace=True)
    
    # merge SXMO, SXYEAR cols (Symptom month, year) into one
    SXMO = "SXMO"; SXYEAR = "SXYEAR"
    # TODO: replace missing values
    pd_features_df[SXMO] = pd_features_df[SXMO].fillna(1)
    pd_features_df[SXYEAR] = pd_features_df[SXYEAR].fillna(2010)
    pd_features_df.loc[:, SYMBOLS.FSYM_COL] = pd_features_df[[SXMO, SXYEAR]] \
            .apply(lambda x: "/".join(map(str,map(int,x))), axis=1)
                                
    pd_features_df.drop(columns=[SXMO, SXYEAR], inplace=True)
    # Merge sc into bl
    replace_sc_with_bl(pd_features_df)
    
    return pd_features_df

def preprocess_demographics():
    demograph_file = Path.get_path('demographics')
    demograph_df = pd.read_csv(demograph_file, dtype=SYMBOLS.DEMO_COL_DATATYPES)
    demograph_df.drop(columns=SYMBOLS.DEMOGRAPH_DROP_COLS, inplace=True)
    
    return demograph_df

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
    data.loc[:, SYMBOLS.AMBUL_SCORE] = data.filter(items=SYMBOLS.AMBUL_FEATURES)\
                                        .sum(axis=1)
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
    
    # do nothing
    if miss_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.DN:
        return final_data_type, data
    
    # if missing val strategy is to remove the missig events,
    # then add a new column for gap between events
    if miss_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE:
        pat_ids = data[SYMBOLS.PAT_COL].unique()
        drop_pat_ids = []
        for pat_id in pat_ids:
            patient_data = data.loc[ data[SYMBOLS.PAT_COL] == pat_id ]
            event_ids = patient_data[SYMBOLS.EVENT_COL]
            # if patient doesn't have atleast some events, skip 
            #if len(event_ids) < SYMBOLS.MIN_EVENTS:
            #    drop_pat_ids.append(pat_id)
            
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
        data = data.set_index([PAT_COL, EVENT_COL]).reindex(multi_index) \
                .reset_index()
        final_data_type = '_padded'
        
    return (final_data_type, data)
        
'''
Preprocess individual data files
'''
def preprocess_individual_datafiles(datafiles, \
                                    final_data_file_name, **kwargs):
    def merge_on(data):
        merge_on = [PAT_COL]
        if EVENT_COL in data.columns:
            merge_on.append(EVENT_COL)
        return merge_on
    
    if SYMBOLS.UPDRS2 in datafiles:
        data = preprocess_updrs2()
        #data = datamerge(updrs2, on=merge_on(data), how="outer")
        final_data_file_name += "_updrs2"
    
    if SYMBOLS.UPDRS3 in datafiles:
        updrs3 = preprocess_updrs3(**kwargs)
        data = data.merge(updrs3, on=merge_on(data), how="outer")
        final_data_file_name += "_updrs3"
        
    if SYMBOLS.PD_FEAT in datafiles:
        pd_features = preprocess_pd_features()
        data = data.merge(pd_features, on=merge_on(data), how="outer")
        final_data_file_name += "_pdfeat"
        
    if SYMBOLS.DEMOGRAPH in datafiles:
        demographics = preprocess_demographics()
        data = data.merge(demographics, on=[PAT_COL], how="outer")
        final_data_file_name += "_demo"
        
    if SYMBOLS.PATIENT_STATUS in datafiles:
        patient_status = preprocess_patient_status(kwargs['cohorts'])
        data = data.merge(patient_status, on=[PAT_COL], how="outer")
        
    return data, final_data_file_name

'''
Create features like age, time from BL etc..
'''
def create_time_features(data):
    # Drop rows with no date time
    #data = data[data[EVENT_DATE_COL].notnull()]
    
    # Create derived cols
    data.loc[:, SYMBOLS.AGE_COL] = -1
    data.loc[:, SYMBOLS.TIME_FROM_BL_COL] = -1
    data.loc[:, SYMBOLS.TIME_SINCE_DIAG_COL] = -1
    data.loc[:, SYMBOLS.TIME_SINCE_FSYM_COL] = -1
    
    # Convert dates to date times
    all_infodt_cols = list(filter(lambda col: col.startswith(EVENT_DATE_COL), \
                             data.columns))
    data[all_infodt_cols] = data[all_infodt_cols].fillna(0)
    for infodt_col in all_infodt_cols:
        data.loc[:, infodt_col] = \
                    pd.to_datetime(data[infodt_col])
    data.loc[:, SYMBOLS.BRDY_COL] = pd.to_datetime(data[SYMBOLS.BRDY_COL])
    data.loc[:, SYMBOLS.PDDIAG_COL] = pd.to_datetime(data[SYMBOLS.PDDIAG_COL])
    data.loc[:, SYMBOLS.FSYM_COL] = pd.to_datetime(data[SYMBOLS.FSYM_COL])
    
    def convert_to_years(x):
        return round((x / np.timedelta64(1, 'D')) / 365 , 2)
    
    '''
    Get any defined INFODT col date to cal other 
    derived features
    '''
    def get_infodt(pat_record):
        for infodt_col in all_infodt_cols:
            if pat_record[infodt_col] != 0:
                return pat_record[infodt_col]
            
    # Set time from baseline for each event 
    for index, pat_record in data.iterrows():
        pat_id = pat_record[PAT_COL]
        event_id = pat_record[EVENT_COL]
        # date of the event
        now_date =  pat_record[all_infodt_cols].values.max()
        # date of the BL event
        baseline_date = (data.loc[(data[PAT_COL] == pat_id) & \
                        (data[EVENT_COL] == 0), all_infodt_cols]).values.max()
        data.loc[(data[PAT_COL] == pat_id) & \
                 (data[EVENT_COL] == event_id), SYMBOLS.TIME_FROM_BL_COL] = \
                 round((now_date - baseline_date).days/365, 2);
    

    # Set age in years, years from diagnosis, and years from first symptom
    data.loc[:, SYMBOLS.AGE_COL] = \
    (data[all_infodt_cols].max(axis=1) - data[SYMBOLS.BRDY_COL]) \
    .apply(convert_to_years)
    
    #TODO: these are not getting correct values
    data.loc[data["HAS_PD"] == 1, SYMBOLS.TIME_SINCE_DIAG_COL] = \
        (data.loc[data["HAS_PD"] == 1, all_infodt_cols].max(axis=1) - \
        data.loc[data["HAS_PD"] == 1, SYMBOLS.PDDIAG_COL]).apply(convert_to_years)
    
    data.loc[data["HAS_PD"] == 1, SYMBOLS.TIME_SINCE_FSYM_COL] = \
        (data.loc[data["HAS_PD"] == 1, all_infodt_cols].max(axis=1) - \
         data.loc[data["HAS_PD"] == 1, SYMBOLS.FSYM_COL]).apply(convert_to_years)

    #TODO: drop the date cols?
    data.drop(columns=[SYMBOLS.BRDY_COL, SYMBOLS.PDDIAG_COL, \
                       SYMBOLS.FSYM_COL] + all_infodt_cols, inplace=True)
    # Return data
    return data

def preprocess_data(**kwargs):
    final_data_file_name = 'final_'
    
    if 'cohorts' in kwargs:
        cohorts = kwargs['cohorts']
    else:
        kwargs['cohorts'] = COHORTS
        
    final_data_file_name += "_".join(cohorts)
    
    data_to_consider = kwargs['data_to_consider']
    
    # Preprocess the datafiles and merge into one
    #data = preprocess_patient_status(cohorts)
    data, final_data_file_name =  preprocess_individual_datafiles(data_to_consider,\
                                  final_data_file_name, **kwargs)
    
    
    # filter patients based on passed cohorts
    data = data[data[ENROLL_COL].isin(cohorts)]
    
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
    
    # Create time feats like age, time from BL etc.
    data = create_time_features(data)
    
    # Add new columns for sum of scores for updrs2,3 and combined
    data = generate_updrs_subsets(data=data, features=[])
    
    #if 'pred_types' in kwargs and SYMBOLS.AMBUL_SCORE in kwargs['pred_types']:
    data = generate_ambul_score(data)
    
    # Do one hot encoding for categorical vars
    cat_vars = []
    for col in data.columns:
        if (data[col].dtype.name == 'category') or (col in CATEGORICAL_VARS):
            cat_vars.append(col)
    data[cat_vars] = data[cat_vars].astype('category')
    data = pd.get_dummies(data, columns=cat_vars)
    
    data = data.astype(SYMBOLS.INDEX_VAR_DTYPES)

    # Fill NA with passed param value
    if 'fill_na' in kwargs:
        fill_na = kwargs['fill_na']
    else:
        fill_na = FILL_NA
    
    # handle missing data
    datatype, data = handle_missing_vals(data, kwargs)
    final_data_file_name += datatype
    
    data = data.fillna(fill_na)
    
    data.to_csv(Path.get_preprocessed(final_data_file_name), index=False)

if __name__ == "__main__":
    args = {
            SYMBOLS.MISSING_VAL_STRATEGY : SYMBOLS.MISSING_VAL_STRATEGIES.DN,
            "cohorts" : SYMBOLS.COHORTS,
            ON_OFF_DOSE : "off",
            "data_to_consider" : [SYMBOLS.UPDRS2, SYMBOLS.UPDRS3, \
                                  SYMBOLS.PATIENT_STATUS, \
                                  SYMBOLS.PD_FEAT, SYMBOLS.DEMOGRAPH]
            }
    preprocess_data(**args)