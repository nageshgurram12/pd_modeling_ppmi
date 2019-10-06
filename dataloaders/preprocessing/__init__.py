# -*- coding: utf-8 -*-

import pandas as pd
#import numpy as np
import os

from mypath import Path

data_path = os.path.dirname(__file__) + '/../../raw-data/'

FILENAMES = {
        'patient_status' : data_path + 'Patient_Status.csv',
        'updrs2' : data_path + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
        'updrs3' : data_path + 'MDS_UPDRS_Part_III.csv',
        'merged_data' : data_path + 'final_merged_data.csv'
        }

# Remove unncessary cols from UPDRS2 file
# TODO: Add comments for why we're adding 
UPDRS2_DROP_COLS = ['REC_ID',
                    'F_STATUS',
                    'INFODT',
                    'ORIG_ENTRY',
                    'LAST_UPDATE',
                    'QUERY',
                    'SITE_APRV',
                    'PAG_NAME']

UPDRS3_DROP_COLS = ['REC_ID',
                    'F_STATUS',
                    'INFODT',
                    'EXAMTM',
                    'ORIG_ENTRY',
                    'LAST_UPDATE',
                    'QUERY',
                    'SITE_APRV',
                    'PAG_NAME',
                    'CMEDTM', # TODO: these two?
                    'EXAMTM'
                    ]

PAT_COL = "PATNO" # Patient col in data files
EVENT_COL = "EVENT_ID" # Event id col in data files (sc, bl, ..)

# If cohorts are not passed from input, then take these for analysis
COHORTS = ['PD', 'GENPD', 'REGPD'] 

# Encode events as a sequence of numbers
ENCODING = {
        EVENT_COL : {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, \
        "V05": 5, "V06": 6, "V07": 7, "V08": 8, "V09": 9, "V10": 10, \
        "V11": 11, "V12": 12, "V13": 13,"V14": 14, "V15" : 15, "V16" : 16,
        "ST": -1}
        }

# Do one hot encoding for these variables
CATEGORICAL_VARS = ['NUPSOURC', 'DYSKPRES', 'DYSKIRAT', 'PD_MED_USE', 
                    'ON_OFF_DOSE', 'IS_TREATED','ENROLL_CAT']

LAST_EVENT_IX = 16
        
# TODO: check whether correct or not ?
UPDRS3_NA_COLS_FILL = {
        'ANNUAL_TIME_BTW_DOSE_NUPDRS' : 0,
        'ON_OFF_DOSE' : 0,
        'PD_MED_USE' : 0
        }

#Fill other columns with some arbitrary number not in data
FILL_NA = -1

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
    pat_coh = patient_df['ENROLL_CAT'].isin(cohorts) 
    return patient_df.loc[pat_coh, (PAT_COL, 'ENROLL_CAT')]
    
# Generate UPDRS_I, UPDRS_II, and UPDRS_III
def generate_updrs_subsets(data, features):
    # set features
    new_features = ["UPDRS_II", "UPDRS_III", "UPDRS_II_AND_III", "TOTAL"]
    for feature in new_features:
        if feature not in features:
            features.append(feature)

    # Sum UPDRS subsets
    # data.loc[:, "UPDRS_I"] = data.filter(regex="NP1.*").sum(axis=1)
    data.loc[:, "UPDRS_II"] = data.filter(regex="NP2.*").sum(axis=1)
    data.loc[:, "UPDRS_III"] = data.filter(regex="NP3.*").sum(axis=1)
    data.loc[:, "UPDRS_II_AND_III"] = data["UPDRS_II"] + data["UPDRS_III"]
    data.loc[:, "TOTAL"] = data["UPDRS_II"] + data["UPDRS_III"]

    # Return new data
    return data
    
def preprocess_data(**kwargs):
    if 'cohorts' in kwargs:
        cohorts = kwargs['cohorts']
    else:
        cohorts = COHORTS
    
    # Preprocess the datafiles
    patient_status = preprocess_patient_status(cohorts)
    updrs2 = preprocess_updrs2()
    updrs3 = preprocess_updrs3()
    
    # merge updrs2, 3 files into one
    data = updrs2.merge(updrs3, on=[PAT_COL, EVENT_COL], how="outer")
    data = data.merge(patient_status, on=PAT_COL, how="outer")
    
    # filter patients based on passed cohorts
    data = data[data['ENROLL_CAT'].isin(cohorts)]
    
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
    
    # Fill NA with passed param value
    if 'fill_na' in kwargs:
        fill_na = kwargs['fill_na']
    else:
        fill_na = FILL_NA
        
    # Do one hot encoding for categorical vars
    data[CATEGORICAL_VARS] = data[CATEGORICAL_VARS].astype('category')
    data = pd.get_dummies(data, columns=CATEGORICAL_VARS)
        
    # Pad missing visits sequences with fill_na value
    final_datatype = 'merged'
    if 'pad_missing_visits' in kwargs and kwargs['pad_missing_visits']:
        multi_index = pd.MultiIndex.from_product([data[PAT_COL].values, \
        range(0, LAST_EVENT_IX+1)], names=[PAT_COL, EVENT_COL])
        data = data.set_index([PAT_COL, EVENT_COL]).reindex(multi_index).reset_index()
        final_datatype = 'merged_and_padded'
    
    data = data.fillna(fill_na)
    
    data.to_csv( Path.get_path(final_datatype), index=False)

if __name__ == "__main__":
    preprocess_data(pad_missing_visits=True)