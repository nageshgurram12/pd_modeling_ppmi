# -*- coding: utf-8 -*-

import pandas as pd
#import numpy as np
import os

data_path = os.path.dirname(__file__) + '/../../raw-data/'


FILENAMES = {
        'patient_status' : data_path + 'Patient_Status.csv',
        'updrs2' : data_path + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
        'updrs3' : data_path + 'MDS_UPDRS_Part_III.csv'
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

ENCODING = {
        EVENT_COL : {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, \
        "V05": 5, "V06": 6, "V07": 7, "V08": 8, "V09": 9, "V10": 10, \
        "V11": 11, "V12": 12, "V13": 13,"V14": 14, "V15" : 15, "V16" : 16,
        "ST": -1}
        }
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
    updrs3_file = FILENAMES['updrs3']
    updrs3_df = pd.read_csv(updrs3_file)
    
    # do this before dropping as its dependent on INFODT
    set_treated_flag(updrs3_df)
    
    # Drop unnecessary cols
    updrs3_df.drop(columns=UPDRS2_DROP_COLS, inplace=True)
    
    # Merge sc into bl
    replace_sc_with_bl(updrs3_df)
    
    return updrs3_df

def preprocess_updrs2():
    updrs2_file = FILENAMES['updrs2']
    updrs2_df = pd.read_csv(updrs2_file)

    # Drop unnecessary cols
    updrs2_df.drop(columns=UPDRS2_DROP_COLS, inplace=True)
    
    # Merge sc into bl
    replace_sc_with_bl(updrs2_df)
    
    return updrs2_df
    

def preprocess_patient_status(cohorts):
    patient_file = FILENAMES['patient_status']
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
    data.loc[:, "TOTAL"] = data["UPDRS_I"] + data["UPDRS_II"] + data["UPDRS_III"]

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
    data.to_csv('final_data.csv', index=False)
    
    # filter patients based on passed cohorts
    data.drop([~data['ENROLL_CAT'].isin(cohorts)], inplace=True)
    
    # Drop duplicates
    data.drop_duplicates(subset=[PAT_COL, EVENT_COL], keep="first", inplace=True)
    
    # Encode events with successive numbers from 0
    # BL-0, V01-1 ..
    data.replace(ENCODING, inplace=True)
    
    # Add new columns for sum of scores for updrs2,3
    data = generate_updrs_subsets(data=data, features=[])

if __name__ == "__main__":
    preprocess_data()