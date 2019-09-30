# -*- coding: utf-8 -*-

import pandas as pd
#import numpy as np
import os

data_path = os.path.dirname(__file__) + '/../../raw-data/'

filenames = {
        'patient_status' : data_path + 'Patient_Status.csv',
        'updrs2' : data_path + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
        'updrs3' : data_path + 'MDS_UPDRS_Part_IIIpat.csv'
        }

def preprocess_updrs3():
    pass

def preprocess_updrs2():
    pass

def preprocess_patient_status():
    patient_file = filenames['patient_status']
    patient_df = pd.read_csv(patient_file)
    
    # Get patients who has cohort name
    pat_co = patient_df['ENROLL_CAT'].notnull()
    return patient_df.loc[pat_co, ('PATNO', 'ENROLL_CAT')]
    
    
def preprocess_data():
    patient_status = preprocess_patient_status()
    patient_status
    

if __name__ == "__main__":
    preprocess_data()