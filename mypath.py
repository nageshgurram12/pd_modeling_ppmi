# -*- coding: utf-8 -*-
import os

class Path(object):
    @staticmethod
    def get_path(datatype):
        data_path = os.path.dirname(__file__) + '/raw-data/'
        FILENAMES = {
            'patient_status' : data_path + 'Patient_Status.csv',
            'updrs2' : data_path + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
            'updrs3' : data_path + 'MDS_UPDRS_Part_III.csv',
            'merged' : data_path + 'final_merged.csv',
            'merged_padded' : data_path + 'final_merged_padded.csv',
            'merged_na_removed' : data_path + 'final_merged_na_removed.csv',
        }
        
        if datatype in FILENAMES:
            return FILENAMES[datatype]
        else:
            print('No file with type ' + datatype)
            raise ValueError 