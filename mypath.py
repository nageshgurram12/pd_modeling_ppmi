# -*- coding: utf-8 -*-
import os

class Path(object):
    @staticmethod
    def get_path(datatype, mode='r'):
        data_path = os.path.dirname(__file__) + '/raw-data/'
        
        FILENAMES = {
            'patient_status' : data_path + 'Patient_Status.csv',
            'updrs2' : data_path + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
            'updrs3' : data_path + 'MDS_UPDRS_Part_III.csv',
            'merged' : data_path + 'final_merged.csv',
            'merged_padded' : data_path + 'final_merged_padded.csv',
            'merged_na_removed' : data_path + 'final_merged_na_removed.csv',
            'merged_pad_mvg_avg' : data_path + 'final_merged_pad_mvg_avg.csv'
        }
        
        if datatype in FILENAMES:
            return FILENAMES[datatype]
        else:
            
            print('No file with type ' + datatype)
            raise ValueError 
            
    @staticmethod
    def get_preprocessed(filename):
        data_path = os.path.dirname(__file__) + '/raw-data/preprocessed/'
        
        file_path = data_path + filename + ".csv"
        if not os.path.exists(file_path):
            open(file_path, "w+")
        return file_path
            