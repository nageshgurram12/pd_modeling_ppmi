# -*- coding: utf-8 -*-

from collections import namedtuple

sym_dict = {
        "PAT_COL" : "PATNO",
        "EVENT_COL" : "EVENT_ID",
        
        "ENROLL_COL" : "ENROLL_CAT", 
        # Remove unncessary cols from UPDRS2 file
        # TODO: Add comments for why we're adding 
        "UPDRS2_DROP_COLS" : ['REC_ID',
                    'F_STATUS',
                    'INFODT',
                    'ORIG_ENTRY',
                    'LAST_UPDATE',
                    'QUERY',
                    'SITE_APRV',
                    'PAG_NAME'],
                              
        "UPDRS3_DROP_COLS" : ['REC_ID',
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
                    ],
                              
        # If cohorts are not passed from input, then take these for analysis
        "COHORTS" : ['PD', 'GENPD', 'REGPD'],
        
        # Encode events as a sequence of numbers
        "ENCODING" : {
            "EVENT_ID" : {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, \
            "V05": 5, "V06": 6, "V07": 7, "V08": 8, "V09": 9, "V10": 10, \
            "V11": 11, "V12": 12, "V13": 13,"V14": 14, "V15" : 15, "V16" : 16,
            "ST": -1}
        },
        
        # Do one hot encoding for these variables
        "CATEGORICAL_VARS" : ['NUPSOURC', 'DYSKPRES', 'DYSKIRAT', 'PD_MED_USE', 
                    'ON_OFF_DOSE', 'IS_TREATED','ENROLL_CAT'],
                              
        "TOTAL_VISITS" : 17,
        
        # TODO: check whether correct or not ?
        "UPDRS3_NA_COLS_FILL" : {
            'ANNUAL_TIME_BTW_DOSE_NUPDRS' : 0,
            'ON_OFF_DOSE' : 0,
            'PD_MED_USE' : 0
        },
        
        "FILL_NA" : -1,
        
        # Ambulatory score features
        "AMBUL_FEATURES" : ["NP2WALK", "NP2FREZ", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"],
        
        "AMBUL_SCORE" : "AMBUL_SCORE",
        
        "TOTAL_UPDRS2" : "TOTAL_UPDRS2",
        
        "TOTAL_UPDRS3" : "TOTAL_UPDRS3",
        
        "TOTAL_UPDRS2_3" : "TOTAL_UPDRS2_3",
        
        # Derivable totals
        "DERIVABLE_TOTALS" : ["TOTAL_UPDRS2", "TOTAL_UPDRS3", "TOTAL_UPDRS2_3"],
        
        # MISSING VAL REPLACING STRATEGIES
        "MISSING_VAL_STRATEGIES" : namedtuple("MISSING_VAL_STRATEGIES",
                                              ("REMOVE", "PAD")) \
                                    ("REMOVE_NA_EVENTS","PAD_WITH_NA"),
                                    
        "MISSING_VAL_STRATEGY" : "MISSING_VAL_STRATEGY",
        
        # If missing strategy is removing NA events, then patient should have atleast 
        # this many to be considered in samples
        "MIN_EVENTS" : 8,
        
        "EVENT_GAP_COL" : "EVENT_GAP",
        
        # prediction varaibles
        
        # data files we have
        "DATA_TYPES" : namedtuple("DATA_TYPES", 
                                  ("MERGED_PADDED", "MERGED_NA_REMOVED")) \
                                  ("merged_padded", "merged_na_removed")
                                  
        }

SYMBOLS = namedtuple("SYMBOLS", sym_dict.keys())(*sym_dict.values())