# -*- coding: utf-8 -*-
import numpy as np

from collections import namedtuple

sym_dict = {
        "PAT_COL" : "PATNO",
        "EVENT_COL" : "EVENT_ID",
        "ENROLL_COL" : "ENROLL_CAT", 
        "EVENT_DATE_COL" : "INFODT",
        "BRDY_COL" : "BIRTHDT", # birthday
        "PDDIAG_COL" : "PDDXDT", # PD Diagnose date
        "FSYM_COL" : "SXDT", # FIRST SYMPTOM DATE
        
        "ON_OFF_DOSE" :  "ON_OFF_DOSE",
        # DERIVED FEATURES
        "AGE_COL" : "AGE",
        "TIME_FROM_BL_COL" : "TIME_FROM_BL",
        "TIME_SINCE_DIAG_COL" : "TIME_SINCE_DIAGNOSIS",
        "TIME_SINCE_FSYM_COL" : "TIME_SINCE_FIRST_SYMPTOM",
        
        # Remove unncessary cols from UPDRS2 file
        # TODO: Add comments for why we're adding 
        "UPDRS2_DROP_COLS" : ['REC_ID',
                    'F_STATUS',
                    'ORIG_ENTRY',
                    'LAST_UPDATE',
                    'QUERY',
                    'SITE_APRV',
                    'PAG_NAME'],
          
        #column datatypes
        "UPDRS2_COL_DATATYPES" : {"NP2SPCH":np.float64,"NP2SALV":np.float64,
                                  "NP2SWAL":np.float64,"NP2EAT":np.float64,
                                  "NP2DRES":np.float64,"NP2HYGN":np.float64,
                                  "NP2HWRT":np.float64,"NP2HOBB":np.float64,
                                  "NP2TURN":np.float64,"NP2TRMR":np.float64,
                                  "NP2RISE": np.float64},
                              
        "UPDRS3_DROP_COLS" : ['REC_ID',
                    'F_STATUS',
                    'EXAMTM',
                    'ORIG_ENTRY',
                    'LAST_UPDATE',
                    'QUERY',
                    'SITE_APRV',
                    'PAG_NAME',
                    'CMEDTM', # TODO: these two?
                    'EXAMTM'
                    ],

        "UPDRS3_COL_DATATYPES" : {
                "NP3SPCH":np.float64,"NP3FACXP":np.float64,"NP3RIGN":np.float64,
                                  "NP3RIGRU":np.float64,"NP3RIGLU":np.float64,
                                  "PN3RIGRL":np.float64,"NP3RIGLL":np.float64,
                                  "NP3FTAPR":np.float64,"NP3FTAPL":np.float64,
                                  "NP3HMOVR":np.float64,"NP3HMOVL":np.float64,
                                  "NP3PRSPR":np.float64,"NP3PRSPL":np.float64,
                                  "NP3TTAPR":np.float64,"NP3TTAPL":np.float64,
                                  "NP3LGAGR":np.float64,"NP3LGAGL":np.float64,
                                  "NP3RISNG":np.float64,"NP3GAIT":np.float64,
                                  "NP3FRZGT":np.float64,"NP3PSTBL":np.float64,
                                  "NP3POSTR":np.float64,"NP3BRADY":np.float64,
                                  "NP3PTRMR":np.float64,"NP3PTRML":np.float64,
                                  "NP3KTRMR":np.float64,"NP3KTRML":np.float64,
                                  "NP3RTARU":np.float64,"NP3RTALU":np.float64,
                                  "NP3RTARL":np.float64,"NP3RTALL":np.float64,
                                  "NP3RTALJ":np.float64,"NP3RTCON":np.float64,
                    "NHY":np.float64,"ANNUAL_TIME_BTW_DOSE_NUPDRS": np.float64,
                                  "DYSKPRES":"category","DYSKIRAT":"category",
                                  "ON_OFF_DOSE":"category","PD_MED_USE":"category",
                                  
                                  },
        
        # Drop some PD feature cols
        "PD_FEAT_DROP_COLS" : [
                "REC_ID",
                "F_STATUS",
                "PAG_NAME",
                'ORIG_ENTRY',
                'LAST_UPDATE',
                'QUERY',
                'SITE_APRV',
                ], 
                
        "PD_FEAT_COL_DATATYPES" : {
                "PDDXEST":'category',"DXTREMOR":'category',
                "DXRIGID":'category',"DXBRADY":'category',"DXPOSINS":'category',
                "DXOTHSX":'category',"DXOTHCM":'category',"DOMSIDE":"category",
                "SXYEAR":"int64", "SXMO":"int8"
                },
         
        # Drop demograph features
        "DEMOGRAPH_DROP_COLS": [
                "REC_ID",
                "F_STATUS",
                "PAG_NAME",
                "EVENT_ID", # its just CONSENT
                "SIGNCNST",
                "CONSNTDT",
                "PRJENRDT",
                "REFERRAL",
                "DECLINED",
                "RSNDEC",
                "EXCLUDED",
                "RSNEXC",
                "ORIG_ENTRY",
                "LAST_UPDATE",
                "QUERY",
                "SITE_APRV"
                ],
                
        "DEMO_COL_DATATYPES" : {
                "APPRDX":'category',"P3GRP":'category',"GENDER":'category',
                "HISPLAT":'category',"RAINDALS":'category',"RAASIAN":'category',
                "RABLACK":'category',"RAHAWOPI":'category',"RAWHITE":'category',
                "RANOS":'category',"RANOSCM":'category'},
                
        # data to consider  keys
        "UPDRS2" : "UPDRS2",
        
        "UPDRS3" : "UPDRS3",
        
        "PD_FEAT" : "PD_FEATURES",
        
        "DEMOGRAPH" : "DEMOGRAPHICS",
        
        # If cohorts are not passed from input, then take these for analysis
        "COHORTS" : ['PD'],
        
        # Encode events as a sequence of numbers
        "ENCODING" : {
            "EVENT_ID" : {"BL": 0, "V01": 1, "V02": 2, "V03": 3, "V04": 4, \
            "V05": 5, "V06": 6, "V07": 7, "V08": 8, "V09": 9, "V10": 10, \
            "V11": 11, "V12": 12, "V13": 13,"V14": 14, "V15" : 15, "V16" : 16,
            "ST": -1}
        },
        
        # Do one hot encoding for these variables
        "CATEGORICAL_VARS" : ['HAS_PD', 
                              'IS_TREATED','ENROLL_CAT'],
                              
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
                                ("REMOVE", "PAD", "MA")) \
                                ("REMOVE_NA_EVENTS","PAD_WITH_NA", "MOVING_AVG"),
                                    
        "MISSING_VAL_STRATEGY" : "missing_val_strategy",
        
        # Moving avg width
        "MVG_AVG_WIDTH" : 4,
        
        # If missing strategy is removing NA events, then patient should have atleast 
        # this many to be considered in samples
        "MIN_EVENTS" : 8,
        
        "EVENT_GAP_COL" : "EVENT_GAP",
        
        # prediction varaibles
        
        # data files we have
        "DATA_TYPES" : namedtuple("DATA_TYPES", 
                                  ("MERGED_PADDED", "MERGED_NA_REMOVED", \
                                   "MERGED_MA")) \
                                  ("merged_padded", "merged_na_removed", \
                                   "merged_pad_mvg_avg")
                                  
        }

SYMBOLS = namedtuple("SYMBOLS", sym_dict.keys())(*sym_dict.values())