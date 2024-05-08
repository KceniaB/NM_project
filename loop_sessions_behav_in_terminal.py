#%%
"""
2024-May-06
KceniaB 

Update: 
    
        
""" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from functions_nm import load_trials 
import iblphotometry.kcenia as kcenia
import neurodsp.utils
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal

import ibllib.plots
from one.api import ONE #always after the imports 
one = ONE()

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype)
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str)
     
     
for i,rec in df1.iterrows():
    regions = kcenia.get_regions(rec)
    eid, df_trials = kcenia.get_eid(rec)
    print(i, rec, eid, df_trials)
