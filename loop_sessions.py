#%%
"""
2024-April-11
KceniaB 

Update: 
    Apr17 
        optimized extract_data_info function 
        
""" 

import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
from one.api import ONE
one = ONE()
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm2 import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 

df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024') 


#%%
""" PHOTOMETRY """ 
df_test = df1[(df1.date == "2024-03-22") & (df1.Mouse == "ZFM-06275")]

mouse, date, nphfile_number, bncfile_number, region, region2, nm = extract_data_info(df = df_test)

#photometry 
df_nph, df_nphttl = get_nph(date=date, nphfile_number=nphfile_number, bncfile_number=bncfile_number)

#behavior 
eid = get_eid(mouse=mouse,date=date)


# %%
