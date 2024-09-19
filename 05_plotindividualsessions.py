"""
Code created in order to plot each session from the complete mouse list
Excel file must have the following columns: 

"""

#%% 
"""
IMPORTS AND FUNCTIONS 
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.colors as mcolors
from matplotlib.dates import date2num
from brainbox.io.one import SessionLoader
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#functions 
def get_eid(mouse,date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    try:
        # Try to load the trials directly
        a = one.load_object(eid, 'trials')
        trials = a.to_df()
    except Exception as e:
        # If loading fails, use the alternative method
        print("Failed to load trials directly. Using alternative method...")
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{mouse}/{date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 


#%%
##############################################################
"""
LOAD DATA 
""" 
dtype = {'nph_bnc': int, 'region': int, 'NM': str, 'mouse': str} #nph_file not included, bcs there are rows with empty cells for this column, but should be int
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx', 'A4_2024', dtype=dtype)
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 
# %%
