"""
2024-March-15
KceniaB 

""" 

#%%
# imports and loading data
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ibldsp.utils
from functions_nm import load_trials 

#%%
mouse           = 'N5' 
date            = '2023-08-31' 
region          = 'Region4G' 
main_path       = '/home/kcenia/Documents/Photometry_results/' + date + '/' 
session_path    = main_path+'raw_photometry2.csv' 
session_path_behav = main_path + mouse + '/'
io_path         = main_path+'bonsai_DI12.csv' 
init_idx = 100 


""" PHOTOMETRY """
df_PhotometryData = pd.read_csv(session_path) 
df_ph = pd.read_csv(io_path)  # Index(['Timestamp', 'Value.Seconds', 'Value.Value'], dtype='object') 

""" BEHAVIOR """ 
# Alternative1
# df_trials = pd.read_parquet(main_path + mouse + '/alf/_ibl_trials.table.pqt') 

#%%
# Alternative2 
from one.api import ONE 
one = ONE(mode="remote") #new way to load the data KB 01092023
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from brainbox.behavior.training import compute_performance 
# %%
# Alternative2 loading the behavior data
eids = one.search(subject='ZFM-06275') 
len(eids)
eid = eids[23]
ref = one.eid2ref(eid)
print(ref)

# %%
