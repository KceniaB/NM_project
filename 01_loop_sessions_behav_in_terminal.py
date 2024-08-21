#%%
"""
2024-May-06
KceniaB 

Update: 
    2024-June-20 
        added SessionLoader 
        one = ONE(directory) to save them there 
    2024-June-21
        changed neurodsp to ibldsp 
        
""" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from functions_nm import load_trials 
import iblphotometry.kcenia as kcenia
import ibldsp.utils
from pathlib import Path
# import iblphotometry.plots
# import iblphotometry.dsp
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal

import ibllib.plots
from one.api import ONE #always after the imports 
# one = ONE()
one = ONE(cache_dir="/mnt/h0/kb/data/one")

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype)
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str)
     
     
for i,rec in df1.iterrows():
    regions = kcenia.get_regions(rec)
    eid, df_trials = kcenia.get_eid(rec)
    print(i, rec, eid, df_trials)





#%%
"""
20240620 another way of having the trials data 
"""
mouse_list = ["ZFM-06271", "ZFM-06272", "ZFM-05245", "ZFM-05248", "ZFM-05235", "ZFM-05236", "ZFM-04392", "ZFM-04019", "ZFM-04022", 
"ZFM-04026", "ZFM-04533", "ZFM-04534", "ZFM-06305", "ZFM-06948", "ZFM-03059", "ZFM-03061", "ZFM-03065", "ZFM-03447", "ZFM-03448", "ZFM-06946"]
for a in mouse_list: 
    eids = one.search(subject=a) 
    for i in range(len(eids)): 
        eid = eids[i]
        ref = one.eid2ref(eid)
        print(eid)
        print(ref) 
        try: 
            sl = SessionLoader(one=one, eid=eid)            
            sl.load_trials()    
        except: 
            print("Failed to load trials directly. "+eid)




#%% 
""" 
new data (old sessions)
""" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from functions_nm import load_trials 
# import iblphotometry.kcenia as kcenia
import ibldsp.utils
from pathlib import Path
# import iblphotometry.plots
# import iblphotometry.dsp
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal

import ibllib.plots
from one.api import ONE #always after the imports 
# one = ONE()
one = ONE(cache_dir="/mnt/h0/kb/data/one")

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (1).xlsx' , 'todelete',dtype=dtype)

# Function to extract date from the bncfile column
def extract_date_from_bncfile(path):
    # Split the path by '/' and get the part where the date is located
    parts = path.split('/')
    for part in parts:
        # Check if the part matches the date format YYYY-MM-DD
        if len(part) == 10 and part[4] == '-' and part[7] == '-':
            return part
    return None

# Apply the function to the bncfile column and create the 'date' column
df1['date'] = df1['bncfile'].apply(extract_date_from_bncfile)

# List of columns to drop
columns_to_drop = ["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7"]

# Drop the columns
df1 = df1.drop(columns=columns_to_drop) 

# Create a mapping dictionary
mapping = {
    "S5": "ZFM-04392",
    "D5": "ZFM-04019",
    "D6": "ZFM-04022",
    "D4": "ZFM-04026",
    "N1": "ZFM-04533",
    "N2": "ZFM-04534",
    "D1": "ZFM-03447",
    "D2": "ZFM-03448"
}

# Create the new 'Subject' column by mapping the 'Mouse' column using the dictionary
df1['Subject'] = df1['Mouse'].map(mapping) 
df1 = df1.rename(columns={"Mouse": "mouse"})
df1 = df1.rename(columns={"Patch cord": "region"})


mouse_list = ["ZFM-04392","ZFM-04019","ZFM-04022","ZFM-04026","ZFM-04533","ZFM-04534","ZFM-03447","ZFM-03448"]
for a in mouse_list: 
    eids = one.search(subject=a) 
    for i in range(len(eids)): 
        eid = eids[i]
        ref = one.eid2ref(eid)
        print(eid)
        print(ref) 
        try: 
            sl = SessionLoader(one=one, eid=eid)            
            sl.load_trials()    
        except: 
            print("Failed to load trials directly. "+eid)