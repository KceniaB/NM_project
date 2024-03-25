

#%%
import pandas as pd 
from one.api import ONE 
one = ONE(mode="remote") #new way to load the data KB 01092023

#%%
""" PHOTOMETRY """
date = "2024-01-19"
mouse = "ZFM-06275"
nphfile_number = "1"
bncfile_number = "1"
region = "Region6G"

source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv")

"""
EXPLAINING df_nph: 
    Timestamp       - used to align 
    LedState        - ignore anything which is not a 2 or 1; process the data in order to start with one of them and end with another; 1 is isosbestic and 2 is GCaMP(?) 
    Input0, Input1  - whenever "1" is when the TTL was sent, but the Timestamp corresponds to the time when the LED turned on (so it is most likely that the TTL arrived in between the previous row and the "current" row) 
                    - use df_nphttl for this 
    RegionXG        - recorded "brightness" of the ROI

TAKE INTO ACCOUNT: 
LedState, GCaMP and isos, is interleaved - so we need to extract 2 df's from there 
"""

# %% 
""" BEHAVIOR """
eids = one.search(subject=mouse, date=date) 
len(eids)
eid = eids[0]
ref = one.eid2ref(eid)
print(ref) 

# %% 

