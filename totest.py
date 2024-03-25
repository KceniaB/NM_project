

#%%
import pandas as pd 
from one.api import ONE 
one = ONE(mode="remote") 
import matplotlib.pyplot as plt 
import seaborn as sns
from functions_nm import load_trials 


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
********** EXPLAINING df_nph **********
    LedState        - ignore anything which is not a 2 or 1; process the data in order to start with one of them and end with another; 1 is isosbestic and 2 is GCaMP 
    Input0, Input1  - whenever "1" is when the TTL was sent, but the Timestamp corresponds to the time when the LED turned on (so it is most likely that the TTL arrived in between the previous row and the "current" row) 
                    - use df_nphttl for this 
    RegionXG        - recorded "brightness" of the ROI

TAKE INTO ACCOUNT: 
LedState, GCaMP and isos, is interleaved - so we need to extract 2 df's from there 

********** EXPLAINING df_nphttl ********** 
    Value.Seconds       - used to align 
    Value.Value         - filter only for True (it's when the TTL was received) 

"""

# %% 
""" BEHAVIOR """
eids = one.search(subject=mouse, date=date) 
len(eids)
eid = eids[0]
ref = one.eid2ref(eid)
print(ref) 

# ~~~~~ issue ~~~~~
# compare the choice to the rest of the data, like firstMovement_times - one appears to happen while the other one does not, which is not correct 
# {'subject': 'ZFM-06171', 'date': datetime.date(2024, 1, 19), 'sequence': 1}

# %% 
# Load the behavior data 
trials = load_trials(eid) 

# Create an empty dictionary to store the transformed data
transformed_data = {}

# Iterate over keys
for key, value in trials.items():
    # If the key is 'intervals', handle it differently
    if key == 'intervals':
        # Create two new columns for intervals_0 and intervals_1
        transformed_data['intervals_0'] = value[:, 0]
        transformed_data['intervals_1'] = value[:, 1]
    else:
        # For other keys, just store the data as is
        transformed_data[key] = value

# Convert the transformed data into a DataFrame
df = pd.DataFrame(transformed_data)

# Display the DataFrame
print(df)

