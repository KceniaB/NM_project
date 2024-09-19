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
##############################################################
"""
PROCESS DATA 
    1 - get mouse, date, region names
    2 - get eid and df_trials through get_eid function 
    3 - read the raw nph_nph 
            times already in bpod time, raw_isosbestic, raw_calcium 
    4 - add mouse info to df_nph and df_trials 
    5 - add allContrasts and allSContrasts
""" 
test_01 = df1
i=30
EVENT = "feedback_times"

""" 1. """
mouse = test_01.mouse[i] 
date = test_01.date[i]
region = str(f"Region{test_01.region[i]}G")

""" 2. """
eid, df_trials = get_eid(mouse,date)
print(f"{mouse} | {date} | {region} | {eid}") 

""" 3. """
try: 
    nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/001/alf/{region}/raw_photometry.pqt'
    df_nph = pd.read_parquet(nph_path)
except:
    try:
        nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/002/alf/{region}/raw_photometry.pqt'
        df_nph = pd.read_parquet(nph_path)
    except:
        try:
            nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/003/alf/{region}/raw_photometry.pqt'
            df_nph = pd.read_parquet(nph_path)
        except:
            print(f"Could not find raw_photometry.pqt in paths 001, 002, or 003 for mouse {mouse} on date {date}")
            df_nph = None  # Optionally set df_nph to None or handle it appropriately

""" 4. """
df_nph["mouse"] = mouse
df_nph["date"] = date
df_nph["region"] = region
df_nph["eid"] = eid 
# create trialNumber
df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
df_trials["mouse"] = mouse
df_trials["date"] = date
df_trials["region"] = region
df_trials["eid"] = eid 

# create allContrasts 
idx = 2
new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
# create allSContrasts 
df_trials['allSContrasts'] = df_trials['allContrasts']
df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

# create reactionTime
reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
df_trials["reactionTime"] = reactionTime 
responseTime = np.array((df_trials["feedback_times"])-(df_trials["stimOnTrigger_times"])) 
df_trials["responseTime"] = responseTime 

# df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 
session_start = df_trials.intervals_0.values[0] - 10  # Start time, 100 seconds before the first tph value
session_end = df_trials.intervals_1.values[-1] + 10   # End time, 100 seconds after the last tph value

# Select data within the specified time range
selected_data = df_nph[
    (df_nph['times'] >= session_start) &
    (df_nph['times'] <= session_end)
] 

df_nph = selected_data.reset_index(drop=True) 

time_diffs = df_nph["times"].diff().dropna()
fs = round(1 / time_diffs.median())
# Process the calcium signal and add to df  
# nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
# df_nph["calcium"] = nph_j
df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_timestamps = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps, event_timestamps) #check idx where they would be included, in a sorted way 

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = fs #not a constant: print(1/np.mean(np.diff(array_timestamps))) #sampling rate #acq_FR

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 

stimOnT_idx = np.searchsorted(array_timestamps, event_stimOnT) #check idx where they would be included, in a sorted way 

psth_idx += stimOnT_idx



# %%
