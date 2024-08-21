#%%
"""
2024-June-20
KceniaB 

Update: 

        
""" 

#%%
"""
#################### IMPORTS ####################
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
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
# one = ONE()
ROOT_DIR = Path("/mnt/h0/kb/data/external_drive")
one = ONE(cache_dir="/mnt/h0/kb/data/one")

#%%
"""
#################### path to store the photometry file ####################
""" 
dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

df2 = df1[0:5] 

def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
    eid = eids[0]
    return eid

def get_regions(rec): 
    regions = [f"Region{rec.region}G"] 
    return regions 

def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/") 
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    return df_nph  

#%%
# EXCLUDES = [66,166, 215, 294, 360, 361, 362] #1st loop 
#294 - AssertionError: drift is more than 100 ppm 
#360 - KeyError: "None of [Index(['Region2G'], dtype='object')] are in the [columns]"
#361 - KeyError: "None of [Index(['Region2G'], dtype='object')] are in the [columns]"
#362 - KeyError: "None of [Index(['Region2G'], dtype='object')] are in the [columns]" 
#66 nph file is 3G (in this loop?) 

# EXCLUDES = [2, 4, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 84, 95, 141, 147] #2nd loop 
#2 - ValueError: zero-size array to reduction operation minimum which has no identity
#4, 6 - AssertionError: drift is more than 100 ppm
#13, 14, 15, ..., 23, 29 - FileNotFoundError: [Errno 2] No such file or directory: '/mnt/h0/kb/data/external_drive/2022-08-04/raw_photometry0.csv'
#84 - ValueError: zero-size array to reduction operation minimum which has no identity
#95 - FileNotFoundError: [Errno 2] No such file or directory: '/mnt/h0/kb/data/external_drive/2022-11-22/raw_photometry1.csv'
#141, 147 - FileNotFoundError: [Errno 2] No such file or directory: '/mnt/h0/kb/data/external_drive/2022-12-24/raw_photometry3.csv'

# EXCLUDES = [] #3rd

EXCLUDES = [166, 367, 369, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 392, 453, 499, 504] #4th 
# 166, 367, 369 - AssertionError: drift is more than 100 ppm
# 376 (...) check the excluded ones, 453, 499 - FileNotFoundError: [Errno 2] No such file or directory: '/mnt/h0/kb/data/external_drive/2022-08-04/raw_photometry0.csv'

IMIN = 0 #to start from here when rerunning; from scratch: write 0 
for i,rec in df1.iterrows(): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    #get data info
    regions = get_regions(rec)
    #get behav
    eid_0 = get_eid(rec) 
    #get photometry 
    df_ph = get_nph(source_path=(f'/mnt/h0/kb/data/external_drive/{rec.date}/'), rec=rec)

    sl = SessionLoader(one=one, eid=eid_0) 
    sl.load_trials()
    df_trials = sl.trials #trials table
    tbpod = sl.trials['stimOnTrigger_times'].values #bpod TTL times

    """
    CHANGE INPUT AUTOMATICALLY 
    """
    iup = ibldsp.utils.rises(df_ph[f'Input{rec.nph_bnc}'].values) #idx nph TTL times 
    tph = (df_ph['Timestamp'].values[iup] + df_ph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
    if len(tph)/len(tbpod) < .9: 
        print("mismatch in sync, will try to add ITI duration to the sync")
        tbpod = np.sort(np.r_[
            df_trials['intervals_0'].values,
            df_trials['intervals_1'].values - 1,  # here is the trick
            df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
        )
        fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        if len(tph)/len(tbpod) > .9:
            print("still mismatch, maybe this is an old session")
            tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
            assert len(iph)/len(tbpod) > .9
            print("recovered from sync mismatch, continuing #2")
    assert abs(drift_ppm) < 100, "drift is more than 100 ppm"

    df_ph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_ph["Timestamp"]) 

    fcn_nph_to_bpod_times(df_ph["Timestamp"])

    df_ph["Timestamp"]


    # Assuming tph contains the timestamps in seconds
    tbpod_start = tbpod[0] - 30  # Start time, 100 seconds before the first tph value
    tbpod_end = tbpod[-1] + 30   # End time, 100 seconds after the last tph value

    # Select data within the specified time range
    selected_data = df_ph[
        (df_ph['bpod_frame_times'] >= tbpod_start) &
        (df_ph['bpod_frame_times'] <= tbpod_end)
    ]

    # Now, selected_data contains the rows of df_ph within the desired time range 
    selected_data 

    # Plotting the new filtered data 
    # plt.figure(figsize=(20, 10))
    # plt.plot(selected_data.bpod_frame_times, selected_data[regions],color = "#25a18e") 
    # xcoords = tbpod
    # for xc in zip(xcoords):
    #     plt.axvline(x=xc, color='blue',linewidth=0.3)
    # plt.title("Entire signal, raw data")
    # plt.legend(["GCaMP","isosbestic"],frameon=False)
    # sns.despine(left = False, bottom = False) 
    # # plt.axvline(x=init_idx) 
    # # plt.axvline(x=end_idx) 
    # plt.show(block=False)
    # plt.close()

    df_ph = selected_data

    #===========================================================================
    #      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
    #===========================================================================
    df_ph = df_ph.reset_index(drop=True)
    df_ph = kcenia.LedState_or_Flags(df_ph)

    """ 4.1.2 Check for LedState/previous Flags bugs """ 
    """ 4.1.2.1 Length """
    # Verify the length of the data of the 2 different LEDs
    df_470, df_415 = kcenia.verify_length(df_ph)
    """ 4.1.2.2 Verify if there are repeated flags """ 
    kcenia.verify_repetitions(df_ph["LedState"])
    """ 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
    # session_day=rec.date
    # plot_outliers(df_470,df_415,region,mouse,session_day) 

    df_ph_1 = df_ph

    # Remove rows with LedState 1 at both ends if present
    if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
        df_ph_1 = df_ph_1.iloc[1:]

    # Remove rows with LedState 2 at both ends if present
    if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
        df_ph_1 = df_ph_1.iloc[:-2]

    # Filter data for LedState 2 (470nm)
    df_470 = df_ph_1[df_ph_1['LedState'] == 2]

    # Filter data for LedState 1 (415nm)
    df_415 = df_ph_1[df_ph_1['LedState'] == 1]

    # Check if the lengths of df_470 and df_415 are equal
    assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

    # Plot the data
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
    plt.plot(df_415[regions], c='#803896', linewidth=0.5)
    session_info = one.eid2ref(sl.eid)
    plt.title("Cropped signal "+session_info.subject+' '+str(session_info.date))
    plt.legend(["GCaMP", "isosbestic"], frameon=False)
    sns.despine(left=False, bottom=False)
    plt.show(block=False)
    plt.close() 
    # Print counts
    print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

    df_ph = df_ph_1.reset_index(drop=True)  
    df_470 = df_ph[df_ph.LedState==2] 
    df_470 = df_470.reset_index(drop=True)
    df_415 = df_ph[df_ph.LedState==1] 
    df_415 = df_415.reset_index(drop=True) 
    #================================================
    """ 4.1.4 FRAME RATE """ 
    acq_FR = kcenia.find_FR(df_470["Timestamp"]) 

    raw_reference = df_415[regions] #isosbestic 
    raw_signal = df_470[regions] #GCaMP signal 
    raw_timestamps_bpod = df_470["bpod_frame_times"]
    raw_timestamps_nph_470 = df_470["Timestamp"]
    raw_timestamps_nph_415 = df_415["Timestamp"]
    raw_TTL_bpod = tbpod
    raw_TTL_nph = tph

    my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

    df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])

    """
    ADD SAVE THE DATA AT THIS POINT 
    """
    file_photometry = sl.session_path.joinpath("alf", regions[0], "raw_photometry.pqt")
    file_photometry.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_photometry)


    import shutil
    file_raw_photometry = ROOT_DIR.joinpath(rec.date, f'raw_photometry{rec.nph_file}.csv')
    file_raw_photometry_out = sl.session_path.joinpath("raw_photometry_data", "raw_photometry.csv")
    file_raw_photometry_out.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(file_raw_photometry, file_raw_photometry_out)




#%%

#%%

#%%
""" 
OLDER SESSIONS 

KB
20240820 
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
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
# one = ONE()
ROOT_DIR = Path("/mnt/h0/kb/data/external_drive")
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

def get_eid(rec): 
    eids = one.search(subject=rec.Subject, date=rec.date) 
    eid = eids[0]
    return eid

def get_regions(rec): 
    regions = [f"Region{rec.region}G"] 
    return regions 

def get_nph(rec): 
    df_nph = pd.read_csv(rec.photometryfile) 
    df_nphttl = pd.read_csv(rec.bncfile) 
    return df_nph, df_nphttl 

def get_ttl(df_DI0): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    elif 'Timestamp' in df_DI0.columns: 
        df_DI0["Timestamp"] = df_DI0["Timestamp"] #for the old ones #KB added 20082024
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    return tph 


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


EXCLUDES = [15, 35, 36, 41, 42, 43, 44, 45] #5th 
# 15, 36, 41, 42, 43, 44, 45 - AssertionError: Sync arrays are of different lengths 
#35 - IndexError: list index out of range

IMIN = 36 #to start from here when rerunning; from scratch: write 0 



for i,rec in df1.iterrows(): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    #get data info
    regions = get_regions(rec)
    #get behav
    eid_0 = get_eid(rec) 
    #get photometry 
    df_ph, df_nphttl = get_nph(rec)

    sl = SessionLoader(one=one, eid=eid_0) 
    sl.load_trials()
    df_trials = sl.trials #trials table
    tbpod = sl.trials['stimOnTrigger_times'].values #bpod TTL times

    tph = get_ttl(df_nphttl)    
    iup = tph
    # tph = (df_ph['Timestamp'].values[iup] + df_ph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
    tph = iup
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
    if len(tph)/len(tbpod) < .9: 
        print("mismatch in sync, will try to add ITI duration to the sync")
        tbpod = np.sort(np.r_[
            df_trials['intervals_0'].values,
            df_trials['intervals_1'].values - 1,  # here is the trick
            df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
        )
        fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        if len(tph)/len(tbpod) > .9:
            print("still mismatch, maybe this is an old session")
            tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
            assert len(iph)/len(tbpod) > .9
            print("recovered from sync mismatch, continuing #2")
    assert abs(drift_ppm) < 100, "drift is more than 100 ppm"

    df_ph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_ph["Timestamp"]) 

    fcn_nph_to_bpod_times(df_ph["Timestamp"])

    df_ph["Timestamp"]


    # Assuming tph contains the timestamps in seconds
    tbpod_start = tbpod[0] - 30  # Start time, 100 seconds before the first tph value
    tbpod_end = tbpod[-1] + 30   # End time, 100 seconds after the last tph value

    # Select data within the specified time range
    selected_data = df_ph[
        (df_ph['bpod_frame_times'] >= tbpod_start) &
        (df_ph['bpod_frame_times'] <= tbpod_end)
    ]

    # Now, selected_data contains the rows of df_ph within the desired time range 
    selected_data 

    df_ph = selected_data

    #===========================================================================
    #      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
    #===========================================================================
    df_ph = df_ph.reset_index(drop=True)
    df_ph = kcenia.LedState_or_Flags(df_ph)

    """ 4.1.2 Check for LedState/previous Flags bugs """ 
    """ 4.1.2.1 Length """
    # Verify the length of the data of the 2 different LEDs
    df_470, df_415 = kcenia.verify_length(df_ph)
    """ 4.1.2.2 Verify if there are repeated flags """ 
    kcenia.verify_repetitions(df_ph["LedState"])
    """ 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
    df_ph_1 = df_ph

    # Remove rows with LedState 1 at both ends if present
    if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
        df_ph_1 = df_ph_1.iloc[1:]

    # Remove rows with LedState 2 at both ends if present
    if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
        df_ph_1 = df_ph_1.iloc[:-2]

    # Filter data for LedState 2 (470nm)
    df_470 = df_ph_1[df_ph_1['LedState'] == 2]

    # Filter data for LedState 1 (415nm)
    df_415 = df_ph_1[df_ph_1['LedState'] == 1]

    # Check if the lengths of df_470 and df_415 are equal
    assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

    # Plot the data
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
    plt.plot(df_415[regions], c='#803896', linewidth=0.5)
    session_info = one.eid2ref(sl.eid)
    plt.title("Cropped signal "+session_info.subject+' '+str(session_info.date))
    plt.legend(["GCaMP", "isosbestic"], frameon=False)
    sns.despine(left=False, bottom=False)
    plt.show(block=False)
    plt.close() 
    # Print counts
    print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

    df_ph = df_ph_1.reset_index(drop=True)  
    df_470 = df_ph[df_ph.LedState==2] 
    df_470 = df_470.reset_index(drop=True)
    df_415 = df_ph[df_ph.LedState==1] 
    df_415 = df_415.reset_index(drop=True) 
    #================================================
    """ 4.1.4 FRAME RATE """ 
    acq_FR = kcenia.find_FR(df_470["Timestamp"]) 

    raw_reference = df_415[regions] #isosbestic 
    raw_signal = df_470[regions] #GCaMP signal 
    raw_timestamps_bpod = df_470["bpod_frame_times"]
    raw_timestamps_nph_470 = df_470["Timestamp"]
    raw_timestamps_nph_415 = df_415["Timestamp"]
    raw_TTL_bpod = tbpod
    raw_TTL_nph = tph

    my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

    df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])

    """
    ADD SAVE THE DATA AT THIS POINT 
    """ 
    file_photometry = sl.session_path.joinpath("alf", regions[0], "raw_photometry.pqt")
    file_photometry.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_photometry)
    import shutil
    file_raw_photometry = (f'{rec.photometryfile}')
    file_raw_photometry_out = sl.session_path.joinpath("raw_photometry_data", "raw_photometry.csv")
    file_raw_photometry_out.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(file_raw_photometry, file_raw_photometry_out)




#%% 











































newpath = r'/mnt/h0/kb/data/one/mainenlab/Subjects/' 
new_path = 'test'
if not os.path.exists(newpath+new_path):
    os.makedirs(newpath+new_path) 

subject = 
date = 
session = 
region = "Region"+region_number+"G"
f'/mnt/h0/kb/data/one/mainenlab/Subjects/{subject}/{date}/{session}/alf/{region}/' 
#save the raw_photometry.pqt 

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

for i,rec in df1.iterrows():
    regions = kcenia.get_regions(rec)
    eid, df_trials = kcenia.get_eid(rec)
    print(i, rec, eid, df_trials)





#use fewer sessions first, to test the code 
df2 = df1[0:3]

#%%
""" PHOTOMETRY """ 
for i,rec in df2.iterrows(): 











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