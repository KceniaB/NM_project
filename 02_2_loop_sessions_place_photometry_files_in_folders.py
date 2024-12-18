#%%
"""
2024-June-20
KceniaB 

Update: 
    pick all new and old sessions from 2 different sheets in the excel and join them in 1 df
        
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
# import kcenia as kcenia #added 16092024, works, it seems? 
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

#%%
"""
#################### path to store the photometry file ####################
""" 
dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

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


# ######################### 2nd loop, for the older sessions KB 16092024 
# # Load the CSV file
# df_goodsessions = pd.read_csv('/mnt/h0/kb/Mice performance tables 100 2.csv')

# # Convert 'nph_file' to integers, keeping NaN values
# df_goodsessions['nph_file'] = pd.to_numeric(df_goodsessions['nph_file'], errors='coerce').astype('Int64')

# # Ensure the 'date' column is in the yyyy-mm-dd format
# df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d').dt.date

# df_goodsessions['Date'] = df_goodsessions.date
# df_goodsessions['Mouse'] = df_goodsessions.mouse

# df1 = df_goodsessions

##########################################################


#%%
EXCLUDES = [] 
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
    tbpod_start = tbpod[0] - 10  # Start time, 100 seconds before the first tph value
    tbpod_end = tbpod[-1] + 10   # End time, 100 seconds after the last tph value

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
20240822 
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
    eids = one.search(subject=rec.mouse, date=rec.date) 
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
# df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables .xlsx' , 'todelete',dtype=dtype) 
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (3).xlsx' , 'todelete',dtype=dtype) #M1,M2,M3,M4 
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (4).xlsx' , 'todelete',dtype=dtype) #M1,M2,M3,M4 

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
#df1.loc[df1['date'].isna(), 'date'] = df1.loc[df1['date'].isna(), 'bncfile'].apply(extract_date_from_bncfile)

# List of columns to drop
columns_to_drop = ["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7"] 
# columns_to_drop = ["Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"]  
# columns_to_drop = ["Unnamed: 12", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"]

""" M1 M2 M3 M4 """
df1['date'] = df1['bncfile'].str.extract(r'_(\d{1,2}[A-Za-z]{3}\d{4})/DI')
df1['date'] = pd.to_datetime(df1['date'], format='%d%b%Y').dt.strftime('%Y-%m-%d')

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
    "D2": "ZFM-03448", 
    "D3": "ZFM-03450", 
    "M1": "ZFM-03059",
    "M2": "ZFM-03062",
    "M3": "ZFM-03065",
    "M4": "ZFM-03061", 
    "ZFM-02128": "ZFM-02128"  
}

# OTHER MAPPING 
# Create the mapping for both NM and subject
nm_mapping = {
    "S5": "5-HT",
    "D5": "DA",
    "D6": "DA",
    "D4": "DA",
    "N1": "NE",
    "N2": "NE",
    "D1": "DA",
    "D2": "DA",
    "D3": "DA",
    "M1": "5-HT",
    "M2": "5-HT",
    "M3": "5-HT",
    "M4": "5-HT", 
    "ZFM-02128": "ZFM-02128"
} 

# Map the NM and subject to each mouse
df1['NM'] = df1['Mouse'].map(nm_mapping)
df1['Subject'] = df1['Mouse'].map(mapping) 


# Create the new 'Subject' column by mapping the 'Mouse' column using the dictionary
df1['Subject'] = df1['Mouse'].map(mapping) 
df1 = df1.rename(columns={"Mouse": "subject"})
df1 = df1.rename(columns={"Subject": "mouse"})
df1 = df1.rename(columns={"Patch cord": "region"})

# Save df1 to an Excel file
# df1.to_excel("df1_data.xlsx", index=False)


EXCLUDES = [15, 35, 36, 41, 42, 43, 44, 45] #5th #most sessions 21June2022-04August2022 
# 15, 36, 41, 42, 43, 44, 45 - AssertionError: Sync arrays are of different lengths 
#35 - IndexError: list index out of range 
EXCLUDES = [5,14,15,18, 68] #6th #07July2022 29July2022 
#68 - no region in the file... 
EXCLUDES = [2, 5, 6, 7, 10, 11, 12, 18, 19, 35, 38, 54, 64, 85, 86, 88, 89, 95, 100, 101, 105, 106, 110, 111, 120, 151, 153, 154, 155, 156, 179, 202, 203, 204, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 222, 223, 224]


#%%
EXCLUDES=[9,13,17,28,37,39,40,43,44,49,53,54,55,56,58,69,71,72,99,105,110,112,115,120,139] #7th 28082024
IMIN = 0 #to start from here when rerunning; from scratch: write 0 


#%%
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
"""
.########..#######.....########..########....###....########.....########.####.##.......########
....##....##.....##....##.....##.##.........##.##...##.....##....##........##..##.......##......
....##....##.....##....##.....##.##........##...##..##.....##....##........##..##.......##......
....##....##.....##....########..######...##.....##.##.....##....######....##..##.......######..
....##....##.....##....##...##...##.......#########.##.....##....##........##..##.......##......
....##....##.....##....##....##..##.......##.....##.##.....##....##........##..##.......##......
....##.....#######.....##.....##.########.##.....##.########.....##.......####.########.########
"""""" M1 etc, good old sessions """
""" TRANSFORMED INTO IPNB """
#%%
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
from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
import os
from brainbox.io.one import SessionLoader 
import glob 
from one.api import ONE #always after the imports 
# one = ONE() 
ROOT_DIR = Path("/mnt/h0/kb/data/external_drive")
one = ONE(cache_dir="/mnt/h0/kb/data/one") 



def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
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
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (3).xlsx' , 'todelete',dtype=dtype) #M1,M2,M3,M4 
# df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (3)_2.xlsx' , 'todelete',dtype=dtype) #M1,M2,M3,M4 

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
#df1.loc[df1['date'].isna(), 'date'] = df1.loc[df1['date'].isna(), 'bncfile'].apply(extract_date_from_bncfile)
columns_to_drop = ["Unnamed: 12", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"]

df1['date'] = df1['bncfile'].str.extract(r'_(\d{1,2}[A-Za-z]{3}\d{4})/DI')
df1['date'] = pd.to_datetime(df1['date'], format='%d%b%Y').dt.strftime('%Y-%m-%d')

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
    "D2": "ZFM-03448", 
    "D3": "ZFM-03450", 
    "M1": "ZFM-03059",
    "M2": "ZFM-03062",
    "M3": "ZFM-03065",
    "M4": "ZFM-03061", 
    "ZFM-02128": "ZFM-02128"  
}

# OTHER MAPPING 
# Create the mapping for both NM and subject
nm_mapping = {
    "S5": "5-HT",
    "D5": "DA",
    "D6": "DA",
    "D4": "DA",
    "N1": "NE",
    "N2": "NE",
    "D1": "DA",
    "D2": "DA",
    "D3": "DA",
    "M1": "5-HT",
    "M2": "5-HT",
    "M3": "5-HT",
    "M4": "5-HT", 
    "ZFM-02128": "ZFM-02128"
} 

# Map the NM and subject to each mouse
df1['NM'] = df1['Mouse'].map(nm_mapping)
df1['Subject'] = df1['Mouse'].map(mapping) 


# Create the new 'Subject' column by mapping the 'Mouse' column using the dictionary
df1['Subject'] = df1['Mouse'].map(mapping) 
df1 = df1.rename(columns={"Mouse": "subject"})
df1 = df1.rename(columns={"Subject": "mouse"})
df1 = df1.rename(columns={"Patch cord": "region"})

EXCLUDES = [35, 130, 150, 153, 154, 155, 156, 165, 170, 172, 177, 179, 183,188,202,203,204,206,207,208,210,211,212,214,215,216,  
                    218,219,220,222,223,224] #35 #130
IMIN = 0

#%%
for i, rec in df1.iterrows(): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    mouse = rec.mouse 
    date = rec.date
    regions = get_regions(rec) 
    region = regions[0] 
    eid = get_eid(rec) 
    NM = rec.NM

    sl = SessionLoader(one=one, eid=eid) 
    sl.load_trials()
    df_trials = sl.trials #trials table 
    df_trials['mouse'] = mouse
    df_trials['date'] = date
    df_trials['region'] = region
    df_trials["eid"] = eid 

    # Create `allContrasts`
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
    df_trials['allContrasts'] = new_col
    # Create `allSContrasts`
    df_trials['allSContrasts'] = df_trials['allContrasts']
    # Apply negative sign for `contrastRight` values (including 0)
    df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = (
        df_trials['allContrasts'] * -1
    )
    # Ensure `allSContrasts` is properly formatted as strings
    df_trials['allSContrasts'] = df_trials['allSContrasts'].apply(
        lambda x: f"{x:.4g}"  # Converts to a string with up to 4 significant digits
    )
    # # Move `allSContrasts` to the 3rd column position
    df_trials.insert(3, 'allSContrasts', df_trials.pop('allSContrasts')) 


    try: 
        try: 
            dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
            values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
            # values gives the block length 
            # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
            # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

            values_sum = np.cumsum(values) 
            assert isinstance(values, list)

            # Initialize a new column 'probL' with NaN values
            df_trials['probL'] = np.nan

            # Set the first block (first `values_sum[0]` rows) to 0.5
            df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 


            df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']

            previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 


            # Iterate over the blocks starting from values_sum[1]
            for i in range(1, len(values_sum)-1):
                print("i = ", i)
                start_idx = values_sum[i]
                end_idx = values_sum[i+1]-1
                print("start and end _idx = ", start_idx, end_idx)
                
                # Assign the block value based on the previous one
                if previous_value == 0.2:
                    current_value = 0.8
                else:
                    current_value = 0.2
                print("current value = ", current_value)


                # Set the 'probL' values for the current block
                df_trials.loc[start_idx:end_idx, 'probL'] = current_value
                
                # Update the previous_value for the next block
                previous_value = current_value

            # Handle any remaining rows after the last value_sum block
            if len(df_trials) > values_sum[-1]:
                df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

            # plt.plot(df_trials.probabilityLeft, alpha=0.5)
            # plt.plot(df_trials.probL, alpha=0.5)
            # plt.title(f'behavior_{subject}_{session_date}_{eid}')
            # plt.show() 
        except: 
            df_trials["probL"] = df_trials["probabilityLeft"] 
    except: 
        print(f"No prob Left, it is probabily a TCW session {mouse} {date} {region}")
        continue 

    try: 
        EVENT = "feedback_times"
        path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
        path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy'

        # Load psth_idx from file
        psth_idx = np.load(path) 

        psth_combined = psth_idx
        df_trials_combined = df_trials
        # PLOT heatmap and correct vs incorrect 
        psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)]
        psth_error = psth_combined[:,(df_trials_combined.feedbackType == -1)]
        # Calculate averages and SEM
        psth_good_avg = psth_good.mean(axis=1)
        sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
        psth_error_avg = psth_error.mean(axis=1)
        sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

        # Create the figure and gridspec
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        # Plot the heatmap and line plot for correct trials
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
        ax1.invert_yaxis()
        ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax1.set_title('Correct Trials')

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
        ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax2.set_ylabel('Average Value')
        ax2.set_xlabel('Time')

        # Plot the heatmap and line plot for incorrect trials
        ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
        sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
        ax3.invert_yaxis()
        ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax3.set_title('Incorrect Trials')

        ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
        ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
        ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
        ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax4.set_ylabel('Average Value')
        ax4.set_xlabel('Time')

        # MIGHT BE WRONG!!!!! 
        # ticks = np.linspace(0, len(psth_good_avg), 4)  # Assuming 91 points, set 4 tick marks
        # tick_labels = [-1, 0, 1, 2]    # Labels corresponding to time from -1 to 2 seconds

        fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
        plt.tight_layout() 
        plt.savefig(f'/mnt/h0/kb/data/plots/performance_old_20241210/A_jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}.png')
        plt.show() 
        print(f"PSTH FILE LOADED AND PLOTED {mouse} {date} {region}")
    except: 
        # Base path up to the point where 00X folders are located
        base_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/'
        # Construct the pattern to match any folder with 00X (e.g., 001, 002, etc.)
        pattern = f'{base_path}00*/alf/Region*G/raw_photometry.pqt'
        # Use glob to find matching files
        matching_files = glob.glob(pattern)
        # Check and read the file if it exists
        if matching_files:
            # Assuming you want to process the first matching file
            pqt_file = matching_files[0]
            print(f"Found .pqt file: {pqt_file}")
            
            # Read the parquet file into a DataFrame
            test = pd.read_parquet(pqt_file)
            
            # Extract parts of the path
            path_parts = pqt_file.split(os.sep)  # Split the path by the OS-specific separator
            
            mouse = path_parts[8]  # Get the mouse identifier (ZFM-03059)
            date = path_parts[9]   # Get the date (2021-10-15)
            region = os.path.basename(os.path.dirname(pqt_file))  # Get the Region*G folder name
            
            # Add the new columns to the DataFrame
            test['mouse'] = mouse
            test['date'] = date
            test['region'] = region
            test['eid'] = eid

            df_nph = test 


            time_diffs = df_nph["times"].diff().dropna()
            fs = 1 / time_diffs.median()
            # Process the calcium signal and add to df  
            # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
            # df_nph["calcium"] = nph_j 

            df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
            df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
            df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
            df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
            df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
            df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
            df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
            df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs) 



            """ SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
            EVENT = "feedback_times" 
            time_bef = -1
            time_aft = 2
            PERIEVENT_WINDOW = [time_bef,time_aft]
            SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

            array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
            event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
            idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
            """ create a column with the trial number in the nph df """
            df_nph["trial_number"] = 0 #create a new column for the trial_number 
            df_nph.loc[idx_event,"trial_number"]=1
            df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

            sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
            n_trials = df_trials.shape[0]

            psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

            event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

            event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

            psth_idx += event_idx

            print(f"PSTH CALCULATED {mouse} {date} {region}")


            psth_combined = psth_idx
            df_trials_combined = df_trials
            try: 
                # PLOT heatmap and correct vs incorrect 
                psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)]
                psth_error = psth_combined[:,(df_trials_combined.feedbackType == -1)]
                # Calculate averages and SEM
                psth_good_avg = psth_good.mean(axis=1)
                sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
                psth_error_avg = psth_error.mean(axis=1)
                sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

                # Create the time vector based on the PERIEVENT_WINDOW and sampling rate
                n_timepoints = psth_good.shape[0]
                time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

                # Find the index of time=0 in the time vector
                zero_idx = np.where(time_vector == 0)[0][0]


                # Create the figure and gridspec
                fig = plt.figure(figsize=(10, 12))
                gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

                # Plot the heatmap and line plot for correct trials
                ax1 = fig.add_subplot(gs[0, 0])
                sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
                ax1.invert_yaxis()
                ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
                ax1.set_title('Correct Trials')

                ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
                ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
                # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
                ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
                ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
                ax2.set_ylabel('Average Value')
                ax2.set_xlabel('Time')

                # Plot the heatmap and line plot for incorrect trials
                ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
                sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
                ax3.invert_yaxis()
                ax3.axvline(x=zero_idx, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
                ax3.set_title('Incorrect Trials')

                ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
                ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
                ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
                ax4.axvline(x=zero_idx, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
                ax4.set_ylabel('Average Value')
                ax4.set_xlabel('Time')

                ticks = np.linspace(0, len(time_vector)-1, num=5)
                tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], num=5), 2)
                ax1.set_xticks(ticks)
                ax1.set_xticklabels(tick_labels)

                fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
                plt.tight_layout()
                plt.savefig(f'/mnt/h0/kb/data/plots/performance_old_20241210/B_jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}.png')
                plt.show() 
                print(f"DONE {mouse} {date} {region}")
            except: 
                # PLOT heatmap and correct vs incorrect 
                psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)]
                # Calculate averages and SEM
                psth_good_avg = psth_good.mean(axis=1)
                sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])

                # Create the time vector based on the PERIEVENT_WINDOW and sampling rate
                n_timepoints = psth_good.shape[0]
                time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

                # Find the index of time=0 in the time vector
                zero_idx = np.where(time_vector == 0)[0][0]

                # Create the time vector based on the PERIEVENT_WINDOW and sampling rate
                n_timepoints = psth_good.shape[0]
                time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

                # Find the index of time=0 in the time vector
                zero_idx = np.where(time_vector == 0)[0][0]


                # Create the figure and gridspec
                fig = plt.figure(figsize=(10, 12))
                gs = fig.add_gridspec(2,1, height_ratios=[3, 1])

                zero_idx = np.where(time_vector == 0)[0][0]

                # Plot the heatmap and line plot for correct trials
                ax1 = fig.add_subplot(gs[0, 0])
                sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
                ax1.invert_yaxis()
                ax1.axvline(x=zero_idx, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
                ax1.set_title('Correct Trials')

                ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
                ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
                # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
                ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
                ax2.axvline(x=zero_idx, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
                ax2.set_ylabel('Average Value')
                ax2.set_xlabel('Time')

                ticks = np.linspace(0, len(time_vector)-1, num=5)
                tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], num=5), 2)
                ax1.set_xticks(ticks)
                ax1.set_xticklabels(tick_labels)

                fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
                plt.tight_layout() 
                plt.savefig(f'/mnt/h0/kb/data/plots/performance_old_20241210/C_jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}.png')
                plt.show() 
                print(f"DONE(not BCW?) {mouse} {date} {region}")


    else:
        print(f"No matching .pqt file found. {i} {mouse} {date}") 

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