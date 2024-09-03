#%%
"""
2024-April-11
KceniaB 

Update: 
    Apr17 
        optimized extract_data_info function 
    2024-June-20 
        added SessionLoader 
        one = ONE(directory) to save them there 
    2024-June-21
        changed neurodsp to ibldsp 
""" 

import ibldsp.utils 
import pandas as pd
from pathlib import Path
import iblphotometry.plots
# import iblphotometry.dsp 
from brainbox.io.one import SessionLoader
import scipy.signal
import ibllib.plots
from iblphotometry.kcenia import *
from one.api import ONE #always after the imports 
one = ONE(base_url="/mnt/h0/kb/data/one")
# one = ONE(base_url="/mnt/h0/kb/data/one", username='kcenia', password='top_secret') 


def get_regions(rec): 
    """ 
    extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
    """
    regions = [f"Region{rec.region}G"] 
    return regions

def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/")
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    df_nphttl = pd.read_csv(source_folder+f"bonsai_DI{rec.nph_bnc}{rec.nph_file}.csv") 
    return df_nph, df_nphttl 

def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    # session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/' 
    base_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{rec.mouse}/{rec.date}/' 
    session_path_pattern = f'{base_path}00*/'
    session_paths = glob.glob(session_path_pattern)
    if session_paths:
        session_path_behav = session_paths[0]  # or handle multiple matches as needed
    else:
        session_path_behav = None  # or handle the case where no matching path is found
    # file_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-12-30/001/alf/_ibl_trials.table.pqt' #KB commented 04Aug2024 
    file_path = session_path_behav+'alf/_ibl_trials.table.pqt' #KB added 04Aug2024 

    df = pd.read_parquet(file_path)

    df_alldata = extract_all(session_path_behav)
    table_data = df_alldata[0]['table']
    trials = pd.DataFrame(table_data) 
    return eid, trials 
    
def get_ttl(df_DI0, df_trials): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])
    return tph, tbpod 

def start_2_end_1(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=0, starting at flag=2, finishing at flag=1, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["LedState"][0] == 0: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if (array1["LedState"][0] != 2) or (array1["LedState"][0] != 1): 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][0] == 1: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][len(array1)-1] == 2: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 

def start_17_end_18(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=16, starting at flag=17, finishing at flag=18, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["Flags"][0] == 16: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][0] == 18: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][len(array1)-1] == 17: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 

def change_flags(df_with_flags): 
    df_with_flags = df_with_flags.reset_index(drop=True)
    if 'LedState' in df_with_flags.columns: 
        array1 = np.array(df_with_flags["LedState"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["LedState"] = array2
        return(df_with_flags) 
    else: 
        array1 = np.array(df_with_flags["Flags"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["Flags"] = array2
        return(df_with_flags) 

def LedState_or_Flags(df_PhotometryData): 
    if 'LedState' in df_PhotometryData.columns:                         #newversion 
        df_PhotometryData = start_2_end_1(df_PhotometryData)
        df_PhotometryData = df_PhotometryData.reset_index(drop=True)
        df_PhotometryData = (change_flags(df_PhotometryData))
    else:                                                               #oldversion
        df_PhotometryData = start_17_end_18(df_PhotometryData) 
        df_PhotometryData = df_PhotometryData.reset_index(drop=True) 
        df_PhotometryData = (change_flags(df_PhotometryData))
        df_PhotometryData["LedState"] = df_PhotometryData["Flags"]
    return df_PhotometryData

def verify_length(df_PhotometryData): 
    """
    Checking if the length is different
    x = df_470
    y = df_415
    """ 
    x = df_PhotometryData[df_PhotometryData.LedState==2]
    y = df_PhotometryData[df_PhotometryData.LedState==1] 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 
    print("470 = ",x.LedState.count()," 415 = ",y.LedState.count())
    return(x,y)

def verify_repetitions(x): 
    """
    Checking if there are repetitions in consecutive rows
    x = df_PhotometryData["Flags"]
    """ 
    for i in range(1,(len(x)-1)): 
        if x[i-1] == x[i]: 
            print("here: ", i) 

def find_FR(x): 
    """
    find the frame rate of acquisition
    x = df_470["Timestamp"]
    """
    acq_FR = round(1/np.mean(x.diff()))
    # check to make sure that it is 15/30/60! (- with a loop)
    if acq_FR == 30 or acq_FR == 60 or acq_FR == 120: 
        print("All good, the FR is: ", acq_FR)
    else: 
        print("CHECK FR!!!!!!!!!!!!!!!!!!!!") 
    return acq_FR 


dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

# """ TEST MOUSE """ 
# df_test=df1[df1.mouse=="ZFM-04022"].reset_index(drop=True)
# df_test = df_test[4:6].reset_index(drop=True)

#%%
""" PHOTOMETRY """ 
# df_test = df1[(df1.date == "2024-01-24") & (df1.mouse == "ZFM-06948")] 
# df_test = df1[(df1.date == "2024-03-22") & (df1.mouse == "ZFM-06948")]

for i,rec in df1.iterrows(): 
 
    regions = get_regions(rec)

    eid, df_trials = get_eid(rec.mouse, rec.date) #KB added 04Aug2024

    df_nph, df_nphttl = get_nph(f"/mnt/h0/kb/data/external_drive/{rec.date}/", rec) #KB added 04Aug2024

    tph, tbpod = get_ttl(df_DI0 = df_nphttl, df_trials = df_trials) 

    df_PhotometryData = df_nph 

    try:
        tbpod = np.sort(np.r_[
            df_trials['intervals_0'].values,
            df_trials['intervals_1'].values,
            df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
        )
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
    except AssertionError:
        print("mismatch in sync, will try to add ITI duration to the sync")
        try:
            tbpod = np.sort(np.r_[
                df_trials['intervals_0'].values,
                df_trials['intervals_1'].values - 1,  # here is the trick
                df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
            )
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
            assert len(iph)/len(tbpod) > .9
            print("recovered from sync mismatch, continuing")
        except AssertionError:
            print("mismatch, maybe this is an old session")
            tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
            assert len(iph)/len(tbpod) > .9
            print("recovered from sync mismatch, continuing #2")


    fig, axs = plt.subplots(2, 1)
    axs[0].plot(np.diff(tph))
    axs[0].plot(np.diff(tbpod))
    axs[0].legend(['ph', 'pbod'])
    print('max deviation:', np.max(np.abs(fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]) * 1e6), 'drift: ', drift_ppm, 'ppm')
    #fcn_nph_to_bpod_times  # apply this function to whatever photometry timestamps
    axs[1].plot(np.diff(fcn_nph_to_bpod_times(tph[iph])))
    axs[1].plot(np.diff(tbpod[ibpod]))
    axs[1].legend(['ph', 'pbod']) 
    # fig.savefig(f'/home/kceniabougrova/Documents/results_for_OW/Fig00_TTL_{rec.mouse}_{rec.date}_{rec.region}.png')

    # transform the nph TTL times into bpod times 
    nph_sync = fcn_nph_to_bpod_times(tph[iph]) 
    bpod_sync = tbpod[ibpod] #same bpod_sync = tbpod
    fig1, ax = plt.subplots()
    ax.set_box_aspect(1)
    plt.plot(nph_sync, bpod_sync) 
    plt.show(block=False)
    plt.close()

    df_PhotometryData["bpod_frame_times_feedback_times"] = fcn_nph_to_bpod_times(df_PhotometryData["Timestamp"]) 

    # Assuming nph_sync contains the timestamps in seconds
    nph_sync_start = nph_sync[0] - 30  # Start time, 100 seconds before the first nph_sync value
    nph_sync_end = nph_sync[-1] + 30   # End time, 100 seconds after the last nph_sync value

    # Select data within the specified time range
    selected_data = df_PhotometryData[
        (df_PhotometryData['bpod_frame_times_feedback_times'] >= nph_sync_start) &
        (df_PhotometryData['bpod_frame_times_feedback_times'] <= nph_sync_end)
    ]

    plt.figure(figsize=(20, 10))
    plt.plot(selected_data[regions], linewidth=1)
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(selected_data.bpod_frame_times_feedback_times, selected_data[regions],color = "#25a18e") 
    xcoords = nph_sync
    for xc in zip(xcoords):
        plt.axvline(x=xc, color='blue',linewidth=0.1, alpha=0.85)
    plt.title("Entire signal, raw data")
    plt.legend(["GCaMP","isosbestic"],frameon=False)
    sns.despine(left = False, bottom = False) 
    # plt.axvline(x=init_idx) 
    # plt.axvline(x=end_idx) 
    plt.show(block=False)
    plt.close()
    plt.show()

    df_PhotometryData = selected_data

    # Now, selected_data contains the rows of df_PhotometryData within the desired time range 
    selected_data 

    # Plotting the new filtered data 
    plt.figure(figsize=(20, 10))
    plt.plot(selected_data.bpod_frame_times_feedback_times, selected_data[regions],color = "#25a18e") 
    xcoords = nph_sync
    for xc in zip(xcoords):
        plt.axvline(x=xc, color='blue',linewidth=0.3)
    plt.title("Entire signal, raw data")
    plt.legend(["GCaMP","isosbestic"],frameon=False)
    sns.despine(left = False, bottom = False) 
    # plt.axvline(x=init_idx) 
    # plt.axvline(x=end_idx) 
    plt.show(block=False)
    plt.close()

    df_PhotometryData = selected_data

    #===========================================================================
    #      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
    #===========================================================================
    df_PhotometryData = df_PhotometryData.reset_index(drop=True)
    df_PhotometryData = LedState_or_Flags(df_PhotometryData)

    """ 4.1.2 Check for LedState/previous Flags bugs """ 
    """ 4.1.2.1 Length """
    # Verify the length of the data of the 2 different LEDs
    df_470, df_415 = verify_length(df_PhotometryData)
    """ 4.1.2.2 Verify if there are repeated flags """ 
    verify_repetitions(df_PhotometryData["LedState"])
    """ 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
    # session_day=rec.date
    # plot_outliers(df_470,df_415,region,mouse,session_day) 

    df_PhotometryData_1 = df_PhotometryData

    # Remove rows with LedState 1 at both ends if present
    if df_PhotometryData_1['LedState'].iloc[0] == 1 and df_PhotometryData_1['LedState'].iloc[-1] == 1:
        df_PhotometryData_1 = df_PhotometryData_1.iloc[1:]

    # Remove rows with LedState 2 at both ends if present
    if df_PhotometryData_1['LedState'].iloc[0] == 2 and df_PhotometryData_1['LedState'].iloc[-1] == 2:
        df_PhotometryData_1 = df_PhotometryData_1.iloc[:-2]

    # Filter data for LedState 2 (470nm)
    df_470 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 2]

    # Filter data for LedState 1 (415nm)
    df_415 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 1]

    # Check if the lengths of df_470 and df_415 are equal
    assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

    # Plot the data
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
    plt.plot(df_415[regions], c='#803896', linewidth=0.5)
    plt.title("Cropped signal")
    plt.legend(["GCaMP", "isosbestic"], frameon=False)
    sns.despine(left=False, bottom=False)
    plt.show(block=False)
    plt.close() 
    # Print counts
    print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

    df_PhotometryData = df_PhotometryData_1.reset_index(drop=True)  
    df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
    df_470 = df_470.reset_index(drop=True)
    df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
    df_415 = df_415.reset_index(drop=True) 
    #================================================
    """ 4.1.4 FRAME RATE """ 
    acq_FR = find_FR(df_470["Timestamp"]) 

    raw_reference = df_415[regions] #isosbestic 
    raw_signal = df_470[regions] #GCaMP signal 
    raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]
    raw_timestamps_nph_470 = df_470["Timestamp"]
    raw_timestamps_nph_415 = df_415["Timestamp"]
    raw_TTL_bpod = bpod_sync
    raw_TTL_nph = nph_sync

    my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

    df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])





    ###############################################################################################
    #KB ADDED 04Aug2024 
    test = df

    # Define the window size
    window_size = 5
    # Apply the median filter
    df_filtered = test.rolling(window=window_size, center=True).median()
    # Handle edge cases by filling NaN values
    df_filtered = df_filtered.fillna(method='ffill').fillna(method='bfill')
    print("Original DataFrame:\n", test)
    print("\nFiltered DataFrame:\n", df_filtered)
    plt.figure(figsize=(20, 10))
    plt.plot(df.raw_calcium, linewidth=0.1, alpha=0.9)
    plt.plot(df_filtered.raw_calcium, linewidth=0.1, alpha=0.85)
    plt.show()

    # Function to detect outliers using IQR
    def detect_outliers_iqr(df_filtered):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df < lower_bound) | (df > upper_bound)
        return outliers
    # Detect outliers
    outliers = detect_outliers_iqr(df_filtered)

    outliers_df = df_filtered[outliers.any(axis=1)]

    plt.figure(figsize=(20, 10))
    plt.plot(df.raw_calcium, linewidth=0.1, alpha=0.9)
    plt.plot(df_filtered.raw_calcium, linewidth=0.1, alpha=0.85) 
    plt.plot(outliers_df.raw_calcium, color='red', alpha=0.5)
    plt.show()

    print("\nOutliers detected (True indicates an outlier):\n", outliers)
    print("\nOutliers DataFrame:\n", outliers_df)
    ###############################################################################################






#%%
"""
ADD SAVE THE DATA AT THIS POINT 


"""
#%%



    # df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df)

    filepath = (f'/home/kceniabougrova/Documents/results_for_OW/Fig01_{rec.mouse}_{rec.date}_{rec.region}.png') 
    fig, ax = iblphotometry.plots.plot_raw_data_df(df_photometry, event_times=tbpod, output_file=filepath) 

    # pd.read_parquet(f'/home/kceniabougrova/Documents/results_for_OW/demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt')
    #path lib create folder 
    # df.to_parquet(f'/home/kceniabougrova/Documents/results_for_OW/demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt')
    session_path = one.eid2path(eid)
    path_rp = session_path.joinpath('raw_photometry/')
    path_rp.mkdir(parents=True, exist_ok=True)
    # df.to_parquet(path_rp.joinpath(f'demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt'))



























# %% 
########################################################################
# KB 2024-06-03 - WORKS 
# 
# loop in order to also have the behavior and the psth_idx tables saved 
# from load_processed_data.py 
########################################################################
import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import * 
import ibldsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from one.api import ONE
one = ONE() 

# ######################""" TO READ THE DATA """#################### 
# # Read the CSV file
# df_csv = pd.read_csv(os.path.join(path, path_nph+'.csv'))
# # Read the Parquet file 
# path = '/home/kceniabougrova/Documents/results_for_OW/' 
path = '/mnt/h0/kb/data/external_drive' 

# mouse = 'ZFM-04022' 
# date = '2022-12-30'
# region_number = '4'
# region = f'Region{region_number}G' 

# mouse = 'ZFM-04022' 
# date = '2022-12-30'
# region_number = '3'
# region = f'Region{region_number}G' 

# mouse = 'ZFM-04019' 
# date = '2023-01-12'
# region_number = '3'
# region = f'Region{region_number}G' 

def list_files_in_folder(folder_path, prefix):
    try:
        # Get the list of all files and directories in the specified folder
        files = os.listdir(folder_path)
        # Filter the list to include only files that start with the specified prefix
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.startswith(prefix)]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] 
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
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

# Example usage
################################################
""" CHANGE HERE """ 
# prefix = "demux_nph_ZFM-04019_" 
prefix = "demux_" 
################################################
""" CHANGE HERE """
EVENT_NAME = "feedback_times"
################################################ 
################################################

file_list = list_files_in_folder(path, prefix)

#############################################################################################
# ##### USEFUL TO CONTINUE THE LOOP IGNORING SESSIONS WITH ERRORS - already added below #####
# # The prefix you are looking for
# prefix = f'demux_nph_{mouse}_{date}'
# # Find the index using a loop
# index = -1
# for i, file_name in enumerate(file_list):
#     if file_name.startswith(prefix):
#         index = i
#         break
# if index != -1:
#     print(f"The index of the file starting with '{prefix}' is {index}.")
# else:
#     print(f"No file starting with '{prefix}' found.") 

# file_list_2 = file_list[index+1:]
#############################################################################################
file_list_2 = file_list 
#%% 
# for file_name in file_list: 
for file_name in file_list_2: 
    try: 
        mouse = file_name[10:19]
        date = file_name[20:30] 
        region_number = file_name[31:32] 
        print(mouse, date, region_number)

        eid,df_trials = get_eid(mouse,date) 
        path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
        df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

        # create trialNumber
        df_trials['trialNumber'] = range(1, len(df_trials) + 1)

        # create allContrasts 
        idx = 2
        new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
        df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
        # create allSContrasts 
        df_trials['allSContrasts'] = df_trials['allContrasts']
        df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
        df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

        # # create reactionTime
        # reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
        # df_trials["reactionTime"] = reactionTime 

        # add session info 
        df_trials["mouse"] = mouse
        df_trials["date"] = date 
        df_trials["regionNumber"] = region_number
        df_trials["eid"] = eid 
        # df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

        # remove all trials that are not totally associated with photometry, 2 seconds after the photometry was turned off 
        while (df_trials["intervals_1"].iloc[-1] + 62) >= df_nph["times"].iloc[-1]:
            df_trials = df_trials.iloc[:-1]

        # SAVE THE BEHAVIOR TABLE 
        # df_trials.to_parquet(f'/home/kceniabougrova/Documents/results_for_OW/trialstable_{mouse}_{date}_{region_number}_{eid}.pqt') 

        array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
        event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
        idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
        # print(idx_event) 

        """ create a column with the trial number in the nph df """
        df_nph["trial_number"] = 0 #create a new column for the trial_number 
        df_nph.loc[idx_event,"trial_number"]=1
        df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

        PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
        SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
        n_trials = df_trials.shape[0]

        psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

        event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

        feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

        psth_idx += feedback_idx

        photometry_feedback = df_nph.calcium.values[psth_idx] 

        np.save(f'/home/kceniabougrova/Documents/results_for_OW/psthidx_{EVENT_NAME}_{mouse}_{date}_{region_number}_{eid}.npy', photometry_feedback) 
    except: 
        prefix = f'demux_nph_{mouse}_{date}'
        # Find the index using a loop
        index = -1
        for i, file_name in enumerate(file_list):
            if file_name.startswith(prefix):
                index = i
                break
        if index != -1:
            print(f"The index of the file starting with '{prefix}' is {index}.")
        else:
            print(f"No file starting with '{prefix}' found.") 
        file_list_2 = file_list[index+1:]


#%% 
#%% 
##################################################################
# load the data and plot it - for multiple sessions 

# ######################""" TO READ THE DATA AND PLOT IT OPTION1 """#################### 

path = '/home/kceniabougrova/Documents/results_for_OW/' 

def list_files_in_folder(folder_path, prefix): 
    try:
        # Get the list of all files and directories in the specified folder
        files = os.listdir(folder_path)
        # Filter the list to include only files that start with the specified prefix
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.startswith(prefix)]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] 
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
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

psth_idx_1 = np.load(path+"psthidx_feedback_ZFM-04019_2023-03-16_3_46fe69ff-d001-4608-a15e-d5e029c14fc3.npy") 
psth_idx_2 = np.load(path+"psthidx_feedback_ZFM-04019_2022-09-16_3_69544b1b-7788-4b41-8cad-2d56d5958526.npy")
psth_appended = np.append(psth_idx_1, psth_idx_2,axis=1) 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04019_2023-03-16_3_46fe69ff-d001-4608-a15e-d5e029c14fc3.pqt") 
behav_2 = pd.read_parquet(path+"trialstable_ZFM-04019_2022-09-16_3_69544b1b-7788-4b41-8cad-2d56d5958526.pqt")
behav_multiple = [behav_1, behav_2] 
behav_concat = pd.concat(behav_multiple)
behav_concat = behav_concat.reset_index(drop=True)

indices = behav_concat["feedbackType"]==1 #pick the feedback == 1
indices_incorrect = behav_concat["feedbackType"]==-1 #pick the feedback == 1

plt.plot(psth_appended)
feedback_correct = psth_appended[:,indices]
plt.plot(feedback_correct)
feedback_incorrect = psth_appended[:,indices_incorrect] 
plt.plot(feedback_incorrect)

plt.figure(figsize=(10,10))
sns.heatmap(feedback_correct.T)
plt.axvline(x=30) 

#%%
##################################################################
# KB 2024-06-05 
# 

#######################
# IMPORT TRIALS TABLE IN A SORTED WAY
import re
from datetime import datetime 
def list_files_in_folder(path, prefix):
    return [f for f in os.listdir(path) if f.startswith(prefix)]
path = '/home/kceniabougrova/Documents/results_for_OW/' 
# Define the path and prefix
prefix = 'trialstable_'

# List files in the folder
file_list_3 = list_files_in_folder(path, prefix)

# Define a regex pattern to capture the relevant parts of the filename
pattern = re.compile(r'trialstable_(ZFM-\d+)_(\d{4}-\d{2}-\d{2})_(\d+)_')

# Function to extract sorting key from filename
def extract_key(filename):
    match = pattern.search(filename)
    if match:
        id_part = match.group(1)
        date_part = match.group(2)
        trial_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return id_part, date_obj, trial_number
    return filename  # Fallback to filename itself if pattern doesn't match

# Sort the file list using the extracted keys
sorted_file_list = sorted(file_list_3, key=extract_key)

# Store the variable names for future access
variable_names = []

# Process each file in the sorted order
for file_name in sorted_file_list:
    # Check if file name starts with the given prefix
    if file_name.startswith(prefix):
        # Extract name, date, and region from the file name
        name = file_name[12:21].replace('-', '_')
        date = file_name[22:32].replace('-', '_')
        region = file_name[33:34].replace('-', '_')
        
        # Load the parquet file into a DataFrame
        data_array_b = pd.read_parquet(os.path.join(path, file_name))
        
        # Dynamically create a variable name and assign the DataFrame
        var_name_b = f"trialstable_{name}_{date}_{region}"
        
        # Use globals() to set the variable in the global namespace
        globals()[var_name_b] = data_array_b

        # Store the variable name
        variable_names.append(var_name_b)

        # Optionally print the variable name to verify
        print(f"Loaded DataFrame: {var_name_b}")

# To access the data: 
globals()[variable_names[0]]

behav_multiple = [globals()[variable_names[0]], globals()[variable_names[5]], globals()[variable_names[7]]]
behav_concat = pd.concat(behav_multiple)
behav_concat = behav_concat.reset_index(drop=True)





# Function to list files in a folder with a given prefix
def list_files_in_folder(path, prefix):
    return [f for f in os.listdir(path) if f.startswith(prefix)]

# Define the path and prefix
prefix = 'psthidx_feedback_'

# List files in the folder
file_list = list_files_in_folder(path, prefix)

# Define a regex pattern to capture the relevant parts of the filename
pattern = re.compile(r'psthidx_feedback_(.+?)_(\d{4}-\d{2}-\d{2})_(\d+)_')

# Function to extract sorting key from filename
def extract_key(filename):
    match = pattern.search(filename)
    if match:
        name_part = match.group(1)
        date_part = match.group(2)
        region_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return name_part, date_obj, region_number
    return filename  # Fallback to filename itself if pattern doesn't match

# Sort the file list using the extracted keys
sorted_file_list = sorted(file_list, key=extract_key)

# Store the variable names for future access
variable_names = []

# Process each file in the sorted order
for file_name in sorted_file_list:
    # Check if file name starts with the given prefix
    if file_name.startswith(prefix):
        # Extract name, date, and region from the file name
        name = file_name[23:32].replace('-', '_')
        date = file_name[33:43].replace('-', '_')
        region = file_name[44:45].replace('-', '_')
        
        # Load the numpy array
        data_array = np.load(os.path.join(path, file_name))
        
        # Dynamically create a variable name and assign the numpy array
        var_name = f"psth_idx_{name}_{date}_{region}"
        
        # Use globals() to set the variable in the global namespace
        globals()[var_name] = data_array

        # Store the variable name
        variable_names.append(var_name)

        # Optionally print the variable name to verify
        print(f"Loaded numpy array: {var_name}")

# Collect variable names to plot
plot_vars = [var for var in variable_names if var.startswith('psth_idx_')]

# Example: Accessing the first numpy array
if plot_vars:
    first_var_name = plot_vars[0]
    first_array = globals()[first_var_name]
    print(f"Accessed numpy array: {first_var_name}")
    print(first_array)  # Print the numpy array or part of it
else:
    print("No numpy arrays were loaded.")
psth_multiple = [globals()[plot_vars[1]].T, globals()[plot_vars[5]].T, globals()[plot_vars[7]].T]
psth_concat = pd.concat(psth_multiple)
psth_concat = psth_concat.reset_index(drop=True)


#%%
#this one works, delete prev... 
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np

def list_files_in_folder(path, prefix):
    return [f for f in os.listdir(path) if f.startswith(prefix)]

def extract_trialstable_key(filename):
    pattern = re.compile(r'trialstable_(ZFM-\d+)_(\d{4}-\d{2}-\d{2})_(\d+)_')
    match = pattern.search(filename)
    if match:
        id_part = match.group(1)
        date_part = match.group(2)
        region_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return id_part, date_obj, region_number
    return None

def extract_psthidx_key(filename):
    pattern = re.compile(r'psthidx_feedback_(.+?)_(\d{4}-\d{2}-\d{2})_(\d+)_')
    match = pattern.search(filename)
    if match:
        name_part = match.group(1)
        date_part = match.group(2)
        region_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return name_part, date_obj, region_number
    return None

# Define paths
trialstable_path = '/home/kceniabougrova/Documents/results_for_OW/'
psthidx_path = '/home/kceniabougrova/Documents/results_for_OW/'

# List and sort trialstable files
trialstable_prefix = 'trialstable_'
trialstable_files = list_files_in_folder(trialstable_path, trialstable_prefix)
sorted_trialstable_files = sorted(trialstable_files, key=extract_trialstable_key)

# List and sort psthidx_feedback files
psthidx_prefix = 'psthidx_feedback_'
psthidx_files = list_files_in_folder(psthidx_path, psthidx_prefix)
sorted_psthidx_files = sorted(psthidx_files, key=extract_psthidx_key)

# Create dictionaries to hold the loaded data
trialstable_data = {}
psthidx_data = {}

# Load trialstable data
for file_name in sorted_trialstable_files:
    key = extract_trialstable_key(file_name)
    if key:
        name, date, region = key
        var_name_b = f"trialstable_{name}_{date.strftime('%Y_%m_%d')}_{region}"
        data_array_b = pd.read_parquet(os.path.join(trialstable_path, file_name))
        trialstable_data[(name, date, region)] = data_array_b
        print(f"Loaded DataFrame: {var_name_b}")

# Load psthidx_feedback data
for file_name in sorted_psthidx_files:
    key = extract_psthidx_key(file_name)
    if key:
        name, date, region = key
        var_name = f"psth_idx_{name}_{date.strftime('%Y_%m_%d')}_{region}"
        data_array = np.load(os.path.join(psthidx_path, file_name))
        psthidx_data[(name, date, region)] = data_array
        print(f"Loaded numpy array: {var_name}")

# Find matching keys based on mouse, date, and region number
matching_keys = set(trialstable_data.keys()) & set(psthidx_data.keys())

# Concatenate matching DataFrames and numpy arrays
behav_multiple = [trialstable_data[key] for key in matching_keys]
psth_multiple = [pd.DataFrame(psthidx_data[key].T) for key in matching_keys]

if behav_multiple:
    behav_concat = pd.concat(behav_multiple).reset_index(drop=True)
    print("Concatenated behavior DataFrame:")
    print(behav_concat.head())

if psth_multiple:
    psth_concat = pd.concat(psth_multiple).reset_index(drop=True)
    print("Concatenated PSTH DataFrame:")
    print(psth_concat.head())





#%% 
##################################################################
# load the data and plot it - for 1 individual session 
nph_feedback = np.load(f'/home/kceniabougrova/Documents/results_for_OW/psthidx_feedback_{mouse}_{date}_{region_number}_{eid}.npy') 

sns.heatmap(nph_feedback.T, cbar=True)
plt.show() 
#%%
width_ratios = [1, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15 
data = df_nph.calcium.values[psth_idx]
event_time = 30 
average_values = data.mean(axis=1)
plt.plot(average_values, color='black')
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.grid(False) 
plt.show()
#%%
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 
event_time = 30 

psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1))]]

# plt.plot(psth_50, color='#cdb4db', linewidth=1, alpha=0.2) 
# plt.plot(psth_20, color='#ffafcc', linewidth=1, alpha=0.2) 
# plt.plot(psth_80, color='#a2d2ff', linewidth=1, alpha=0.2) 
average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 
plt.plot(average_values_50, color='#cdb4db', linewidth=3) 
plt.plot(average_values_20, color='#ffafcc', linewidth=3) 
plt.plot(average_values_80, color='#a2d2ff', linewidth=3) 
plt.title("NM response at feedback outcome = correct, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("0.5","0.2","0.8"), fontsize=FONTSIZE_3, frameon=False) 

# plt.ylim(-0.005,0.004) 
plt.grid(False) 
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.show() 


psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1))]] 
average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 
plt.plot(average_values_50, color='#cdb4db', linewidth=3) 
plt.plot(average_values_20, color='#ffafcc', linewidth=3) 
plt.plot(average_values_80, color='#a2d2ff', linewidth=3) 
plt.title("NM response at feedback outcome = incorrect, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("0.5","0.2","0.8"), fontsize=FONTSIZE_3, frameon=False) 
plt.grid(False) 
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.show()

# %% 
# ######################""" TO READ THE DATA AND PLOT IT OPTION2 """#################### 
#####stuff
# Step 1: Load numpy arrays from files
for file_name in file_list:
    # Check if file name starts with 'psthidx_feedback_'
    if file_name.startswith('psthidx_feedback_'):
        # Extract name, date, and region from the file name
        name = file_name[23:32].replace('-', '_')
        date = file_name[33:43].replace('-', '_')
        region = file_name[44:45].replace('-', '_')
        
        # Load the numpy array
        data_array = np.load(path + file_name)
        
        # Dynamically create a variable name and assign the numpy array
        var_name = f"psth_idx_{name}_{date}_{region}"
        
        # Use globals() to set the variable in the global namespace
        globals()[var_name] = data_array

        # Optionally print the variable name to verify
        print(f"Loaded numpy array: {var_name}")

# Collect variable names to plot
plot_vars = [var for var in globals() if var.startswith('psth_idx_')]

# Step 2: Plot all loaded numpy arrays
# plt.figure(figsize=(10, 6))
# for var in plot_vars:
#     data_array = globals()[var]
#     plt.plot(data_array, label=var)

# plt.title('Line Plot of All Loaded Arrays')
# plt.xlabel('X-axis Label')  # Replace with appropriate label
# plt.ylabel('Y-axis Label')  # Replace with appropriate label
# plt.legend()
# plt.show() 
  







#%% 
##############################################################################
# complete code to load and have multiple sessions, then filter them and plot 
############################################################################## 















indices = behav_concat["mouse"]=="ZFM-06275" 

psth_concat_2 = psth_concat
feedback_correct = psth_concat_2[indices].T

plt.plot(feedback_correct, alpha=0.2, color="gray", linewidth=1)
plt.axvline(x=30)






#%% 
""" WORKS - ALL PHOTOMETRY PQT FILES INTO 1 FOLDER 07Aug2024 """
""" WORKS, WITH PLOTS """
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv')
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']]

# test = df_gs[100:102].reset_index(drop=True)

EXCLUDES = [] 
IMIN = 0  # To start from here when rerunning; from scratch: write 0

for i in range(len(df_gs)):
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue

    mouse = df_gs.Mouse[i]
    date = df_gs.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_gs.region[i]
    region = f"Region{region}G"
    
    eid, df_trials = get_eid(mouse, date)
    
    print(f"{mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    
    sl = SessionLoader(one=one, eid=eid) 
    file_photometry = sl.session_path.joinpath("alf", region, "raw_photometry.pqt")
    df_ph = pd.read_parquet(file_photometry) 
    print(df_ph)
    new_file_name = f"nph_{mouse}_{date}_{region}.pqt"
    new_file_path = os.path.join("/mnt/h0/kb/data/nph_pqt", new_file_name)
    new_file_name2 = f"nph_session_cut_{mouse}_{date}_{region}.pqt"
    new_file_path2 = os.path.join("/mnt/h0/kb/data/nph_pqt/session_cut", new_file_name2)
    
    df_ph_entire_signal = df_ph
    test=df_ph_entire_signal
    test['calcium_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_calcium"].values, fs=fs) #KB
    test['isosbestic_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_isosbestic"], fs=fs)
    test['calcium_jove2019'] = jove2019(df_ph_entire_signal["raw_calcium"], df_ph_entire_signal["raw_isosbestic"], fs=fs) 
    test['isosbestic_jove2019'] = jove2019(df_ph_entire_signal["raw_isosbestic"], df_ph_entire_signal["raw_calcium"], fs=fs)
    test['calcium_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_calcium"].values, df_ph_entire_signal["times"].values, fs=fs)
    test['isosbestic_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_isosbestic"].values, df_ph_entire_signal["times"].values, fs=fs) 
    df_ph_entire_signal=test
    df_ph_entire_signal.to_parquet(new_file_path)

    df_ph_crop = df_ph
    trial_start = df_trials["intervals_0"].iloc[0] - 30
    trial_end = df_trials["intervals_1"].iloc[-1] + 30
    selected_df_nph = df_ph_crop[(df_ph_crop["times"] >= trial_start) & (df_ph_crop["times"] <= trial_end)]
    selected_df_nph.to_parquet(new_file_path2)



    import os

    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    color_1 = '#8B4513'  # Brownish/greyish for first plot
    color_2 = '#A0522D'  # Brownish/greyish for second plot
    color_3 = '#708090'  # Greyish for third plot
    color_4 = '#008080'  # Teal for fourth plot

    axes[0].plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
    axes[0].set_ylabel('Raw Calcium')
    axes[0].set_title('Entire Signal, Raw Calcium')

    axes[1].plot(df_ph_entire_signal.times, df_ph_entire_signal.calcium_jove2019, color=color_2, linewidth=0.1)
    axes[1].set_ylabel('Processed Calcium')
    axes[1].set_title('Entire Signal, Calcium Preprocessed')

    axes[2].plot(selected_df_nph.times, selected_df_nph.raw_calcium, color=color_3, linewidth=0.1)
    axes[2].set_ylabel('Raw Calcium')
    axes[2].set_title('Signal Cut, Raw Calcium')

    axes[3].plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
    axes[3].set_ylabel('Processed Calcium')
    axes[3].set_title('Signal Cut, Calcium Preprocessed')
    axes[3].set_xlabel('Time (s)')

    # Check if the 399th trial exists and draw a vertical line
    if len(df_trials) > 399:
        trial_400 = df_trials["intervals_1"].iloc[399]
        axes[3].axvline(x=trial_400, linestyle="dashed", color="gray")

    fig.suptitle(f"{mouse}_{date}_{region}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



    color_1 = '#8B4513'  # Brownish/greyish for the first plot
    color_4 = '#008080'  # Teal for the fourth plot

    new_file_name2 = f"nph_session_cut_{mouse}_{date}_{region}.png"
    new_file_path2 = os.path.join("/mnt/h0/kb/data/nph_pqt/session_cut", new_file_name2)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
    axes[0].set_ylabel('Raw Calcium')
    axes[0].set_title('Entire Signal, Raw Calcium')

    axes[1].plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
    axes[1].set_ylabel('Processed Calcium')
    axes[1].set_title('Signal Cut, Calcium Preprocessed')
    axes[1].set_xlabel('Time (s)')

    # Check if the 399th trial exists and draw a vertical line on the second subplot
    if len(df_trials) > 399:
        trial_400 = df_trials["intervals_1"].iloc[399]
        axes[1].axvline(x=trial_400, linestyle="dashed", color="gray")

    fig.suptitle(f"{mouse}_{date}_{region}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(new_file_path2, format='png')
    plt.show()

    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path2}")

    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path}")




#%% 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv')
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']]

# test = df_gs[100:102].reset_index(drop=True)

EXCLUDES = [] 
IMIN = 0  # To start from here when rerunning; from scratch: write 0

for i in range(len(df_gs)):
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue

    mouse = df_gs.Mouse[i]
    date = df_gs.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_gs.region[i]
    region = f"Region{region}G"
    
    eid, df_trials = get_eid(mouse, date) 
    df_trials["mouse"] = mouse
    df_trials["date"] = date 
    df_trials["region"] = region
    
    print(f"{mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    
    sl = SessionLoader(one=one, eid=eid) 
    file_photometry = sl.session_path.joinpath("alf", region, "raw_photometry.pqt")
    df_ph = pd.read_parquet(file_photometry) 
    print(df_ph)
    new_file_name = f"nph_{mouse}_{date}_{region}.pqt"
    new_file_path = os.path.join("/mnt/h0/kb/data/nph_pqt", new_file_name)
    new_file_name2 = f"nph_session_cut_{mouse}_{date}_{region}.pqt"
    new_file_path2 = os.path.join("/mnt/h0/kb/data/nph_pqt/session_cut", new_file_name2)
    
    df_ph_entire_signal = df_ph
    test=df_ph_entire_signal
    test['calcium_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_calcium"].values, fs=fs) #KB
    test['isosbestic_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_isosbestic"], fs=fs)
    test['calcium_jove2019'] = jove2019(df_ph_entire_signal["raw_calcium"], df_ph_entire_signal["raw_isosbestic"], fs=fs) 
    test['isosbestic_jove2019'] = jove2019(df_ph_entire_signal["raw_isosbestic"], df_ph_entire_signal["raw_calcium"], fs=fs)
    test['calcium_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_calcium"].values, df_ph_entire_signal["times"].values, fs=fs)
    test['isosbestic_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_isosbestic"].values, df_ph_entire_signal["times"].values, fs=fs) 
    df_ph_entire_signal=test
    df_ph_entire_signal.to_parquet(new_file_path)

    df_ph_crop = df_ph
    trial_start = df_trials["intervals_0"].iloc[0] - 30
    trial_end = df_trials["intervals_1"].iloc[-1] + 30
    selected_df_nph = df_ph_crop[(df_ph_crop["times"] >= trial_start) & (df_ph_crop["times"] <= trial_end)]
    selected_df_nph = selected_df_nph.reset_index(drop=True)
    selected_df_nph.to_parquet(new_file_path2)


    nph_times = np.array(selected_df_nph.times) #pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
    idx_event = np.searchsorted(nph_times, event_test) #check idx where they would be included, in a sorted way 
    # print(idx_event) 

    selected_df_nph["trial_number"] = 0 #create a new column for the trial_number 
    selected_df_nph.loc[idx_event,"trial_number"]=1
    selected_df_nph["trial_number"] = selected_df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = int(1/np.mean(np.diff(nph_times))) #not a constant: print(1/np.mean(np.diff(nph_times))) #sampling rate #acq_FR
    EVENT_NAME = "feedback_times" 

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    df_trials_400 = df_trials[0:401] 
    n_trials_400 = df_trials_400.shape[0] 

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

    feedback_idx = np.searchsorted(nph_times, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    photometry_feedback = selected_df_nph.calcium_jove2019.values[psth_idx] 

    photometry_feedback_avg = np.mean(photometry_feedback, axis=1)
    # plt.plot(photometry_feedback_avg) 

    
    psth_idx_400 = np.tile(sample_window[:,np.newaxis], (1, n_trials_400)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback_400 = np.array(df_trials_400[EVENT_NAME]) #pick the feedback timestamps 

    feedback_idx_400 = np.searchsorted(nph_times, event_feedback_400) #check idx where they would be included, in a sorted way 

    psth_idx_400 += feedback_idx_400

    photometry_feedback_400 = selected_df_nph.calcium_jove2019.values[psth_idx_400] 

    photometry_feedback_avg_400 = np.mean(photometry_feedback_400, axis=1)
    # plt.plot(photometry_feedback_avg) 



    import os

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    color_1 = '#008080'  # Brownish/greyish for first plot
    color_2 = '#aa3e98' #add a purple greyish color 
    color_3 = '#708090'  # Greyish for third plot
    color_4 = '#008080'  # Teal for fourth plot

    axes[0].plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
    axes[0].set_ylabel('Raw Calcium')
    axes[0].set_title('Entire Signal, Raw Calcium')

    axes[1].plot(selected_df_nph.times, selected_df_nph.raw_isosbestic, color=color_2, linewidth=0.1)
    axes[1].set_ylabel('Raw isosbestic')
    axes[1].set_title('Signal Cut, Raw Isosbestic')

    axes[2].plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
    axes[2].set_ylabel('Processed Calcium')
    axes[2].set_title('Signal Cut, Calcium Preprocessed')
    axes[2].set_xlabel('Time (s)') 

    # Check if the 399th trial exists and draw a vertical line
    if len(df_trials) > 399:
        trial_400 = df_trials["intervals_1"].iloc[399]
        axes[2].axvline(x=trial_400, linestyle="dashed", color="gray")

    fig.suptitle(f"{mouse}_{date}_{region}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show() 



    def avg_sem(data):
        avg = data.mean(axis=1)
        sem = data.std(axis=1) / np.sqrt(data.shape[1])
        return avg, sem

    # Create the figure and gridspec
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

    def plot_neuromodulator(psth_combined, df_trials, psth_combined_400, df_trials_400, title, mouse):
        psth_correct = psth_combined[:, (df_trials.feedbackType == 1)] 
        psth_correct_400 = psth_combined_400[:, (df_trials_400.feedbackType == 1)] 

        avg, sem = avg_sem(psth_correct)
        color = "#218380"
        plt.plot(avg, color=color, linewidth=2, label=f'{date}')
        plt.fill_between(range(len(avg)), avg - sem, avg + sem, color=color, alpha=0.18)
        avg2, sem2 = avg_sem(psth_correct_400)
        color = "#aa3e98"
        plt.plot(avg2, color=color, linewidth=2, label="first 400 trials")
        plt.fill_between(range(len(avg2)), avg2 - sem2, avg2 + sem2, color=color, alpha=0.18)
        
        plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        plt.ylabel('Average Value')
        plt.xlabel('Time')
        plt.title(title+ ' mouse '+mouse, fontsize=16)

    # Plot for DA
    fig = plt.figure(figsize=(12, 12))
    plot_neuromodulator(photometry_feedback, df_trials, photometry_feedback_400, df_trials_400, title = "psth aligned to feedback ", mouse=df_trials.mouse[0]) #change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Adding legend outside the plots
    plt.legend(fontsize=14)
    fig.suptitle('Neuromodulator activity for correct trials across different sessions in 1 mouse', y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()










    color_1 = '#8B4513'  # Brownish/greyish for the first plot
    color_4 = '#008080'  # Teal for the fourth plot

    new_file_name2 = f"nph_session_cut_{mouse}_{date}_{region}.png"
    new_file_path2 = os.path.join("/mnt/h0/kb/data/nph_pqt/session_cut", new_file_name2)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
    axes[0].set_ylabel('Raw Calcium')
    axes[0].set_title('Entire Signal, Raw Calcium')

    axes[1].plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
    axes[1].set_ylabel('Processed Calcium')
    axes[1].set_title('Signal Cut, Calcium Preprocessed')
    axes[1].set_xlabel('Time (s)')

    # Check if the 399th trial exists and draw a vertical line on the second subplot
    if len(df_trials) > 399:
        trial_400 = df_trials["intervals_1"].iloc[399]
        axes[1].axvline(x=trial_400, linestyle="dashed", color="gray")

    fig.suptitle(f"{mouse}_{date}_{region}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(new_file_path2, format='png')
    plt.show()

    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path2}")

    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path}")
# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming df_ph_entire_signal, selected_df_nph, df_trials, and the necessary data for the neuromodulator plot are already defined.

def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def plot_neuromodulator(ax, psth_combined, df_trials, psth_combined_400, df_trials_400, title, mouse, frame_rate=30):
    psth_correct = psth_combined[:, (df_trials.feedbackType == 1)] 
    psth_correct_400 = psth_combined_400[:, (df_trials_400.feedbackType == 1)] 

    avg, sem = avg_sem(psth_correct)
    color = "#218380"
    x_values_seconds = (np.arange(len(avg)) - 30) / frame_rate
    ax.plot(x_values_seconds, avg, color=color, linewidth=2, label=f'{date}')
    ax.fill_between(x_values_seconds, avg - sem, avg + sem, color=color, alpha=0.18)

    avg2, sem2 = avg_sem(psth_correct_400)
    ax.plot(x_values_seconds, avg2, color="#aa3e98", linewidth=2, label="first 400 trials")
    ax.fill_between(x_values_seconds, avg2 - sem2, avg2 + sem2, color="#aa3e98", alpha=0.18)
    
    ax.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax.set_ylabel('Average Value')
    ax.set_xlabel('Time (s)')
    ax.set_title(title + ' mouse ' + mouse, fontsize=16)

# Create the figure and a 2x2 gridspec layout
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(3, 4, width_ratios=[2, 2, 2, 3])

# Subplots on the left (3 vertically stacked)
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[1, 0:3])
ax3 = fig.add_subplot(gs[2, 0:3])

# Neuromodulator plot on the right
ax4 = fig.add_subplot(gs[:, 3])

# First three plots
ax1.plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
ax1.set_ylabel('Raw Calcium')
ax1.set_title('Entire Signal, Raw Calcium')
ax1.axvline(x=0, linestyle="dashed", color="gray")

ax2.plot(selected_df_nph.times, selected_df_nph.raw_isosbestic, color=color_2, linewidth=0.1)
ax2.axvline(x=0, linestyle="dashed", color="gray")

ax2.set_ylabel('Raw isosbestic')
ax2.set_title('Signal Cut, Raw Isosbestic')

ax3.plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
ax3.set_ylabel('Processed Calcium')
ax3.set_title('Signal Cut, Calcium Preprocessed')
ax3.set_xlabel('Time (s)')
ax3.axvline(x=0, linestyle="dashed", color="gray")

# Check if the 399th trial exists and draw a vertical line
if len(df_trials) > 399:
    trial_400 = df_trials["intervals_1"].iloc[399]
    ax3.axvline(x=trial_400, linestyle="dashed", color="gray")

# Neuromodulator plot
plot_neuromodulator(ax4, photometry_feedback, df_trials, photometry_feedback_400, df_trials_400, title="psth aligned to feedback", mouse=df_trials.mouse[0])

# Adding legend to the neuromodulator plot
ax4.legend(fontsize=14)

# Super title
fig.suptitle(f"{mouse}_{date}_{region}", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
