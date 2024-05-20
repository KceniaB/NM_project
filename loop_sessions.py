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
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
import iblphotometry.kcenia as kcenia 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal

import ibllib.plots

from one.api import ONE #always after the imports 
one = ONE()

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str)

#%%
""" PHOTOMETRY """ 
# df_test = df1[(df1.date == "2024-01-24") & (df1.mouse == "ZFM-06948")] 
# df_test = df1[(df1.date == "2024-03-22") & (df1.mouse == "ZFM-06948")]

for i,rec in df1.iterrows(): 
    # if rec.mouse == 'ZFM-06275': #comment (*A)
    #     # continue #COMMENT (*A)
    # continue
    #get data info
    regions = kcenia.get_regions(rec)
    #get behav
    eid, df_trials = kcenia.get_eid(rec) 
    #get photometry 
    df_nph, df_nphttl = kcenia.get_nph(rec)
    #get TTLs 
    tph, tbpod = kcenia.get_ttl(df_DI0 = df_nphttl, df_trials = df_trials) 

    df_PhotometryData = df_nph 

    # try:
    #     tbpod = np.sort(np.r_[
    #     df_trials['intervals_0'].values,
    #     df_trials['intervals_1'].values,
    #     df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    #     )
    #     fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    #     assert len(iph)/len(tbpod) > .9
    # except AssertionError:
    #     print("mismatch in sync, will try to add ITI duration to the sync")
    #     tbpod = np.sort(np.r_[
    #     df_trials['intervals_0'].values,
    #     df_trials['intervals_1'].values - 1,  # here is the trick
    #     df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    #     )
    #     fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    #     assert len(iph)/len(tbpod) > .9
    #     print("recovered from sync mismatch, continuing") 
    #alternative 
    try:
        tbpod = np.sort(np.r_[
            df_trials['intervals_0'].values,
            df_trials['intervals_1'].values,
            df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
        )
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
    except AssertionError:
        print("mismatch in sync, will try to add ITI duration to the sync")
        try:
            tbpod = np.sort(np.r_[
                df_trials['intervals_0'].values,
                df_trials['intervals_1'].values - 1,  # here is the trick
                df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
            )
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
            assert len(iph)/len(tbpod) > .9
            print("recovered from sync mismatch, continuing")
        except AssertionError:
            print("mismatch, maybe this is an old session")
            tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
            fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
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
    fig.savefig(f'/home/kceniabougrova/Documents/results_for_OW/Fig00_TTL_{rec.mouse}_{rec.date}_{rec.region}.png')

    # intervals_0_event = np.sort(np.r_[df_trials['intervals_0'].values]) 
    # len(intervals_0_event)

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
    df_PhotometryData = kcenia.LedState_or_Flags(df_PhotometryData)

    """ 4.1.2 Check for LedState/previous Flags bugs """ 
    """ 4.1.2.1 Length """
    # Verify the length of the data of the 2 different LEDs
    df_470, df_415 = kcenia.verify_length(df_PhotometryData)
    """ 4.1.2.2 Verify if there are repeated flags """ 
    kcenia.verify_repetitions(df_PhotometryData["LedState"])
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
    acq_FR = kcenia.find_FR(df_470["Timestamp"]) 

    raw_reference = df_415[regions] #isosbestic 
    raw_signal = df_470[regions] #GCaMP signal 
    raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]
    raw_timestamps_nph_470 = df_470["Timestamp"]
    raw_timestamps_nph_415 = df_415["Timestamp"]
    raw_TTL_bpod = bpod_sync
    raw_TTL_nph = nph_sync

    my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

    df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])

    df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df)

    filepath = (f'/home/kceniabougrova/Documents/results_for_OW/Fig01_{rec.mouse}_{rec.date}_{rec.region}.png') 
    fig, ax = iblphotometry.plots.plot_raw_data_df(df_photometry, event_times=tbpod, output_file=filepath) 

    # pd.read_parquet(f'/home/kceniabougrova/Documents/results_for_OW/demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt')
    #path lib create folder 
    df.to_parquet(f'/home/kceniabougrova/Documents/results_for_OW/demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt')
    session_path = one.eid2path(eid)
    path_rp = session_path.joinpath('raw_photometry/')
    path_rp.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_rp.joinpath(f'demux_nph_{rec.mouse}_{rec.date}_{rec.region}_{eid}.pqt'))







# %%
