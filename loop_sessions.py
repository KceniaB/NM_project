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
from brainbox.io.one import SessionLoader
from one.api import ONE
one = ONE()
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm2 import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
import neurodsp.utils


df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024') 


#%%
""" PHOTOMETRY """ 
df_test = df1[(df1.date == "2024-01-24") & (df1.Mouse == "ZFM-06948")]

#get data info
mouse, date, nphfile_number, bncfile_number, region, region2, nm = extract_data_info(df = df_test)
#get behav
eid, df_trials = get_eid(mouse=mouse,date=date) 
#get photometry 
df_nph, df_nphttl = get_nph(date=date, nphfile_number=nphfile_number, bncfile_number=bncfile_number)
#get TTLs 
tph, tbpod = get_ttl(df_DI0 = df_nphttl, df_trials = df_trials) 









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
    tbpod = np.sort(np.r_[
    df_trials['intervals_0'].values,
    df_trials['intervals_1'].values - 1,  # here is the trick
    df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    assert len(iph)/len(tbpod) > .9
    print("recovered from sync mismatch, continuing") 

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.diff(tph))
axs[0].plot(np.diff(tbpod))
axs[0].legend(['ph', 'pbod'])

print('max deviation:', np.max(np.abs(fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]) * 1e6), 'drift: ', drift_ppm, 'ppm')

#fcn_nph_to_bpod_times  # apply this function to whatever photometry timestamps

axs[1].plot(np.diff(fcn_nph_to_bpod_times(tph[iph])))
axs[1].plot(np.diff(tbpod[ibpod]))
axs[1].legend(['ph', 'pbod']) 




# %%
