"""
2024-March-15
KceniaB


""" 


""" 
    1. get data files 
        df_np
        df_di
        df_behav
        
    2. add needed columns
        session day
        mouse name
        task protocol
        region 
    
""" 

#%%

# imports and loading data
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ibldsp.utils




#%%

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

# %%
def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    import numpy as np
    import pandas as pd
    if one is None:
        from one.api import ONE 
        ONE() 
        one = ONE() 

    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'],
         trials['feedback_times'], trials['firstMovement_times'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times', 'trials.firstMovement_times',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.shape[0] == 0:
            return
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
#        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
#          trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
#          trials['feedbackType'], trials['choice'], trials['firstMovement_times'],
#          trials['feedback_times']) = one.load(
#                              eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
#                                                  'trials.goCue_times', 'trials.probabilityLeft',
#                                                  'trials.contrastLeft', 'trials.contrastRight',
#                                                  'trials.feedbackType', 'trials.choice',
#                                                  'trials.firstMovement_times',
#                                                  'trials.feedback_times'])
        try:
            trials = one.load_object(eid, 'trials') #210810 Updated by brandon due to ONE update
        except:
            return {}
            
            
            
    if len(trials['probabilityLeft']) == 0: # 210810 Updated by brandon due to ONE update
        return
#     if trials.shape[0] == 0:
#         return
#     trials['signed_contrast'] = trials['contrastRight']
#     trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
#     trials['correct'] = trials['feedbackType']
#     trials.loc[trials['correct'] == -1, 'correct'] = 0
#     trials['right_choice'] = -trials['choice']
#     trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
#     trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
#     trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
#                'stim_side'] = 1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
#                'stim_side'] = -1
    assert np.all(np.logical_xor(np.isnan(trials['contrastRight']),np.isnan(trials['contrastLeft'])))
    
    trials['signed_contrast'] = np.copy(trials['contrastRight'])
    use_trials = np.isnan(trials['signed_contrast'])
    trials['signed_contrast'][use_trials] = -np.copy(trials['contrastLeft'])[use_trials]
    trials['correct'] = trials['feedbackType']
    use_trials = (trials['correct'] == -1)
    trials['correct'][use_trials] = 0
    trials['right_choice'] = -np.copy(trials['choice'])
    use_trials = (trials['right_choice'] == -1)
    trials['right_choice'][use_trials] = 0
    trials['stim_side'] = (np.isnan(trials['contrastLeft'])).astype(int)
    use_trials = (trials['stim_side'] == 0)
    trials['stim_side'][use_trials] = -1
#     if 'firstMovement_times' in trials.columns.values:
    trials['reaction_times'] = np.copy(trials['firstMovement_times'] - trials['goCue_times'])
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials








#%%
############################################################################

#%% 
#===========================================================================
#  ?                                ABOUT
#  @author         :  Kcenia Bougrova
#  @repo           :  KceniaB
#  @createdOn      :  photometry_processing_new 12122022
#  @description    :  process the photometry data and align to the behavior 
#  @lastUpdate     :  2023-06-05
#===========================================================================

#===========================================================================
#  *                                 INFO
#    * NEW CODE Sep 2022
#    
#    * UPDATES: 
#       * 2022-10-03 
#           optimized functions to read LedState or Flags
#           - 
#=========================================================================== 

#===========================================================================
#  todo                             TODO
#    * 2022-10-10 
#       - tc_dual_to_plot["all_contrasts_0joined"] CHANGE TO all_contrasts_0separated 
#           [ ] DONE
#       - b_2 = tc_dual_to_plot.loc[:, 'contrastLeft':'all_contrasts_0joined'] ADD 10 COLOR TO PLOT FUNCTION 
#           [ ] DONE 
#===========================================================================


#%%
#===========================================================================
#                            1. IMPORTS
#===========================================================================

from array import ArrayType, array
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.rcmod import set_theme
from scipy import stats
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve 
import os.path
from os import path

#from IBL
#old: 
# from ibllib.io.extractors.habituation_trials import HabituationTrials 
# from ibllib.io.extractors.training_trials import extract_all
from ibllib.io.extractors.biased_trials import extract_all 
#new: 
import ibllib.io.raw_data_loaders as raw
#from ibllib.io.extractors import habituation_trials, training_trials, biased_trials 

#=========================================================================== 

# #S5 HCW1: 
# session_path = '/home/kcenia/Documents/Photometry_results/2023-05-12/raw_photometry2.csv' 
# mouse = "D6"
# region = 'Region1G' 
# session_day = session_path[-30:-20]
# session_path_behav = '/home/kcenia/Documents/Photometry_results/2023-05-12/D6/001' 
# io_path = '/home/kcenia/Documents/Photometry_results/2023-05-12/DI00.csv' 
# df_PhotometryData = pd.read_csv(session_path) 


#%% 
# init_idx = 
# end_idx = 


#%% 
#===========================================================================
#                         3. FUNCTIONS FROM GITHUB
#===========================================================================

def get_zdFF(reference,signal,smooth_win=10,remove=200,lambd=5e4,porder=1,itermax=50): 
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent 
    and calcium-dependent signals
    
    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for 
                    green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer
        remove: the beginning of the traces with a big slope one would like to remove, integer
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,  
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        zdFF - z-score dF/F, 1D numpy array
    '''
    
    import numpy as np
    from sklearn.linear_model import Lasso

    # Smooth signal
    reference = smooth_signal(reference, smooth_win)
    signal = smooth_signal(signal, smooth_win)
    
    # Remove slope using airPLS algorithm
    r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax) 

    # Remove baseline and the begining of recording
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])   

    # Standardize signals    
    reference = (reference - np.median(reference)) / np.std(reference)
    signal = (signal - np.median(signal)) / np.std(signal)
    
    # Align reference signal to calcium signal using non-negative robust linear regression
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(reference)
    lin.fit(reference.reshape(n,1), signal.reshape(n,1))
    reference = lin.predict(reference.reshape(n,1)).reshape(n,)

    # z dFF    
    zdFF = (signal - reference)
    
    return zdFF

def smooth_signal(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """

    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z
#%%
#===========================================================================
#                            BNC TTL DATA 
# WORKS! #KB 20230605
#===========================================================================
def import_DI(io_path): 
    """
    input = raw DI data (TTL) 
    output = "Timestamp"; DI TTL times, only True values, this means only ups 

     > columns: 
         * old ones = "Seconds" 
         * new ones = "Timestamp" 
    """    
    df_DI0 = pd.read_csv(io_path) 
    if 'Value.Value' in df_DI0.columns: #for the new ones 
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else: 
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones 
    #use Timestamp from this part on, for any of the files 
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]  
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"]) 
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    return(df_raw_phdata_DI0_T_timestamp) 

#%% 
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
""" 4.1 PHOTOMETRY """
# * * * * * * * * * * Load Photometry data * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
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
""" 4.1.1 Change the Flags that are combined to Flags that will represent only the LED that was on """ 
"""1 and 17 are isosbestic; 2 and 18 are GCaMP"""
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

#%% 
""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """ 

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

#%% 
""" 4.1.2.2 Verify if there are repeated flags """ 
def verify_repetitions(x): 
    """
    Checking if there are repetitions in consecutive rows
    x = df_PhotometryData["Flags"]
    """ 
    for i in range(1,(len(x)-1)): 
        if x[i-1] == x[i]: 
            print("here: ", i)

# %% 
# """ 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
# #Special for this case: 
# def plot_all_signal_01(df_470,df_415,region): 
#     sns.set(style = "ticks") 
#     f = plt.figure()
#     f.set_figwidth(10)
#     f.set_figheight(5)
#     plt.rcParams['figure.dpi'] = 300
#     plt.plot(df_470[region], linewidth = 0.75, c = '#279F95') 
#     plt.plot(df_415[region], linewidth = 0.75, c = '#803896') 
#     plt.suptitle("Raw signal - "+mouse+" "+session_day+" "+region,y=1.05) 
#     #plt.title("entire session")
#     plt.xlabel("time")
#     plt.ylabel("signal") 
#     plt.xticks(fontsize = 10)
#     plt.yticks(fontsize = 10)
#     leg = plt.legend(["GCAMP","isos"],frameon=False)
#     # change the line width for the legend
#     for line in leg.get_lines():
#         line.set_linewidth(3.0)
#     sns.despine(left = False, bottom = False) 
#     # plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/'+'RawSignal_'+mouse+'_'+session_day+'_'+region+'.png', dpi=300, bbox_inches = "tight") 
#     return f
#================================= OUTLIERS and PLOTS ================================
# https://matplotlib.org/stable/gallery/color/color_by_yvalue.html 
""" 
Calculate the outliers and plot them
1. Calculate the outliers (time_u, upper and time_l, lower) 
2. Plot the entire session, gcamp and isosbestic, scatter for the outliers
3. Plot only gcamp and the outliers with ylim to see the signal zoomed in 
"""
def plot_outliers(df_470,df_415,region,mouse,session_day): 
    df_470 = df_470.reset_index(drop=True) 
    df_415 = df_415.reset_index(drop=True)
    q1, q3= np.percentile(df_470[region],[25,75])
    iqr = q3 - q1 
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr) 
    t=df_470["Timestamp"]
    s=df_470[region]
    t=t.reset_index(drop=True)
    s=s.reset_index(drop=True)
    upper=[]
    time_u=[]
    lower=[] 
    time_l=[]
    for i in range(len(s)): 
        if s[i] > upper_bound: 
            # print(s[i]," value is above the upper threshold")
            upper.append(s[i])
            time_u.append(t[i])
        elif s[i] < lower_bound: 
            # print(s[i]," value is below the upper threshold") 
            lower.append(s[i]) 
            time_l.append(t[i])
    percentage_upper = (len(upper)/len(s)*100)
    percentage_lower = (len(lower)/len(s)*100)
    # print("outliers above: ",percentage_upper," | outliers below: ",percentage_lower) 

    sns.set(style = "ticks") 
    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(5)
    plt.rcParams['figure.dpi'] = 300
    plt.plot(df_470["Timestamp"],df_470[region], linewidth = 0.25, c = '#279F95') 
    plt.plot(df_415["Timestamp"],df_415[region], linewidth = 0.25, c = '#803896') 
    plt.suptitle("Raw signal - "+mouse+" "+session_day+" "+region,y=1.05) 
    #plt.title("entire session")
    plt.xlabel("time")
    plt.ylabel("signal") 
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10) 
    plt.axhline(y=upper_bound, color="red", linewidth=0.25)
    plt.axhline(y=lower_bound, color="crimson", linewidth=0.25)
    # plt.ylim(lower_bound-(lower_bound/20),upper_bound+(upper_bound/20)) 
    leg = plt.legend(["GCAMP","isos"],frameon=False)
    plt.scatter(time_u,upper,color="gold") 
    plt.scatter(time_l,lower,color="orange")
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    sns.despine(left = False, bottom = False) 
    # plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/'+'RawSignal_'+mouse+'_'+session_day+'_'+region+'_all.png', dpi=300, bbox_inches = "tight")     
    plt.show()

    sns.set(style = "ticks") 
    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(5)
    plt.rcParams['figure.dpi'] = 300
    plt.plot(df_470["Timestamp"],df_470[region], linewidth = 0.75, c = '#279F95') 
    plt.suptitle("Raw signal GCaMP - "+mouse+" "+session_day+" "+region,y=1.05) 
    #plt.title("entire session")
    plt.xlabel("time")
    plt.ylabel("signal") 
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10) 
    plt.axhline(y=upper_bound, color="red")
    plt.axhline(y=lower_bound, color="crimson")
    plt.ylim(lower_bound-(lower_bound/25),upper_bound+(upper_bound/25)) 
    plt.scatter(time_u,upper,color="gold", linewidth=0.25) 
    plt.scatter(time_l,lower,color="orange", linewidth=0.25)
    leg = plt.legend(["GCAMP"],frameon=False)
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    sns.despine(left = False, bottom = False) 
    # plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/'+'RawSignal_'+mouse+'_'+session_day+'_'+region+'_GCaMP.png', dpi=300, bbox_inches = "tight")     
    plt.show()

    plt.rcParams['figure.dpi'] = 100
    plt.hist(time_u)
    plt.xlim(df_470["Timestamp"][0],(df_470["Timestamp"][len(df_470["Timestamp"])-1])) 
    plt.suptitle("Outliers density - "+mouse+" "+session_day+" "+region,y=1,size=12) 
    sns.despine(left = False, bottom = False) 
    plt.figure(figsize=(8,5))
    plt.show()

    print("outliers above: ",percentage_upper," | outliers below: ",percentage_lower) 

# %% 
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

#%% 
#===========================================================================
#                           4.3 BEHAVIOR Bpod data
#===========================================================================
# * * * * * * * * * * LOAD BPOD DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
#! RECHECK!!!!! 

# OUT_DATA is the important one 
def extract_behav_t(session_path_behav): 
    from ibllib.io.extractors.training_trials import extract_all 
    out_data, paths = extract_all(session_path_behav, save=True) 
    data_dict = out_data.copy() #behav data
    df_alldata = pd.DataFrame.from_dict(data_dict["table"]) #table to use
    
    data_wheel = pd.DataFrame.from_dict(data_dict["wheel_timestamps"]) #wheel timestamps data 
    data_wheel.rename(columns = {0:'wheel_timestamps'}, inplace=True)
    data_wheel["wheel_position"] = out_data["wheel_position"] 
    data_wheel_intervals = pd.DataFrame.from_dict(data_dict["wheel_moves_intervals"][:,0]) 
    data_wheel_intervals.rename(columns = {0:'wheel_moves_intervals_start'}, inplace=True)
    data_wheel_intervals["wheel_moves_intervals_stop"]    = pd.DataFrame.from_dict(data_dict["wheel_moves_intervals"][:,1])
    data_wheel_moves_peak = pd.DataFrame.from_dict(data_dict["wheel_moves_peak_amplitude"]) 
    data_wheel_moves_peak.rename(columns = {0:'wheel_moves_peak_amplitude'}, inplace=True)
    data_wheel_moves_peak["peakVelocity_times"]    = pd.DataFrame.from_dict(data_dict["peakVelocity_times"])

    del out_data['table'] 
    del out_data['wheel_timestamps'] 
    del out_data['wheel_position'] 
    del out_data['wheel_moves_intervals'] 
    del out_data["wheel_moves_peak_amplitude"] 
    del out_data["peakVelocity_times"] 


    out_data_keys=[]
    for keys in out_data: 
        out_data_keys.append(keys) 
    
    df = df_alldata
    for i in range(len(out_data_keys)): 
        df.insert(loc=len(df.columns), column=out_data_keys[i], value=out_data[out_data_keys[i]])
    df_alldata = df 
    #! CREATE A SAVE CSV FILE FOR THE BEHAVIOR 
    return df_alldata 


def extract_behav_b(session_path_behav): 
    from ibllib.io.extractors.biased_trials import extract_all 
    out_data, paths = extract_all(session_path_behav, save=True) 
    data_dict = out_data.copy() #behav data
    df_alldata = pd.DataFrame.from_dict(data_dict["table"]) #table to use
    
    data_wheel = pd.DataFrame.from_dict(data_dict["wheel_timestamps"]) #wheel timestamps data 
    data_wheel.rename(columns = {0:'wheel_timestamps'}, inplace=True)
    data_wheel["wheel_position"] = out_data["wheel_position"] 
    data_wheel_intervals = pd.DataFrame.from_dict(data_dict["wheel_moves_intervals"][:,0]) 
    data_wheel_intervals.rename(columns = {0:'wheel_moves_intervals_start'}, inplace=True)
    data_wheel_intervals["wheel_moves_intervals_stop"]    = pd.DataFrame.from_dict(data_dict["wheel_moves_intervals"][:,1])
    data_wheel_moves_peak = pd.DataFrame.from_dict(data_dict["wheel_moves_peak_amplitude"]) 
    data_wheel_moves_peak.rename(columns = {0:'wheel_moves_peak_amplitude'}, inplace=True)
    data_wheel_moves_peak["peakVelocity_times"]    = pd.DataFrame.from_dict(data_dict["peakVelocity_times"])

    del out_data['table'] 
    del out_data['wheel_timestamps'] 
    del out_data['wheel_position'] 
    del out_data['wheel_moves_intervals'] 
    del out_data["wheel_moves_peak_amplitude"] 
    del out_data["peakVelocity_times"] 


    out_data_keys=[]
    for keys in out_data: 
        out_data_keys.append(keys) 
    
    df = df_alldata
    for i in range(len(out_data_keys)): 
        df.insert(loc=len(df.columns), column=out_data_keys[i], value=out_data[out_data_keys[i]])
    df_alldata = df 
    #! CREATE A SAVE CSV FILE FOR THE BEHAVIOR 
    return df_alldata 

def extract_behav_tm(session_path_behav): 
    data1 = raw.load_data(session_path_behav) #default is absolute, relative starts at 0
    df_data1 = pd.DataFrame(data1) 
    


# %%
#* contrasts column (negative values are when the stim appeared in the Left side) 
def all_contrasts(df_alldata): 
    df_alldata_2 = df_alldata.reset_index(drop=True)
    array1 = np.array(df_alldata_2["contrastLeft"])
    array3 = np.array(df_alldata_2["contrastRight"]) 
    df_alldata_2["allContrasts"] = 100
    for i in range(0,len(array1)): 
        if array1[i] == 0. or array1[i] == 0.0625 or array1[i] == 0.125 or array1[i] == 0.25 or array1[i] == 0.5 or array1[i] == 1.0: #edited with 0.5 included 20230814
            df_alldata_2["allContrasts"][i] = array1[i] * (-1)
        else: 
            df_alldata_2["allContrasts"][i] = array3[i]
    #array2 = pd.DataFrame(array1)
    #df_alldata_2["allContrasts"] = array2 
    return(df_alldata_2) 

# creating reaction and response time variables 
    #reaction = first mov after stim onset 
def new_time_vars(df_alldata,new_var="test",second_action="firstMovement_times",first_action = "stimOn_times"): 
    df = df_alldata
    df[new_var] = df[second_action] - df[first_action] 
    return df 

# splitting the new_time_vars into correct and incorrect in the df and plotting the histogram/density 
def new_time_vars_c_inc(df_alldata,new_var="reactionTime"): 
    new_var_c = str(new_var+"_c") 
    new_var_inc = str(new_var+"_inc")
    df_alldata[new_var_c] = np.nan
    df_alldata[new_var_inc] = np.nan
    for i in range(0,len(df_alldata[new_var])): 
        if df_alldata["feedbackType"][i] == 1: 
            df_alldata[new_var_c][i] = (df_alldata[new_var][i])
        else: 
            df_alldata[new_var_inc][i] = (df_alldata[new_var][i]) 
    print(new_var," mean time correct = ", np.mean(df_alldata[new_var_c]), " | mean time incorrect = ", np.mean(df_alldata[new_var_inc]))
    return df_alldata 

# # check behav plots #TODO 
def plot_check_behav(df_alldata): 
    b = df_alldata
    fig, axs = plt.subplots(3, 2,figsize=(12,10)) 

    axs[0, 0].plot(b.intervals_1-b.intervals_0, alpha=1, color = '#FFADAD') 
    axs[0, 0].set_title('Trial time (s)')
    axs[0, 0].set(xlabel="intervals_1 - intervals_0")

    axs[0, 1].hist(b.quiescence, bins = 10,alpha=1, color='#A0C4FF')
    axs[0, 1].set_title('Quiescence period (s)') 
    axs[0, 1].set(xlabel="quiescence") 

    axs[1, 1].hist(b.rewardVolume,alpha=1, color='#9BF6FF')
    axs[1, 1].set_title('Quiescence period (s)')
    axs[1, 1].set(xlabel="quiescence") 
    fig.tight_layout()

def plot_reactions_and_responses(df_alldata): 
    df_alldata = df_alldata
    fig, axs = plt.subplots(3, 2,figsize=(15,10)) 
    axs[0, 0].hist(df_alldata.reactionTime, bins = 50,alpha=0.8, color='#4B8EB3')
    axs[0, 0].set_title('Reaction times = 1stMov - stimOn')
    axs[0, 1].hist(df_alldata.reactionTime, bins = 1000,alpha=0.8,color='#4B8EB3') 
    axs[0, 1].set_title('Reaction times = 1stMov - stimOn zoomed in')
    axs[0, 1].set_xlim([0,1.2])
    axs[1, 0].hist(df_alldata.responseTime, bins = 50,alpha=0.8, color='#f9a620')
    axs[1, 0].set_title('Response times = response - stimOn')
    axs[1, 1].hist(df_alldata.responseTime, bins = 1000,alpha=0.8, color='#f9a620') 
    axs[1, 1].set_title('Response times = response - stimOn zoomed in')
    axs[1, 1].set_xlim([0,1.2])
    axs[2, 0].hist(df_alldata.responseTime_mov, bins = 50,alpha=0.8, color='#7cb518')
    axs[2, 0].set_title('Response times since mov = response - 1stmov')
    axs[2, 1].hist(df_alldata.responseTime_mov, bins = 1000,alpha=0.8, color='#7cb518') 
    axs[2, 1].set_title('Response times since mov = response - 1stmov zoomed in')
    axs[2, 1].set_xlim([0,1.2])
    for ax in axs.flat:
        ax.set(xlabel='time (s)') 
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()
    plt.show() 

def plot_reactions_and_responses_diff(df_alldata): 
    fig, axs = plt.subplots(3, 2,figsize=(15,10)) 
    axs[0, 0].hist(df_alldata.reactionTime_c, bins = 500,alpha=0.5)
    axs[0, 0].hist(df_alldata.reactionTime_inc, bins = 500,alpha=0.5,color='red')
    axs[0, 0].set_title('Reaction times = 1stMov - stimOn')
    axs[0, 0].set_ylim([0,5])
    axs[0, 1].hist(df_alldata.reactionTime_c, bins = 500,alpha=0.5)
    axs[0, 1].hist(df_alldata.reactionTime_inc, bins = 500,alpha=0.5,color='red')
    axs[0, 1].set_title('Reaction times = 1stMov - stimOn')
    axs[0, 1].set_xlim([0,1.2])
    axs[1, 0].hist(df_alldata.responseTime_c, bins = 500,alpha=0.5)
    axs[1, 0].hist(df_alldata.responseTime_inc, bins = 500,alpha=0.5,color='red')
    axs[1, 0].set_title('Response times = response - stimOn')
    axs[1, 0].set_ylim([0,5])
    axs[1, 1].hist(df_alldata.responseTime_c, bins = 500,alpha=0.5)
    axs[1, 1].hist(df_alldata.responseTime_inc, bins = 500,alpha=0.5,color='red')
    axs[1, 1].set_title('Response times = response - stimOn')
    axs[1, 1].set_xlim([0,1.2])
    axs[2, 0].hist(df_alldata.responseTime_mov_c, bins = 500,alpha=0.5)
    axs[2, 0].hist(df_alldata.responseTime_mov_inc, bins = 500,alpha=0.5,color='red')
    axs[2, 0].set_title('Response times from mov to choice = response - 1stMov')
    axs[2, 0].set_ylim([0,5])
    axs[2, 1].hist(df_alldata.responseTime_mov_c, bins = 500,alpha=0.5)
    axs[2, 1].hist(df_alldata.responseTime_mov_inc, bins = 500,alpha=0.5,color='red')
    axs[2, 1].set_title('Response times from mov to choice = response - 1stMov')
    axs[2, 1].set_xlim([0,1.2])
    for ax in axs.flat:
        ax.set(xlabel='time (s)') 
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    fig.tight_layout()
    plt.show()


#plots #updated 20230605
def show_plot(df_alldata): 
    reactionTime_c_a = np.array(df_alldata.reactionTime_c)
    reactionTime_inc_a = np.array(df_alldata.reactionTime_inc)
    responseTime_c_a = np.array(df_alldata.responseTime_c)
    responseTime_inc_a = np.array(df_alldata.responseTime_inc)
    responseTime_mov_c_a = np.array(df_alldata.responseTime_mov_c)
    responseTime_mov_inc_a = np.array(df_alldata.responseTime_mov_inc)
    mean_A_c = np.nanmean(reactionTime_c_a)
    mean_A_inc = np.nanmean(reactionTime_inc_a)
    mean_B_c = np.nanmean(responseTime_c_a) 
    mean_B_inc = np.nanmean(responseTime_inc_a) 
    mean_C_c = np.nanmean(responseTime_mov_c_a)
    mean_c_inc = np.nanmean(responseTime_mov_inc_a)
    std_A_c = np.nanstd(reactionTime_c_a)
    std_A_inc = np.nanstd(reactionTime_inc_a)
    std_B_c = np.nanstd(responseTime_c_a) 
    std_B_inc = np.nanstd(responseTime_inc_a) 
    std_C_c = np.nanstd(responseTime_mov_c_a)
    std_c_inc = np.nanstd(responseTime_mov_inc_a)
    materials = ['rT_c','rT_inc',' ','rsT_c','rsT_inc',' ','rsT_m_c','rsT_m_inc']
    x_pos = np.arange(len(materials)) 
    CTEs = [round(mean_A_c,2),round(mean_A_inc,2),np.nan,round(mean_B_c,2),round(mean_B_inc,2),np.nan,round(mean_C_c,2),round(mean_c_inc,2)] 
    error = [std_A_c,std_A_inc,np.nan,std_B_c,std_B_inc,np.nan,std_C_c,std_c_inc]
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'left', size = 15)
    c = ['#C1D3FE', '#FFB9AA', 'red', '#ABD9D9', '#FFC19F', 'red', '#ACD8AA', '#FFCB9E']
    # Build the plot
    sns.set(style = "ticks") 
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.95, color=c, capsize=10) #ecolor=c, 
    ax.set_ylabel('Mean values for correct and incorrect', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials,fontsize=13)
    addlabels(materials, CTEs)
    ax.set_title('ReactionTime | ResponseTime | ResponseTime from 1stmov', fontsize=15)
    ax.set_ylim(0)
    ax.yaxis.grid(False) 
    sns.despine(left = False, bottom = False) 
    plt.tight_layout()
    plt.show() 

# #ALTERNATIVE
# import seaborn as sns
# sns.set_theme(style="whitegrid")
# df = pd.DataFrame(reactionTime_c_a,columns=["reactionTime_c"]) 
# df["reactionTime_inc"] = reactionTime_inc_a 
# df["responseTime_c"] = responseTime_c_a 
# df["responseTime_inc"] = responseTime_inc_a 
# df["responseTime_mov_c"] = responseTime_mov_c_a 
# df["responseTime_mov_inc"] = responseTime_mov_inc_a 
# penguins = df
# # Draw a nested barplot by species and sex
# g = sns.catplot(
#     data=penguins, kind="bar",
#     errorbar="sd", palette="pastel", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")



#%% check the difference between nph_sync and bpod_sync 
def plot_diff_nph_bpod_sync(nph_sync,bpod_sync): 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot((np.diff(nph_sync)/np.diff(bpod_sync)),alpha=0.8,color='#a8e6cf') 
    ax.axhline(y=np.mean((np.diff(nph_sync)/np.diff(bpod_sync))),color='#84e3c8', linewidth=3)
    plt.show()












# #%% 
# #===========================================================================
# #                   5. INTERPOLATE BPOD AND NPH TIME
# #===========================================================================


# """ 5.1 Check the same length of the events from BPOD (Behav) & TTL (Nph) """
# # load the bpod reward times - it was the TTL that was sent to the nph system 
# bpod_event = "stimOnTrigger_times" #"goCue_times" #"feedback_times"
# bpod_sync = np.array(df_alldata[bpod_event])#!USE THIS ONE 

# # load the TTL reward times - it was the TTL that was sent to the nph system
# nph_sync = np.array(df_raw_phdata_DI0_true["Timestamp"]) 
# #nph_sync_2 = 
# #to test if they have the same length 
# print(len(bpod_sync),len(nph_sync), len(nph_sync_2))
# print(nph_sync[-1]-nph_sync[-2],nph_sync[-2]-nph_sync[-3], nph_sync[-3]-nph_sync[-4]) 
# print(bpod_sync[-1]-bpod_sync[-2],bpod_sync[-2]-bpod_sync[-3], bpod_sync[-3]-bpod_sync[-4]) 

# #%% 
# #================================================
# #                Check which TTL timestamp to use
# #================================================
# """ 
# Documentation: https://docs.google.com/document/d/1hNmbNFN7LK8zif6w-JyHGl_ObBvH4L_8IzIjeblYueA/edit?usp=sharing 
# """
# #to get the Nph file data
# df_PhotometryData = pd.read_csv(session_path) #path to the PhotometryData csv file

# #to get the DI file data
# df_DI0 = pd.read_csv(io_path) #path to the DI csv file
# df_DI0['Value'] = df_DI0['Value.Value']
# df_DI0['Seconds'] = df_DI0['Value.Seconds']
# #select only the True values - when TTL was received 
# raw_phdata_DI0_true = df_DI0[df_DI0.Value==True] 
# df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
# df_raw_phdata_DI0_T_seconds = pd.DataFrame(raw_phdata_DI0_true, columns=["Seconds"])
# df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index()
# df_raw_phdata_DI0_T_seconds = df_raw_phdata_DI0_T_seconds.reset_index() 
# df_raw_phdata_DI0_T_timestamp["TTL"] = 1
# df_raw_phdata_DI0_T_seconds["TTL"] = 1
# #to plot 
# plt.rcParams["figure.dpi"] = 150
# plt.scatter(df_PhotometryData.Timestamp,df_PhotometryData.Input0, s=20, c='#007e5d',alpha=0.8) #Nph scatter
# plt.plot(df_PhotometryData.Timestamp,df_PhotometryData.Input0, c='#007e5d',alpha=0.8) #Nph line
# plt.scatter(df_raw_phdata_DI0_T_timestamp["Timestamp"],df_raw_phdata_DI0_T_timestamp["TTL"], s=20, c='#9767FF',alpha=0.8) #DI timestamp scatter 
# plt.scatter(df_raw_phdata_DI0_T_seconds["Seconds"],df_raw_phdata_DI0_T_seconds["TTL"], s=20, c='#FF6E46',alpha=0.8) #DI seconds scatter 
# plt.legend(["Nph file Input0", "Nph file Input0", "DI file Timestamp", "DI file Seconds"],loc='best',fontsize=7.5, frameon=False)
# #example xlim 
# plt.axvline(x=6012.45145, linewidth=0.75, c='#9767FF')
# plt.axvline(x=6012.4865, linewidth=0.75, c='#FF6E46')
# plt.suptitle("Comparison of the TTL timestamp values from 2 ≠ files, 3 ≠ columns")
# plt.title("cropped some example timestamps")
# plt.xlabel("Timestamp")
# plt.ylabel("TTL received = 1")
# plt.xlim(6012.35,6012.7)


# #================================================

# %%

#%%
""" 5.2 Assert & reVerify the length from BPOD (Behav) & TTL (Nph) """ 
def assert_length(x,y): 
    """
    If the length is different only by 1 value, run this function (after checking the difference of consecutive values to figure out if it's one more value at the end and not beginning)
    x = nph_sync
    y = bpod_sync
    """ 
    if len(x)-1 == len(y): 
        x = x[0:len(y)]
        print("Option 1: nph_sync had 1 more value")
    else: 
        print("Option 2: EQUAL VALUES OR SOMETHING IS WRONG!")
    return x 

def verify_length_2(x,y): 
    """
    If the length is different only by 1 value, run this function (after checking the difference of consecutive values to figure out if it's one more value at the end and not beginning)
    x = nph_sync
    y = bpod_sync
    """ 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 

#%% #! USING INSIDE THE OTHER FILE...  
# def interpolate_times(nph_sync,bpod_sync): 
#     """
#     matches the values
#     x and y should be the same length 
#     x = nph_sync 
#     y = bpod_sync
#     """
#     # df_raw_phdata_DI0_true = import_DI(io_path) 
#     # nph_sync = np.array(df_raw_phdata_DI0_true["Timestamp"]) 
#     # x=nph_sync
#     # y=bpod_sync

#     nph_to_bpod_times = interp1d(nph_sync, bpod_sync, fill_value='extrapolate') 
#     nph_to_bpod_times
#     nph_frame_times = df_PhotometryData['Timestamp']
#     frame_times = nph_to_bpod_times(nph_frame_times)   # use interpolation function returned by `interp1d`
#     plt.plot(nph_sync, bpod_sync, 'o', nph_frame_times, frame_times, '-')
#     plt.show() 
#     return frame_times
# %%
        




""" 
HOW TO GIT PULL AND USE "WITHOUT UPDATING EVERYTIME"
""" 
# (env) kceniabougrova@joana-ccu-pc:~/Documents/GitHub$ cd ibl-photometry/
# (env) kceniabougrova@joana-ccu-pc:~/Documents/GitHub/ibl-photometry$ pip install -e .

# (env) kceniabougrova@joana-ccu-pc:~/Documents/GitHub/ibl-photometry$ git pull
# Already up to date.
# (env) kceniabougrova@joana-ccu-pc:~/Documents/GitHub/ibl-photometry$ cd ibl-photometry/^C
# (env) kceniabougrova@joana-ccu-pc:~/Documents/GitHub/ibl-photometry$ pip install -e .^C