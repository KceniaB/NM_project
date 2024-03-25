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