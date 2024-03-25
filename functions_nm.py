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
