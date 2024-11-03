#!/usr/bin/env python
# coding: utf-8

# Proyecto Final - Python - Software matemático y estadístico
# reference_datasets.py

import numpy as np
import pandas as pd

### Reference datasets

# Data from the titanic disaster
titanic = np.vstack((  np.array([['Female', 'Deceased']] * 89 ),
                       np.array([['Male',   'Deceased']] * 483), 
                       np.array([['Female', 'Survived']] * 230), 
                       np.array([['Male',   'Survived']] * 112)  ))
np.random.shuffle(titanic)   # Shuffle rows
titanic_df = pd.DataFrame(titanic, columns=['Sex', 'Condition'])

# Discretization example from tutorial P03_Funciones
p03_disc_values = np.array([11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4])
p03_disc_bins = 4

# Data known to have a sample variance of 23.5
sample_variance_23p5_df = pd.DataFrame({ 'data': [3.0, 4.0, 7.0, 12.0, 14.0],})

# Data known to have a population variance of 2.917
pop_variance_2p917_df = pd.DataFrame({ 'die': [1, 2, 3, 4, 5, 6] })

# Data known to give an area under curve (AUC) of 1.0 with ROC
auc_1p0_df = pd.DataFrame({
    'Score': [0.1, 0.4, 0.35, 0.8, 0.5],
    'Label': [False, True, False, True, True]
})

# Data known to give an area under curve (AUC) of 0.75 with ROC
auc_0p75_df = pd.DataFrame({
    'Score': [0.66, 0.09, 0.38, 0.27, 0.81, 0.44, 0.81, 0.81, 0.79, 0.43],
    'Label': [True,False, True,False, True, True,False, True, True,False]
})

# Data from tutorial P03, exercise 4, known to have entropy 0.971
p03_entropy_0p971 = np.array(['a', 'a', 'c', 'c', 'c'])

# Dataframe with mixed column type for column-wise testing
std_mixed_df = pd.DataFrame({
    'A': [1, 1, 1, 3, 3],
    'B': [True, True, True, False, False],
    'C': [4.5, 5.5, 6.5, 5.5, 6.5],
    'D': ['a', 'a', 'a', 'c', 'c'],
    'E': [0.1, 0.4, 0.35, 0.8, 0.5],
    'F': [False, True, False, True, True],
    'G': [4.5, 5.5, 6.5, 6.5, 6.5],
})

# Data known to give a correlation of -0.6847868
corr_m0p685_df = pd.DataFrame({
    'x': [   8,   3,   5,   7,   1,   2,   6,   7,   4,   9 ],
    'y': [ 2.0, 2.0, 1.5, 1.0, 2.5, 3.0, 1.5, 2.0, 2.0, 1.5 ]
})
