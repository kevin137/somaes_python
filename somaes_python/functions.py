#!/usr/bin/env python
# coding: utf-8

# Proyecto Final - Python - Software matemático y estadístico
# Kevin Cook, 2024-11-03

"""
functions.py
A collection of functions created for Software matemático y estadístico
"""

__all__ = ['discretizeEW', 'discretizeEF', 
           'discretize_EW_by_column', 'discretize_EF_by_column',
           'discretize_EW_EF_by_column',
           'calculate_variance', 'calculate_roc_auc', 'calculate_roc_auc',
           'dataset_metrics_summary', 'select_variables_by_metrics',
           'normalize_by_column', 'standardize_by_column',
           'correlation', 'mutual_information',
           'column_relationships', 'plot_relationships',
           'plot_roc_auc'
           ]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### Discretization algorithms

# ----------------------------------------------------------------------------
#   Infrastructure and helper functions for discretization algorithms
# ----------------------------------------------------------------------------

# Helper function to create names for factors based on cut_points
def create_factor_names(cut_points):
    c = cut_points
    names = [f"I1:(-Inf,{round(c[0], 3)})"]  # The first bin is special
    for i in range(1, len(c)):
        names.append(f"I{i+1}:({round(c[i-1], 3)},{round(c[i], 3)})")
    names.append(f"I{len(c)+1}:({round(c[-1], 3)},+Inf)")  # last bin

    return names

# Helper function to categorize individual values based on cut points
def categorize(value, cut_points, factor_names):
    if value < cut_points[0]:              # first bin
        return factor_names[0]
    elif value >= cut_points[-1]:          # last bin 
        return factor_names[-1]
    for i in range(1, len(factor_names)):  # middle bins
        if cut_points[i-1] <= value < cut_points[i]:
            return factor_names[i]
    return None


# ----------------------------------------------------------------------------
#   Equal-width discretization algorithm, vector version
# ----------------------------------------------------------------------------

def discretizeEW(x, num_bins):
    '''Discretize numerical vector into categorical with bins of 
        equal width'''

    #    min                      max 
    #     |    |    |    |    |    |
    #  I1------| I2 | I3 | ...|------I{num_bins} 
    #          |              | 
    #          ^cut_points[0] | 
    #                         ^cut_points[num_bins-1]  

    bin_width = (max(x)-min(x))/num_bins  # divide span by number of bins
    cut_points = min(x) + np.cumsum(np.ones(num_bins-1)*bin_width)

    # Make factor names
    factor_names = create_factor_names(cut_points)
        
    # Apply categorize() to each value in x using the list(map()) idiom
    x_discretized = list(map(
        lambda val: categorize(val, cut_points, factor_names), x)
    )

    return (x_discretized, cut_points)


# ----------------------------------------------------------------------------
#   Equal-frequency discretization algorithm, vector version
# ----------------------------------------------------------------------------

def discretizeEF(x, num_bins):
    '''Discretize numerical vector into categorical with bins of 
        equal frequency'''

    x_ordered = np.sort(x)  # sort input into ascending order for later use

    # Create num_bins bins with an equal number of members with floor division 
    bin_populations = np.full(num_bins,(x.shape[0])//num_bins)

    # Unless the size of x happens to be an exact multiple of the number 
    #   of bins, there will be leftovers from the floor division, we retrieve
    #   the number of leftovers with the modulo and assign then to random bins    
    leftovers = np.random.choice(np.arange(num_bins), 
                                 size=(x.shape[0])%num_bins, 
                                 replace=False) 
    bin_populations[leftovers] += 1

    # Split the vector into bins of size bin_populations
    last_index_this_bin = np.cumsum(bin_populations)
    first_index_next_bin = np.concatenate([[0], last_index_this_bin[:-1]+1])
    end_of_bin = x_ordered[last_index_this_bin[:-1]-1]
    beginning_of_bin = x_ordered[first_index_next_bin[1:]-1]
    # Take cut points at the arithmetric mean of the endpoints of the bins
    cut_points = end_of_bin + (beginning_of_bin-end_of_bin)/2

    # Make factor names with utility function from above
    factor_names = create_factor_names(cut_points)
    
    # Apply categorize() to each value in x using the list(map()) idiom
    x_discretized = list(map(
        lambda val: categorize(val, cut_points, factor_names), x)
    )

    return (x_discretized, cut_points)


# ----------------------------------------------------------------------------
#   Equal-width discretization algorithm, multi-column dataframe version
# ----------------------------------------------------------------------------

def discretize_EW_by_column(df, num_bins, keep_original=False):
    '''Discretize input dataframe column by column with into num_bins bins of 
        equal width and return in a new dataframe. Set keep_original=True to 
        keep the input columns'''

    # Using apply and lambda to apply EW discretization column by column
    ew = df.apply(  
        lambda col: discretizeEW(np.array(col.tolist()), num_bins)[0]
    )
    
    # Combine (optionally) the original df and discretization into a new df
    ret_df = pd.DataFrame()
    for col in df.columns:
        if keep_original:
            ret_df[f'{col}'] = df[col] 
        ret_df[f'{col}_EW'] = ew[col]
    
    return ret_df


# ----------------------------------------------------------------------------
#   Equal-frequency discretization algorithm, multi-column dataframe version
# ----------------------------------------------------------------------------

def discretize_EF_by_column(df, num_bins, keep_original=False):
    '''Discretize input dataframe column by column with into num_bins bins 
        with an approximately equal number of members and return in a new 
        dataframe. Set keep_original=True to keep the input columns'''

    # Using apply and lambda to apply EF discretization column by column
    ef = df.apply(  
        lambda col: discretizeEF(np.array(col.tolist()), num_bins)[0]
    )
    
    # Combine (optionally) the original df and discretization into a new df
    ret_df = pd.DataFrame()
    for col in df.columns:
        if keep_original:
            ret_df[f'{col}'] = df[col] 
        ret_df[f'{col}_EF'] = ef[col]
    
    return ret_df


# ----------------------------------------------------------------------------
#   Compare results of equal-width and equal-frequecy with original columns
# ----------------------------------------------------------------------------

def discretize_EW_EF_by_column(df, num_bins):
    '''Discretize column by column using discretizeEF() and discretizeEW() and 
        combine with input dataframe into a new dataframe.'''

    ef = df.apply(  # Apply equal frequency column by column with lambda
        lambda col: discretizeEF(np.array(col.tolist()), num_bins)[0]
    )
    ew = df.apply(  # Apply equal width discretization column by column
        lambda col: discretizeEW(np.array(col.tolist()), num_bins)[0]
    )
    
    # Combine the original df and discretization results into a new DataFrame
    ret_df = pd.DataFrame()
    for col in df.columns:
        ret_df[f'{col}'] = df[col]
        ret_df[f'{col}_EF'] = ef[col]
        ret_df[f'{col}_EW'] = ew[col]
    
    return ret_df


### Calculation of metrics for the attributes of a dataset

## Supporting functions for continuous variable metrics

# ----------------------------------------------------------------------------
#   Calculation of variance by column from a dataframe 
# ----------------------------------------------------------------------------

def calculate_variance(df, sample=True):
    '''Calculate variance by column from a dataframe with numeric values. 
        If the option sample=True (the default) is passed,
        the sample variance is returned, otherwise the population variance. 
        Returns a numpy array with the selected variance for each column'''

    # Check to be sure that the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("error, input must be a DataFrame")

    # Check to be sure that all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise TypeError("error, all columns must be numeric")

    # We'll do all our math in numpy
    x_cols = df.to_numpy()
    N = x_cols.shape[0]
    ddof = 1 if sample else 0

    # Population variance is defined as:
    #  
    #                1        
    #   variance =  ——— · sum_i=1_N( (x_i - x̄)^2 )
    #                N      
    #
    # Sample variance is defined as:
    #  
    #                1        
    #   variance =  ——— · sum_i=1_N( (x_i - x̄)^2 )
    #               N-1      
    #

    # Calculate sample variance for each column
    x_bar_cols = np.mean(x_cols, axis=0)  # mean value of each column
    variance = (1/(N-ddof)) * np.sum((x_cols - x_bar_cols) ** 2, axis=0) 

    return variance


# ----------------------------------------------------------------------------
#   ROC (AUC) calculation from two-column dataframe 
# ----------------------------------------------------------------------------

def calculate_roc_auc(df):
    '''Calculate receiver operating characteristic (ROC) curve and 
        accompanying area under curve (AUC) value for a variable (instead of 
        the more typical use with full classifier), given an input dataframe 
        with numeric (decimal) values in the first column, and booleans in the 
        second.'''
    
    # Ensure the input is a DataFrame with two columns
    if not isinstance(df, pd.DataFrame) or df.shape[1] != 2:
        raise ValueError('error, input must be a DataFrame with two columns')

    # Ensure that the first column is numeric
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        raise TypeError('error, the first column must be numeric')

    # Ensure that the second column is boolean
    if not pd.api.types.is_bool_dtype(df.iloc[:, 1]):
        raise TypeError('error, the second column must be boolean')

    unordered_numeric_values = (df[df.columns[0]]).to_numpy() 
    unordered_boolean_ground_truth = (df[df.columns[1]]).to_numpy() 

    #print('unordered_numeric_values: ', unordered_numeric_values)
    #print('unordered_boolean_ground_truth: ', unordered_boolean_ground_truth)

    # Processing the threshold values make more sense if they are in order, 
    #   and we want FPR==0.0 at the beginning, and FPR==1.0 at the end of the 
    #   array, so we need to put the numeric input values in _INVERSE_ order.
    ordered_indices = np.argsort(-unordered_numeric_values)

    # Candidate thresholds for ROC are where values exist in the ground truth, 
    #   according to the problem statement, we'll use our ordered indices
    candidate_thresholds = unordered_numeric_values[ordered_indices]
    ground_truth = unordered_boolean_ground_truth[ordered_indices]

    # Calculate total number of positives and negatives in ground truth
    P = np.sum(ground_truth)
    N = len(ground_truth) - P

    # We store one extra point in the output curve (1,1), to make the 
    #   canonical ROC graph and ensure that the AUC integral includes it   
    num_curve_points = ground_truth.size + 1
    fpr_curve,tpr_curve=np.zeros(num_curve_points),np.zeros(num_curve_points)
    fpr_curve[-1], tpr_curve[-1] = 1, 1

    # We need all the candidate thresholds for the curve and the integral
    for i, c in enumerate(candidate_thresholds):
        #print('\n',i,'\n',c,)
        predicted = candidate_thresholds > c
        #print('predicted: ', predicted)
        
        TP = np.sum(ground_truth & predicted)
        #TN = np.sum(~ground_truth & ~predicted)  # unneeded because we have N
        FP = np.sum(~ground_truth & predicted)    
        #FN = np.sum(ground_truth & ~predicted)   # unneeded because we have P
        #TPR = TP/(TP+FN)  # unneeded, can just use P instead
        #FPR = FP/(FP+TN)  # unneeded, can just use N instead
        tpr_curve[i] = TP/P  # TPR = TP/(TP+FN) = TP/P
        fpr_curve[i] = FP/N  # FPR = FP/(FP+TN) = FP/N

    # Integrate under the curve with np.trapezoid()
    AUC = np.trapezoid(tpr_curve, fpr_curve)

    return (fpr_curve, tpr_curve, AUC)


## Supporting functions for discrete variable metrics

# ----------------------------------------------------------------------------
#   Shannon entropy calculation, dataframe version
# ----------------------------------------------------------------------------

def calculate_entropy(df):
    '''Calulate Shannon entropy of a single dataframe column, note that all 
        data types are interpreted as categorical. Returns numpy decimal'''

    # Ensure the input is a DataFrame with one column
    if not isinstance(df, pd.DataFrame) or df.shape[1] != 1:
        raise ValueError('error, input must be a DataFrame with one column')

    # Use Python sets as a shortcut to count unique values
    vector = (df.iloc[:, 0]).tolist()
    unique = set(vector)
    if len(unique) == 0:
        return None  # entropy is undefined in this case
    
    # Calculate the probability for each unique value 
    P = [(vector.count(value))/len(vector) for value in unique]
    
    # Calculate the entropy according to H(X) = -sum(p_i * log2(p_i))
    H = np.sum([-p_i * np.log2(p_i) for p_i in P if p_i > 0])
    
    return H


# ----------------------------------------------------------------------------
#   Extract metrics of the appropriate type from any column in a dataframe
# ----------------------------------------------------------------------------

def extract_dataset_metrics(df):
    '''Extract the implemented metrics (entropy, variance, auc) from 
        compatible columns of the input dataframe. Returns tuple of three  
        Python lists for each metric (entropy, variance, auc), each containing 
        tuples with:
            column name (the pandas column name)
            metric as calculated by the helper functions

        Note that for a column to have the Area Under Curve (auc) metric  
        calculated, it must be a numeric (float) column with a boolean column 
        immediately to its right. Booleans that have been used for AUC are 
        ignored for for further metric. Booleans that have NOT been used for 
        AUC are considered categorical data.'''
    
    # lists for the return start out empty
    entropy_list, variance_list, auc_list = [], [], []

    # cycle through columns of input dataframe
    for i, c in enumerate(df.columns):  # note that we need i for the AUC
        
        col_type = df[c].dtype
        if col_type=='object' or col_type=='string' or col_type=='int64':
            # We consider these categorical, get entropy
            entropy_list.append((c, float(calculate_entropy(df[[c]]))))
        
        elif col_type == 'bool':  # bools can be used for AUC, or categorical
            if ((i-1) >= 0) and df.iloc[:,(i-1)].dtype != 'float64':
                # If column to the left is NOT numeric, also get entropy
                entropy_list.append((c, float(calculate_entropy(df[[c]]))))
        
        elif col_type == 'float64':
            # Numeric, get variance
            col_variance = calculate_variance(df[[c]]).item()
            variance_list.append((c, float(col_variance)))

            # Check the column to the right, if it is boolean...
            if ((i+1) < len(df.columns)) and df.iloc[:,(i+1)].dtype == 'bool':
                # we can calculate the AUC
                _, _, col_auc = calculate_roc_auc(df.iloc[:,i:(i+2)])
                auc_list.append((c, float(col_auc)))
        else:
            # Print an informational message, but do NOT dump out
            print('unknown column type at', c)

    return (entropy_list, variance_list, auc_list)


# ----------------------------------------------------------------------------
#   Generate summary dataframe with the appropriate metrics per column
# ----------------------------------------------------------------------------

def dataset_metrics_summary(df, display_precision=3, append_input=False):
    '''Generate a summary dataframe with the implemented metrics (entropy, 
        variance, auc) from compatible columns of the input dataframe. Returns 
        a dataframe populated with text, with the number of significant digits 
        set by display_precision. Include append_input=True to include the 
        input dataframe in the output.'''
    
    dp = display_precision

    # Initialize return dataframe with metric types as rows
    ret_df = pd.DataFrame( { 'dummy': [None,   None,       None,  None]    },
                              index=['(type)', 'variance', 'AUC', 'entropy'] )
    
    # Extract metrics from the input using extract_dataset_metrics()
    e_list, v_list, a_list = extract_dataset_metrics(df)
    
    # Populate the output dataframe column by column
    for c in df.columns:
        col_type = df[c].dtype
        
        # Initialize column with type
        col = [col_type, '', '', ''] 
        
        # Populate variance, AUC, and entropy from the extracted metrics
        if col_type == 'float64':
            col[1] = next((round(v[1], dp) for v in v_list if v[0] == c), '')
            col[2] = next((round(a[1], dp) for a in a_list if a[0] == c), '')
        if col_type in ['object', 'string', 'int64', 'bool']:
            col[3] = next((round(e[1], dp) for e in e_list if e[0] == c), '')

        # Add column to ret_df
        ret_df = ret_df.assign(**{c: col})
    
    # Drop dummy column
    ret_df.drop('dummy', axis=1, inplace=True)
    
    # Replace None with blanks for a cleaner summary
    ret_df.replace({None: ''}, inplace=True)
    
    # Tack the original dataframe onto the end if requested 
    if append_input:
        ret_df = pd.concat([ret_df, df], axis=0, ignore_index=False)

    return ret_df


### Normalization and standardization of variables

# ----------------------------------------------------------------------------
#   Normalize dataframe by column 
# ----------------------------------------------------------------------------

def normalize_by_column(df):

    # Verify that the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('error, the input must be a Dataframe')

    # Verify that everything is numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError('error, all columns must be numeric')

    # Apply the normalization:
    #   distance_from_minimum_value/distance_from_max_to_min_value
    #   column by column using lambda
    return df.apply(
        lambda col: (col - np.min(col))/(np.max(col) - np.min(col)), 
        axis=0
    )

# ----------------------------------------------------------------------------
#   Standardize dataframe by column 
# ----------------------------------------------------------------------------

def standardize_by_column(df):

    # Verify that the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('error, the input must be a Dataframe')

    # Verify that everything is numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError('error, all columns must be numeric')

    # Apply standardization:
    #   distance_from_column_mean/column_stddev (to achieve mean 0, stddev 1)
    #   column by column using lambda
    return df.apply(
        lambda col: (col - np.mean(col))/np.std(col), 
        axis=0
    )


### Filtering of variables based on the implemented metrics 

# ----------------------------------------------------------------------------
#   Select variables from dataframe based on rules and metrics
# ----------------------------------------------------------------------------

def select_variables_by_metrics(df, rules):
    '''Create a new dataframe by selecting variables/columns/features from an 
        input dataframe whose metrics comply with a set of rules. Returns a 
        numpy dataframe with a subset of the columns of the input. Mandatory 
        parameter rules is a list of tuples defining rules for filtering. Each 
        tuple contains:
            metric (str):   'entropy', 'variance', 'auc'.
            relation (str): '>=', '<=', '=='.
            cutoff (float): comparison value for the metric selected'''
    
    # First we extract the implemented metrics using extract_dataset_metrics() 
    entropy_list, variance_list, auc_list = extract_dataset_metrics(df)
    
    # Now bundle them up into a dictionary with the metric names as keys
    metrics = { 'entropy':  dict(entropy_list),
                'variance': dict(variance_list),
                'auc':      dict(auc_list)       }
    
    # We'll use Python's built-in set functionality to remember the columns
    selected_columns = set(df.columns)
    
    # Processing each rule from the list
    for metric, relation, cutoff in rules:

        metric_values = metrics.get(metric, {})  # retrieve if present
        
        # Refresh a set for the columns that satisfy this particular rule
        columns_complying_with_this_rule = set()
        
        for col, value in metric_values.items():
            # Apply the specified relation
            if ((relation == '>=' and value >= cutoff) or
                (relation == '<=' and value <= cutoff) or
                (relation == '==' and value == cutoff)):
                columns_complying_with_this_rule.add(col)
        
        # Combine and update
        selected_columns &= columns_complying_with_this_rule

    # Return dataframe picks columns out input datafrom with df.loc
    return df.loc[:, list(selected_columns)]


### Calculation of the correlation/mutual information

## Functions for correlation for continous variables 

# ----------------------------------------------------------------------------
#   Correlation calculation, vector version
# ----------------------------------------------------------------------------

def correlation(x, y):
    
    '''Calculate Pearson's correlation coefficient between the two numerical
        input vectors x and y, passed as numpy arrays. Returns numpy 
        float64.'''

    # All subsequent operations depend on equal length, so check to make sure 
    #   the lengths of the x and y arrays match
    if x.size != y.size:
        raise ValueError('error, numpy arrays x and y must be same length.')

    # (Pearson's) correlation coefficient (for samples or populations) is: 
    # 
    #       covariance(X,Y)                 sum( (X-X̄)·(Y-Ȳ) )
    # r = ——————————————————— = —————————————————————————————————————————————
    #     stddev(X)·stddev(Y)   sqrt( sum( (X-X̄)^2) ) · sqrt(sum( (Y-Ȳ)^2 ) )
    #
    # which for clarity and efficiency with numpy is better expressed as:
    #  
    #                   sum_i=1_N( (x_i-x̄)·(y_i-ȳ) )
    #   r = ————————————————————————————————————————————————————
    #       sqrt( sum_i=1_N( (x_i-x̄)^2) · sum_i=1_N( (y-ȳ)^2 ) )
    # 
    # Note: The above applies to both samples and populations--both the 
    #         covariance and product of standard deviations actually have 
    #         either N+1 or N in their denominators depending on whether they 
    #         are samples, or entire populations, but in any case both 
    #         denominators will be the same, and therefore cancel out. 

    # Get mean values for x and y
    x_bar, y_bar = np.mean(x), np.mean(y)
    
    # Get the numerator of the covariance of x and y
    covariance_numerator = np.sum((x-x_bar) * (y-y_bar))
    
    # Get the numerator of the product of the standard deviations of x and y
    stddevprod_numerator = np.sqrt(np.sum((x-x_bar)**2)*np.sum((y-y_bar)**2))
    
    # Check the stddevprod_numerator (for the denominator),
    if stddevprod_numerator == 0:  # if 0, the variability is also 0,
        return 0                   # and the last division would explode
    
    # Calculate correlation coefficient
    correlation = covariance_numerator / stddevprod_numerator
    
    return correlation


## Functions for mutual information for categorical variables

# ----------------------------------------------------------------------------
#   Mutual information calculation, vector version
# ----------------------------------------------------------------------------

def mutual_information(x, y):
    
    '''Calculate mutual information coefficient between the two numerical 
        input vectors x and y, passed in as numpy arrays. Returns numpy 
        float64.'''

    # To avoid problems with dtype=objects, we coerce the inputs into strings.
    x, y = np.array(x).astype(str), np.array(y).astype(str)

    # All subsequent operations depend on equal length, so check to make sure 
    #   the lengths of the x and y arrays match
    if x.size != y.size:
        raise ValueError('error, numpy arrays x and y must be same length.')

    # Mutual information is defined as:
    #  
    #                                  p(x,y)        
    #   I = sum_X( sum_Y( p(x,y)·log(———————————) ) ) 
    #                                 p(x)·p(y)      
    #
    #   to avoid looping the sums, we need p(x), p(y), p(x,y) as numpy arrays

    # Create pairs for subsequent uniquifying and counting
    pairs = np.column_stack((x, y))

    # Get joint probabilities p_xy by uniquifying and counting
    uniq_pairs, count_pairs = np.unique(pairs, axis=0, return_counts=True); 
    p_xy = count_pairs/len(x)

    # Get marginal probabilities p_x, p_y by uniquifying, counting, filtering
    uniq_x, count_x = np.unique(x, return_counts=True)
    x_i = np.array([np.where(uniq_x == pair[0])[0][0] for pair in uniq_pairs])
    uniq_y, count_y = np.unique(y, return_counts=True)
    y_i = np.array([np.where(uniq_y == pair[1])[0][0] for pair in uniq_pairs])
    p_x, p_y = count_x/len(x), count_y/len(y)

    # Calculate mutual information with:
    #  
    #                       p_xy       
    #   I = sum( p_xy·log(—————————) ) 
    #                      p_x·p_y     
    #

    I = np.sum( p_xy * np.log(p_xy / (p_x[x_i] * p_y[y_i])) )

    return I


## Exploring the relationships between columns/features/variables in a dataset

# ----------------------------------------------------------------------------
#   Relationships with correlation and mutual information, dataframe version
# ----------------------------------------------------------------------------

def column_relationships(df):
    '''Build tables exhibiting the relationships between variables in a 
        dataset, one for all numerical features using correlation, and another 
        for all categorical features using mutual information. Input is a 
        pandas dataframe with arbitrary columns, returns two dataframes.'''

    mi_vars, co_vars = [], []  # One list each for categorical and numeric

    # Check each column in the input, divide into columns to be included in 
    #   categorical cross-comparison and numerical cross-comparison
    for column in df.columns:
        col_type = df[column].dtype
        if (col_type=='object' or col_type=='string' or 
            col_type=='int64' or col_type == 'bool'):
            mi_vars.append(column)
        elif col_type == 'float64':
            co_vars.append(column)
        else:
            print(f"'{column}' has unsupported type {col_type}, skipping.")

    # Use Pandas for the square correlation and association grid dataframes
    correlation_df = pd.DataFrame(index=co_vars, columns=co_vars)
    association_df = pd.DataFrame(index=mi_vars, columns=mi_vars)

    for i in range(len(mi_vars)):
        for j in range(i, len(mi_vars)): 

            # Note that the mutual info H(x,x) SHOULD be the entropy, 
            #  but it seems to be distinct from Shannon entropy which
            #  does not seem to give the same results

            I = mutual_information(df[mi_vars[i]].to_numpy(), 
                                   df[mi_vars[j]].to_numpy())
            association_df.loc[mi_vars[i], mi_vars[j]] = I
            association_df.loc[mi_vars[j], mi_vars[i]] = I

    for i in range(len(co_vars)):
        for j in range(i+1, len(co_vars)):  # self-correlation is always 1 
            r = correlation(df[co_vars[i]].to_numpy(), 
                            df[co_vars[j]].to_numpy())
            correlation_df.loc[co_vars[i], co_vars[j]] = r
            correlation_df.loc[co_vars[j], co_vars[i]] = r
    np.fill_diagonal(correlation_df.values, 1)  # avoid needless calculation 

    return (correlation_df, association_df)


### Plots for the AUC and correlation/mutual information matrices

# ----------------------------------------------------------------------------
#   Plot ROC (receiver operating characteristic) curve and display AUC
# ----------------------------------------------------------------------------

def plot_roc_auc(roc_fpr, roc_tpr, auc, title='ROC Curve'):
    '''Draw canonical  heatmap to exhibit relationships between variables in a 
        dataset. Input is a square pandas dataframe. Nothing is returned.'''
    
    plt.clf()  # for hygiene and sanity
    
    # Use matplotlib to plot the ROC curve itself
    plt.plot(roc_fpr, roc_tpr, marker='o', color='mediumblue')
    
    # Paint the area under the curve
    plt.fill_between(roc_fpr, roc_tpr, alpha=0.3, color='lightblue')

    # Add the typical diagonal line representing random chance
    plt.plot([0, 1], [0, 1], 'r--')
    plt.text(0.1, 0.05, 'Random Chance', color='red', fontsize=10, ha='left')
    
    # Setup the axes, labels, and title
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])  # space for points at 0
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title + ', AUC = ' + str(round(auc, 4)))
    plt.grid()
    
    # Draw the plot to the output
    plt.show()


## Visualizing the relationships between features/variables in a dataset

# ----------------------------------------------------------------------------
#   Visualizing relationships with correlation and/or mutual information
# ----------------------------------------------------------------------------

def plot_relationships(df, title='Heatmap'):
    '''Draw triangular heatmap to exhibit relationships between variables in a 
        dataset. Input is a square pandas dataframe. Nothing is returned.'''
    
    if df.empty:  # Bravely refuse to do anything if no data was passed to us
        return
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)  # force numeric

    # If the matrix is symmetric about the diagonal, create a mask to suppress
    #   the repeated information the the bottom left portion of the graph
    mask = None
    if np.allclose(df, df.T, atol=1e-8): # then df is symmetric about diagonal
        mask = np.tril(np.ones_like(df, dtype=bool), k=-1)

    # Plot the heatmap using the heatmap graphic type from Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="coolwarm_r", center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                mask=mask)
    plt.title(title)
    plt.show()
