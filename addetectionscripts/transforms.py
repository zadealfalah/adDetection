import pandas as pd
from typing import List
import numpy as np


def init_datasets(data_folder="datasets", to_load: List[str] = ['X_us', 'y_us', 'test']):
    """
    Reads the split CSV files into pandas DataFrames.
    
    Example usage:
        X_us, y_us, test = init_dataset('raw_data')

    Args:
    data_folder (str): Path to the folder containing the CSV files.
    to_load (List[str]): List of strings for which datasets to load.  Defaults to all three (X_us, y_us, test)
    
    Returns:
    tuple: A tuple containing three pandas DataFrames: (X_us, y_us, test).
    """
    # Read CSV files into pandas DataFrames
    X_us, y_us, test = None, None, None
    if 'X_us' in to_load:
        X_us = pd.read_csv(f'./{data_folder}/X_us.csv')
        X_us['click_time'] = pd.to_datetime(X_us['click_time'])
    if 'y_us' in to_load:
        y_us = pd.read_csv(f'./{data_folder}/y_us.csv')
    if 'test' in to_load:
        test = pd.read_csv(f'./{data_folder}/test.csv')
        test['click_time'] = pd.to_datetime(test['click_time'])
    

    return X_us, y_us, test



def add_hour_day_from_clicktime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the hour and day columns as ints from the click_time column
    Returns the input df with the hour, day columns added.  
    """
    df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df['click_time']).dt.day.astype('uint8')
    return df


def add_groupby_user_features(df: pd.DataFrame, grouping_categories: List[List[str]], grouping_functions: List[str]) -> pd.DataFrame:
    """ Takes an input dataframe, list of groupings to use, and a list of grouping functions (currently just allows for nunique and/or cumcount).
        Adds the grouped values to the input dataframe.  

    Args:
        df (pd.DataFrame): Input dataframe e.g. X_train
        grouping_categories (List[List[str]]): List containing lists of columns to group by as strings
        grouping_functions (List[str]): List containing strings of functions to aggregate with (must be nunique and/or cumcount at the moment)

    Returns:
        pd.DataFrame: Input dataframe with the new aggregated columns added on.
    """
    for u_list in grouping_categories:
        for grouping_function in grouping_functions:
            new_col_name = "_".join(u_list) + "_" + grouping_function
            if grouping_function == 'nunique':
                grp = df[u_list].groupby(by=u_list[0:len(u_list)-1])[u_list[len(u_list)-1]].nunique().reset_index().\
                    rename(index=str, columns={u_list[len(u_list)-1]:new_col_name})
                df = df.merge(grp, on=u_list[0:len(u_list)-1], how='left')
            elif grouping_function == 'cumcount':
                grp = df[u_list].groupby(by=u_list[0:len(u_list)-1])[u_list[len(u_list)-1]].cumcount()
                df[new_col_name] = grp.values
            else:
                raise ValueError(f"That grouping function {grouping_function} is not currently supported.  Use nunique and/or cumcount.")
    return df


def log_bin_column(df: pd.DataFrame, collist: List[str]) -> pd.DataFrame:
    """ Log bins the feature columns given in collist

    Args:
        df (pd.DataFrame): Input dataframe
        collist (List[str]): List of columns to log bin, as strings.

    Returns:
        pd.DataFrame: Input dataframe with the given columns log-binned.  
    """
    for col in collist:
        df[col] = np.log2(1 + df[col].values).astype(int)
    return df


def add_next_click(df: pd.DataFrame, max_num_cats: int = 2**26) -> pd.DataFrame:
    """ Adds the 'next_click' feature to a dataframe

    Args:
        df (pd.DataFrame): Input dataframe.  Copied - not changed.
        max_num_cats (int): Max number of categories in our hash.  Defaults to 2**26. 
    Returns:
        pd.DataFrame: Copy of the input dataframe with the 'next_click' feature added.
    """
    
    max_num_categories = max_num_cats
    df['user_hash'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
            + "_" + df['os'].astype(str)).apply(hash) % max_num_categories
    click_buffer = np.full(max_num_categories, 3000000000, dtype=np.uint32)
    df['epoch_time'] = df['click_time'].astype(np.int64) // 10**9 # Get epoch time of each click
    
    next_clicks = [] # Empty list to be filled for next click by user hash
    # This loop goes backwards through each user by time, gets the time of their next click
    for userhash, time in zip(reversed(df['user_hash'].values), reversed(df['epoch_time'].values)):
        next_clicks.append(click_buffer[userhash] - time)
        click_buffer[userhash] = time
    # Since we went through backwards, reverse the next clicks and add it as a column
    df['next_click'] = list(reversed(next_clicks))
    
    # Last clicks in each user hash have high values as we'll do 3000000000 - (click_time) so we need to address this. 
    # We'll write a function to log-bin features. Separate as we want it for other columns too.  Use it separately for better testing
    # df = log_bin_column(df, ['next_click'])
    
    return df
