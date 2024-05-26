import pandas as pd
from typing import List, Dict, Tuple
import ray
from ray.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from config import logger



def load_data(dataset_loc: str, num_samples: int = None, seed: int = 1325) -> Dataset:
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=seed)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    
    
def stratify_split_ds(ds: Dataset, test_size: float = 0.2, stratify: str = 'is_attributed', shuffle: bool = True, seed: int = 1325) -> Tuple[Dataset, Dataset]:
    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  
        """Naively split a dataframe into train and test splits.
        Add a column specifying whether it's the train or test split."""
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Filter by data points that match the split column's value
        and return the dataframe with the _split column dropped."""
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")  # group by each unique value in the column we want to stratify on
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")  # combine
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")  # combine

    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def add_hour_day_from_clicktime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the hour and day columns as integers from the click_time column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'click_time' column.

    Returns:
        pd.DataFrame: The input DataFrame with 'hour' and 'day' columns added as integers.
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


def add_next_click(df: pd.DataFrame, max_num_cats: int) -> pd.DataFrame:
    """ Adds the 'next_click' feature to a dataframe

    Args:
        df (pd.DataFrame): Input dataframe.  Copied - not changed.
        max_num_cats (int): Max number of categories in our hash.
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


def apply_transforms(ds: Dataset, config: Dict,
                transforms: List[str] = [add_hour_day_from_clicktime, add_groupby_user_features, add_next_click, log_bin_column]) -> Dataset:
    
    for transform in transforms:
        transform_name = transform.__name__
        params = config.get(transform_name, {})
        try:
            ds = ds.map_batches(transform, batch_format='pandas', fn_kwargs=params)
            logger.info(f"Applied transformation: {transform_name}")
        except Exception as e:
            logger.error(f"Error applying transformation {transform_name}: {e}")
            continue
    return ds