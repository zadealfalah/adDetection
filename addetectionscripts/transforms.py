import pandas as pd
from typing import List, Tuple, Optional, Dict
import numpy as np
from xgboost import DMatrix
from sklearn.model_selection import train_test_split
from config import transformations_config, logger


def init_datasets(data_folder: str = "datasets", to_load: List[str] = ["X_us", "y_us", "test"]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads the split CSV files into pandas DataFrames.

    Example usage:
        X_us, y_us, test = init_dataset('raw_data')

    Args:
        data_folder (str): Path to the folder containing the CSV files.
        to_load (List[str]): List of strings for which datasets to load. Defaults to all three (X_us, y_us, test).

    Returns:
        tuple: A tuple containing three pandas DataFrames: (X_us, y_us, test).
    """
    # Read CSV files into pandas DataFrames
    X_us, y_us, test = None, None, None
    if "X_us" in to_load:
        X_us = pd.read_csv(f"../{data_folder}/X_us.csv")
        X_us["click_time"] = pd.to_datetime(X_us["click_time"])
    if "y_us" in to_load:
        y_us = pd.read_csv(f"../{data_folder}/y_us.csv")
    if "test" in to_load:
        test = pd.read_csv(f"../{data_folder}/test.csv")
        test["click_time"] = pd.to_datetime(test["click_time"])

    return X_us, y_us, test


def add_hour_day_from_clicktime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the hour and day columns as integers from the click_time column.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'click_time' column.

    Returns:
        pd.DataFrame: The input DataFrame with 'hour' and 'day' columns added as integers.
    """
    df["hour"] = pd.to_datetime(df["click_time"]).dt.hour.astype("uint8")
    df["day"] = pd.to_datetime(df["click_time"]).dt.day.astype("uint8")
    return df


def add_groupby_user_features(df: pd.DataFrame, grouping_categories: List[List[str]], grouping_functions: List[str]) -> pd.DataFrame:
    """Takes an input dataframe, list of groupings to use, and a list of grouping functions (currently just allows for nunique and/or cumcount).
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
            if grouping_function == "nunique":
                grp = (
                    df[u_list]
                    .groupby(by=u_list[0 : len(u_list) - 1])[u_list[len(u_list) - 1]]
                    .nunique()
                    .reset_index()
                    .rename(index=str, columns={u_list[len(u_list) - 1]: new_col_name})
                )
                df = df.merge(grp, on=u_list[0 : len(u_list) - 1], how="left")
            elif grouping_function == "cumcount":
                grp = df[u_list].groupby(by=u_list[0 : len(u_list) - 1])[u_list[len(u_list) - 1]].cumcount()
                df[new_col_name] = grp.values
            else:
                raise ValueError(f"That grouping function {grouping_function} is not currently supported.  Use nunique and/or cumcount.")
    return df


def log_bin_column(df: pd.DataFrame, collist: List[str]) -> pd.DataFrame:
    """Log bins the feature columns given in collist

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
    """Adds the 'next_click' feature to a dataframe

    Args:
        df (pd.DataFrame): Input dataframe.  Copied - not changed.
        max_num_cats (int): Max number of categories in our hash.  Defaults to 2**26.
    Returns:
        pd.DataFrame: Copy of the input dataframe with the 'next_click' feature added.
    """

    max_num_categories = max_num_cats
    df["user_hash"] = (df["ip"].astype(str) + "_" + df["app"].astype(str) + "_" + df["device"].astype(str) + "_" + df["os"].astype(str)).apply(
        hash
    ) % max_num_categories
    click_buffer = np.full(max_num_categories, 3000000000, dtype=np.uint32)
    df["epoch_time"] = df["click_time"].astype(np.int64) // 10**9  # Get epoch time of each click

    next_clicks = []  # Empty list to be filled for next click by user hash
    # This loop goes backwards through each user by time, gets the time of their next click
    for userhash, time in zip(reversed(df["user_hash"].values), reversed(df["epoch_time"].values)):
        next_clicks.append(click_buffer[userhash] - time)
        click_buffer[userhash] = time
    # Since we went through backwards, reverse the next clicks and add it as a column
    df["next_click"] = list(reversed(next_clicks))

    # Last clicks in each user hash have high values as we'll do 3000000000 - (click_time) so we need to address this.
    # We'll write a function to log-bin features. Separate as we want it for other columns too.  Use it separately for better testing
    # df = log_bin_column(df, ['next_click'])

    return df


### Should separate the dropping of click time in to its own function.
### Can call it drop_columns(collist) or something similar, for pytests
def apply_transformations(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    config: Dict = transformations_config,
    transforms: List[str] = [add_hour_day_from_clicktime, add_groupby_user_features, add_next_click, log_bin_column],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies a list of transforms with their corresponding parameters.
    Also drops the initial click_time feature.

    Args:
        X_df (pd.DataFrame): Input X dataframe to transform
        y_df (pd.DataFrame): Input y dataframe to transform.  Currently not changed.
        config (Dict, optional): Configuration parameters for the given transforms. Defaults to transformations_config.
        transforms (List[str], optional): List of transforms to apply. Defaults to [add_hour_day_from_clicktime, add_groupby_user_features, add_next_click, log_bin_column].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The now transformed X and y dataframes.
    """
    Xdf = X_df.copy()
    ydf = y_df.copy()

    for transform in transforms:
        transform_name = transform.__name__
        transform_params = config.get(transform_name, {})
        try:
            Xdf = transform(Xdf, **transform_params)
            logger.info(f"Applied transformation: {transform_name} \n" f"Using params {transform_params}")
        except Exception as e:
            logger.error(f"Error applying transformation: {transform_name} \n" f"Using params {transform_params}")
            continue
    Xdf.drop(columns=["click_time"], inplace=True)  # Drop the original click_time feature
    return Xdf, ydf


def split_final_datasets(
    X_df: pd.DataFrame, y_df: pd.DataFrame, random_state: int = 1235, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[DMatrix], Optional[DMatrix]]:
    """
    Takes the datasets after their transformations and splits them to training and validation sets.
    Returns the - now split - X and y data, as well as the XGBoost DMatrices for the training and validation sets

    Args:
        X_df (pd.DataFrame): Transformed X dataframe
        y_df (pd.DataFrame): Transformed y dataframe
        random_state (int, optional): Random state for splitting. Defaults to 1235.
        test_size (float, optional): Fraction of dataframe to split to validation set. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[DMatrix], Optional[DMatrix]]: The now split X and y dataframes as well as their corresponding XGBoost DMatrices
    """
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, random_state=random_state, test_size=test_size)
    dtrain = DMatrix(X_train, label=y_train)
    dvalid = DMatrix(X_val, label=y_val)

    return X_train, X_val, y_train, y_val, dtrain, dvalid
