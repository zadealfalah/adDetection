import pytest
import numpy as np
import pandas as pd
from addetectionscripts import transforms


def test_add_hour_day_from_clicktime(sample_data):
    X, _ = sample_data
    X.drop(columns=["hour", "day"], inplace=True)  # Drop the hour, day columns
    result_df = transforms.add_hour_day_from_clicktime(X)

    assert result_df.shape == (2, 8)
    assert result_df["hour"].dtype == "uint8"
    assert result_df["day"].dtype == "uint8"


def test_add_groupby_user_features(sample_data):
    X, _ = sample_data
    grouping_categories = [["ip", "channel"], ["ip", "day", "hour"], ["ip", "device", "os", "app"]]
    grouping_functions = ["nunique", "cumcount", "sum"]  # sum should raise ValueError
    result_df = transforms.add_groupby_user_features(X, grouping_categories, grouping_functions[:2])
    assert result_df.shape[1] == X.shape[1] + (len(grouping_categories) * len(grouping_functions[:2]))

    # Test that ValueError is raised for unsupported function
    with pytest.raises(ValueError):
        transforms.add_groupby_user_features(X, grouping_categories, ["sum"])


# This test is pretty useless as is
# All the things we bin are created features.
# Try to think of what would be a better test of the function.
# def test_log_bin_column(sample_data):
#     X, _ = sample_data
#     # Not actually ones binned in program, but those are created cols.  Check math at least.
#     cols_to_bin = ['click_time']
#     result_df = transforms.log_bin_column(X, cols_to_bin)
#     assert (result_df['click_time'] == np.log2(1 + X['click_time'].values).astype(int)).all()


def test_add_next_click(sample_data):
    X, _ = sample_data
    result_df = transforms.add_next_click(X)

    # No sequence of clicks at the moment, just test if it adds the column and is the proper dtype at edges
    assert "next_click" in result_df.columns
    assert result_df["next_click"].dtype == np.int64
