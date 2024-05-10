import pytest
import numpy as np
from conftest import sample_data
from data import add_hour_day_from_clicktime, add_groupby_user_features, log_bin_column, add_next_click

def test_add_hour_day_from_clicktime(sample_data):
    X, _ = sample_data
    X.drop(columns = ['hour', 'day'], inplace=True) # Drop the hour, day columns
    result_df = add_hour_day_from_clicktime(X)
    
    assert result_df.shape == (2, 8)
    assert 'hour' in result_df.columns
    assert 'day' in result_df.columns
    assert result_df['hour'].dtype == 'datetime64[ns]'
    assert result_df['day'].dtype == 'datetime64[ns]'
    
    
def test_add_groupby_user_features(sample_data):
    X, _ = sample_data
    grouping_categories = [
        ['ip', 'channel'],
        ['ip', 'day', 'hour'],
        ['ip', 'device', 'os', 'app']
    ]
    grouping_functions = ['nunique', 'cumcount', 'sum'] # sum should raise ValueError
    result_df = add_groupby_user_features(X, grouping_categories, grouping_functions[:2])
    for u_list in grouping_categories:
        for grouping_function in grouping_functions[:2]:  # Only test nunique and cumcount
            new_col_name = "_".join(u_list) + "_" + grouping_function
            assert new_col_name in result_df.columns  # Check if the new column is added
            if grouping_function == 'nunique':
                assert result_df[new_col_name].nunique() == len(X[u_list[-1]].unique())  # Check nunique result
            elif grouping_function == 'cumcount':
                assert (result_df[new_col_name] == X.groupby(u_list[:-1]).cumcount()).all()  # Check cumcount result

    # Test that ValueError is raised for unsupported function
    with pytest.raises(ValueError):
        add_groupby_user_features(X, grouping_categories, ['sum'])
        


def test_log_bin_column(sample_data):
    X, _ = sample_data
    # Not actually ones binned in program, but those are created cols.  Check math at least.
    cols_to_bin = ['ip', 'app']
    result_df = log_bin_column(X, cols_to_bin)
    
    assert (result_df['ip'] == np.log2(1 + X['ip']).astype(int)).all()
    assert (result_df['app'] == np.log2(1 + X['app']).astype(int)).all()


def test_add_next_click(sample_data):
    X, _ = sample_data
    result_df = add_next_click(X)
    
    # No sequence of clicks at the moment, just test if it adds the column and is the proper dtype at edges
    assert 'next_click' in result_df.columns
    assert result_df['next_click'].dtype == np.int64