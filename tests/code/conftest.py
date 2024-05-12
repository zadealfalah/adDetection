import pytest
import pandas as pd

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Init sample training dataframes with two sample rows from the 
    initial columns of the dataset. One row with an attribution, the other without.
    Starts with hour and day columns for testing.

    Example usage:
    X, y = sample_data()
    
    Returns:
        tuple: A tuple containing two pandas dataframes: (X, y)
    """
    data = {'ip': [17357, 1504],
            'app': [3, 29692],
            'device': [1, 9],
            'os': [19, 1],
            'channel': [379, 22],
            'click_time': ['2017-11-06 14:33:34', '2017-11-06 16:00:02'],
            'attributed_time': [None, '2017-11-07 10:05:22'],
            'is_attributed': [0, 1],
            'hour':[14, 16],
            'day':[6, 6]}
    
    df = pd.DataFrame(data)
    df['click_time'] = pd.to_datetime(df['click_time'])
    X = df.drop(columns=['is_attributed', 'attributed_time'])
    y = df[['is_attributed']]

    return X, y
