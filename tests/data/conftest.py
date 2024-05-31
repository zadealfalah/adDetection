import great_expectations as ge
import pandas as pd
import pytest


# Separate X and y as the full data is huge, we just want to test it locally for now.
# As such we'll use the undersampled and split X and y for now
def pytest_addoption(parser):
    """Add option to specify X dataset location when executing tests from CLI.
    Ex: pytest --x_dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--x_dataset-loc", action="store", default=None, help="X Dataset location.")


def pytest_addoption(parser):
    """Add option to specify y dataset location when executing tests from CLI.
    Ex: pytest --y_dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--y_dataset-loc", action="store", default=None, help="Y Dataset location.")


@pytest.fixture(scope="module")
def x_df(request):
    x_dataset_loc = request.config.getoption("--x_dataset-loc")
    X_df = ge.dataset.PandasDataset(pd.read_csv(x_dataset_loc))
    return X_df


@pytest.fixture(scope="module")
def y_df(request):
    y_dataset_loc = request.config.getoption("--y_dataset-loc")
    y_df = ge.dataset.PandasDataset(pd.read_csv(y_dataset_loc))
    return y_df
