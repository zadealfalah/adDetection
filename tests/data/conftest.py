import pytest
import pandas as pd
import great_expectations as ge

## Even zipped, the file is 1.9GB so we can't host it on github.  Keep it local for testing, and when in prod we will use an S3 bucket.
## pd.read_csv should suffice for reading zipped, can make more explicit later with compression='zip', etc as kwargs

def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--dataset-loc", action="store", default=None, help="Dataset location.")


@pytest.fixture(scope="module")
def df(request):
    dataset_loc = request.config.getoption("--dataset-loc")
    parse_dates_cols = ['click_time']
    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc, parse_dates=parse_dates_cols))
    return df
