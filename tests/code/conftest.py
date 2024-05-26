import pytest
from addetectionscripts.transforms import apply_transforms


# Think about where to host the full dataset.  Can't be github unless zipped
# If github zipped, add a 'zipped' kwarg to unzip
@pytest.fixture
def dataset_loc():
    pass

@pytest.fixture
def preprocessor():
    return apply_transforms()