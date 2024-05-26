import pytest
## Import the predictor and .predict here once re-written, then write predictor fixture

def pytest_addoption(parser):
    parser.addoption('--run-id', action='store', default=None, help='Run ID of model to use')

@pytest.fixture(scope='module')
def run_id(request):
    return request.config.getoption('--run-id')



## Reassess below after rewriting predictor in model script for Ray/anyscale
# @pytest.fixture(scope='module')
# def predictor(run_id):
#     best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
#     predictor = 