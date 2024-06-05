## Not much to test as the data was already cleaned up.

## Since we are currently splitting the data in to X and y with undersampling,
## this provides us with some more tests to run


## Can add tests to make sure equal number of 1's/0's in y
## Can add tests to make sure equal number of rows between X and y
def dataset(X, y):
    X_column_list = ["ip", "app", "device", "os", "channel", "click_time"]
    y_column_list = ["is_attributed"]
    X.expect_table_columns_to_match_ordered_list(column_list=X_column_list)
    y.expect_table_columns_to_match_ordered_list(column_list=y_column_list)

    y_vals = [0, 1]
    y.expect_column_values_to_be_in_set(column="is_attributed", value_set=y_vals)
    y.expect_column_values_to_not_be_null(column="is_attributed")

    X.expect_column_values_to_be_of_type(column="click_time", type_="pd.datetime")  # type adherence

    X_expectation_suite = X.get_expectation_suite(discard_failed_expectations=False)
    y_expectation_suite = y.get_expectation_suite(discard_failed_expectations=False)

    X_results = X.validate(expectation_suite=X_expectation_suite, only_return_failures=True).to_json_dict()
    y_results = y.validate(expectation_suite=y_expectation_suite, only_return_failures=True).to_json_dict()

    assert X_results["success"] and y_results["success"]
