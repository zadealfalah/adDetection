# Posted dataset was already cleaned, so there isn't much to check here for the initial dataset.
def test_dataset(df):
    column_list = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    df.expect_table_columns_to_match_ordered_list(column_list=column_list)
    tag_values = [0, 1]
    df.expect_column_balues_to_be_in_set(column='is_attributed', value_set=tag_values)
    df.expect_column_values_to_not_be_null(column='is_attributed')
    
    
    expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
    results = df.validate(expectation_suite=expectation_suite, only_return_failures=True).to_json_dict()
    assert results['success']