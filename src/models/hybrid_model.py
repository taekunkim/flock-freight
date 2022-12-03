import pooled_proba_model
import pooled_regression_model
import quote_regression_model

def get_weighted_costs():
    # get training data
    pooled = ... # data_cleaning.WHATEVER()
    merged = ... # data_cleaning.WHATEVER()

    # get test data
    test_data = ...

    pooled_proba = pooled_proba_model(merged)
    pooled_reg = pooled_regression_model(pooled)
    quote_reg = quote_regression_model(merged)

    probas = pooled_proba.predict_proba(test_data)
    pooled_cost = pooled_reg(test_data)
    quote_cost = quote_reg(test_data)

    weighted_cost = probas * pooled_cost  + (1-probas) *  quote_cost

    return weighted_cost
