import pooled_proba_model
import pooled_regression_model
import quote_regression_model

def get_weighted_costs():
    # get data
    # orders, offers = data_for_modelling.get_raw_data()

    # clean data
    # orders, offers, merged, pooled = data_for_modelling.clean_all(orders, offers)

    # read pickle data
    import pandas as pd
    merged = pd.read_pickle("data/pickels/merged.pkl") 
    pooled = pd.read_pickle("data/pickels/pooled.pkl")

    pooled_proba_pl, (df_X_test, df_y_test) = pooled_proba_model(merged)
    # pooled_reg_pl, (df_X_test, df_y_test) = pooled_regression_model(pooled)
    # quote_reg_pl, (df_X_test, df_y_test) = quote_regression_model(merged)

    df_X_test.head(1)



    # probas = pooled_proba.predict_proba(test_data)
    # pooled_cost = pooled_reg(test_data)
    # quote_cost = quote_reg(test_data)

    # weighted_cost = probas * pooled_cost  + (1-probas) *  quote_cost

    # return weighted_cost

get_weighted_costs()