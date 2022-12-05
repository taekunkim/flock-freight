def get_weighted_costs(config_dir):
    from src.models import pooled_proba_model
    from src.models import pooled_regression_model
    from src.models import quote_regression_model
    from src.features import data_for_modelling

    # get data
    orders_raw, offers_raw = data_for_modelling.get_raw_data(config_dir)

    # clean data
    _, _, merged = data_for_modelling.clean_all(orders_raw, offers_raw)

    # split offers into train test
    merged_train, merged_test = data_for_modelling.split_train_test(merged)

    merged_test.sort_values(by=["REFERENCE_NUMBER", "CREATED_ON_HQ"], ascending=True)

    pooled_proba_pl, (_, _) = pooled_proba_model.generate_probability_pipeline(merged_train)
    pooled_reg_pl, (_, _) = pooled_regression_model.generate_pooled_regression_pipeline(merged_train)
    quote_reg_pl, (_, _) = quote_regression_model.generate_quote_regression_pipeline(merged_train)

    probas = pooled_proba_pl.predict_proba(merged_test).T[0]
    pooled_cost = pooled_reg_pl.predict(merged_test)
    quote_cost = quote_reg_pl.predict(merged_test)

    weighted_cost = probas * pooled_cost  + (1-probas) *  quote_cost

    return weighted_cost
