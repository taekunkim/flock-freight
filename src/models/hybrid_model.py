def get_weighted_costs(test_set, pooled_proba_pl, pooled_reg_pl, quote_reg_pl):

    probas = pooled_proba_pl.predict_proba(test_set).T[0]
    pooled_cost = pooled_reg_pl.predict(test_set)
    quote_cost = quote_reg_pl.predict(test_set)

    weighted_cost = probas * pooled_cost  + (1-probas) *  quote_cost

    return weighted_cost
