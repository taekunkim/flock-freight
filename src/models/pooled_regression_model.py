def generate_pooled_regression_pipeline(df_full):
    import pandas as pd
    import sklearn.preprocessing as pp
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Lasso

    BEST_MODEL = Lasso()
    BEST_PARAMETER = {"alpha": 0.21}

    # check df_full is a DataFrame
    if not isinstance(df_full, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

    req_cols = ["APPROXIMATE_DRIVING_ROUTE_MILEAGE", "PALLETIZED_LINEAR_FEET", 
            "BUSINESS_HOURS", "REMAINIG_TIME",
            "ORDER_DAY", "ORDER_MONTH", "ORDER_HOUR",
            "PICKUP_DAY", "PICKUP_MONTH", "PICKUP_HOUR",
            "BUSINESS_HOURS_ORDER_PICKUP",
            "ORIGIN_CITY", "DESTINATION_CITY", "PRORATED_RATE_USD", "LOAD_DELIVERED_FROM_OFFER"]

    if not set(req_cols).issubset(set(df_full.columns)): AssertionError("DataFrame does not contain required columns")

    # filter for delivered offers
    df_full = df_full[df_full["LOAD_DELIVERED_FROM_OFFER"]].reset_index(drop=True)

    # select required columns only 
    df_full = df_full[req_cols]

    # split features and labels
    df_X = df_full.drop(["PRORATED_RATE_USD"], axis=1)
    df_y = df_full["PRORATED_RATE_USD"]

    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.3)

    # Numerical columns and associated transformers
    num_feat = ["APPROXIMATE_DRIVING_ROUTE_MILEAGE", "PALLETIZED_LINEAR_FEET",
                "BUSINESS_HOURS_ORDER_PICKUP", "REMAINIG_TIME"]
    num_transformer = Pipeline(steps=[
        ('scaler', pp.MaxAbsScaler())
    ])

    # Categorical columns and associated transformers
    cat_feat = ["ORIGIN_CITY", "DESTINATION_CITY", "BUSINESS_HOURS",
                "ORDER_DAY", "ORDER_MONTH", "ORDER_HOUR",
                "PICKUP_DAY", "PICKUP_MONTH", "PICKUP_HOUR"]
    cat_transformer = Pipeline(steps=[('onehot', pp.OneHotEncoder(max_categories = 30, handle_unknown = 'ignore'))
    ])

    # Preprocessing pipeline (put them together)
    preproc = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_feat),
            ('cat', cat_transformer, cat_feat)
        ])

    pl= Pipeline(steps=[("preprocessor", preproc), ("regressor", BEST_MODEL(BEST_PARAMETER))])

    pl.fit(df_X_train, df_y_train)

    return pl, (df_X_test, df_y_test)