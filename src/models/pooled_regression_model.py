def generate_pooled_regression_pipeline(df):
    import sklearn.preprocessing as pp
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Lasso

    X = pooled[["APPROXIMATE_DRIVING_ROUTE_MILEAGE", "PALLETIZED_LINEAR_FEET", 
            "BUSINESS_HOURS", "REMAINIG_TIME",
            "ORDER_DAY", "ORDER_MONTH", "ORDER_HOUR",
            "PICKUP_DAY", "PICKUP_MONTH", "PICKUP_HOUR",
            "BUSINESS_HOURS_ORDER_PICKUP",
            "ORIGIN_CITY", "DESTINATION_CITY"]]

    y = pooled["PRORATED_RATE_USD"].to_list()

    if not [col in df.cols for col in X].mean() == 1: AssertionError("DataFrame does not contain required columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

    pl_lasso= Pipeline(steps=[("preprocessor", preproc), ("regressor", Lasso(0.21))])

    pipeline = pl_lasso

    return pipeline