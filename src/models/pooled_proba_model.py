def generate_probability_pipeline(df_full, max_categories=30, split_test=False, random_state=42):
    """
    Creates pipeline for predicting probability of an item having "poole" label.

    Args:
        df_full (DataFrame): dataframe to train model on
        max_categories (int, optional): maximum number of categorical values to be onehot encoded. Defaults to 30.

    Returns:
        Pipeline: pipeline of feature transformation and model
        df_X_test: DataFrame of test set features
        df_y_test: DataFrame of test set labels
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.preprocessing import OneHotEncoder
    import xgboost

    BEST_MODEL = xgboost.XGBClassifier
    BEST_PARAMETERS = {"max_depth": 7, "n_estimators": 100, "reg_alpha": 5, "reg_lambda": 0}

    # check df_full is a DataFrame
    if not isinstance(df_full, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

    # checks if dataframe has required columns
    req_cols = ['LOAD_DELIVERED_FROM_OFFER', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET',
       'ORIGIN_CITY', 'DESTINATION_CITY', 'ORDER_DAY', 'ORDER_MONTH',
       'ORDER_HOUR', 'PICKUP_DAY', 'PICKUP_MONTH', 'PICKUP_HOUR',
       'BUSINESS_HOURS', 'BUSINESS_HOURS_ORDER_PICKUP', "OFFER_TYPE"]

    if not set(req_cols).issubset(set(df_full.columns)): AssertionError("DataFrame does not contain required columns")

    # filter for delivered offers
    df_full = df_full[df_full["LOAD_DELIVERED_FROM_OFFER"]].reset_index(drop=True)

    # select required columns only 
    df_full = df_full[req_cols]

    # split features and labels
    df_X = df_full.drop("OFFER_TYPE", axis=1)
    df_y = df_full["OFFER_TYPE"] == "pool"    

    # create numerical value transformer
    num_feat = ["BUSINESS_HOURS", "APPROXIMATE_DRIVING_ROUTE_MILEAGE", "PALLETIZED_LINEAR_FEET", "BUSINESS_HOURS_ORDER_PICKUP"]
    num_transformer = Pipeline(steps=[
        ('scaler', MaxAbsScaler()) # z-scale
    ])

    # create categorical value transformer
    cat_feat = ["ORIGIN_CITY", "DESTINATION_CITY", "ORDER_DAY", "ORDER_MONTH", "ORDER_HOUR", "PICKUP_DAY", "PICKUP_MONTH", "PICKUP_HOUR"]
    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(max_categories = max_categories, handle_unknown="ignore")) # one-hot encode 
    ])

    # combine numerical and categorical transformers
    preproc = ColumnTransformer(
        transformers=[
            ("numerical", num_transformer, num_feat),
            ("categorization", cat_transformer, cat_feat)
        ])

    # create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preproc), 
        ("xgb", BEST_MODEL(**BEST_PARAMETERS))
    ])

    if split_test:
        # split train test
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=random_state)

        # train model
        pipeline.fit(df_X_train, df_y_train)

        return pipeline, (df_X_test, df_y_test)

    else:
        # train model
        pipeline.fit(df_X, df_y)

        return pipeline, (None, None)

    