def generate_offer_number_pipeline(df, random_state=44, split_test = False, is_test_run = False):
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.preprocessing import MaxAbsScaler
  from sklearn.preprocessing import OrdinalEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  import pandas as pd
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import train_test_split

  # check df is a DataFrame
  if not isinstance(df, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

  # checks if dataframe has required columns
  req_cols = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', "PALLETIZED_LINEAR_FEET", "GIVEN_HOURS", 
              "REMAINING_HOURS", 'EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED','STRAIGHT_TRUCK_ALLOWED', 
              'LOAD_TO_RIDE_REQUESTED', 'ORDER_MONTH', 'ORDER_DAY', 'ORDER_DAY_OF_WEEK','ORDER_WEEK', 'DEADLINE_MONTH', 
              'DEADLINE_DAY', 'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_DAY', 'OFFER_DAY_OF_WEEK', 
              'OFFER_WEEK', 'OFFER_IS_FTL', 'ESTIMATED_MODE_IS_FTL', 'ORIGIN_CLUSTER', 'DESTINATION_CLUTER','ORGIN_DEST_COMB', 
              'OFFER_COUNT']

  if not set(req_cols).issubset(set(df.columns)): AssertionError("DataFrame does not contain required columns")

  X = df[['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', "PALLETIZED_LINEAR_FEET", "GIVEN_HOURS", 
              "REMAINING_HOURS", 'EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED','STRAIGHT_TRUCK_ALLOWED', 
              'LOAD_TO_RIDE_REQUESTED', 'ORDER_MONTH', 'ORDER_DAY', 'ORDER_DAY_OF_WEEK','ORDER_WEEK', 'DEADLINE_MONTH', 
              'DEADLINE_DAY', 'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_DAY', 'OFFER_DAY_OF_WEEK', 
              'OFFER_WEEK', 'OFFER_IS_FTL', 'ESTIMATED_MODE_IS_FTL', 'ORIGIN_CLUSTER', 'DESTINATION_CLUTER','ORGIN_DEST_COMB']]
  y = df[['OFFER_COUNT']]

  num_feat = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', "PALLETIZED_LINEAR_FEET", "GIVEN_HOURS", "REMAINING_HOURS"]
  num_transformer = Pipeline(steps=[
      ('scaler', MaxAbsScaler())
  ])

  cat_feat = ['EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED','STRAIGHT_TRUCK_ALLOWED', 'LOAD_TO_RIDE_REQUESTED', 'ORDER_MONTH', 'ORDER_DAY', 
              'ORDER_DAY_OF_WEEK','ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_DAY', 'OFFER_DAY_OF_WEEK', 
              'OFFER_WEEK', 'OFFER_IS_FTL', 'ESTIMATED_MODE_IS_FTL', 'ORIGIN_CLUSTER', 'DESTINATION_CLUTER','ORGIN_DEST_COMB']
  cat_transformer = Pipeline(steps=[
      ('onehot', OrdinalEncoder())     # output from Ordinal becomes input to OneHot
  ])


  preproc = ColumnTransformer(
    transformers=[
        ("numerical", num_transformer, num_feat),
        ("categorization", cat_transformer, cat_feat)
    ])


  pl = Pipeline(steps=[('preprocessor', preproc), ('clf', DecisionTreeClassifier(max_depth=40, min_samples_leaf=2, min_samples_split=2))])

  if split_test:
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = random_state)

    # train model
    pl.fit(X_train, y_train)

    y_preds = pl.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)

    df["PREDICTED_OFFER_COUNT"] = pl.predict(df)

    return pl, df, mse

  else:
      # train model
      pl.fit(X, y)
      df["PREDICTED_OFFER_COUNT"] = pl.predict(df)
      return pl, df, None

