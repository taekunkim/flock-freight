def generate_estimated_minimum_cost_pipeline(df, random_state=44, split_test = False):
  import pandas as pd
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import MaxAbsScaler
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OrdinalEncoder
  import xgboost
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import train_test_split

  # check df is a DataFrame
  if not isinstance(df, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

  # checks if dataframe has required columns
  req_cols = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET', 'FTL_OFFER_COUNT', 
            "PREDICTED_OFFER_COUNT", 'GIVEN_HOURS','EXCLUSIVE_USE_REQUESTED', 'LOAD_TO_RIDE_REQUESTED',
              'ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 'DEADLINE_WEEK', 'OFFER_WEEK', 'OFFER_IS_FTL', 
              'ESTIMATED_MODE_IS_FTL','DESTINATION_CLUTER', 'MIN_RATE']

  if not set(req_cols).issubset(set(df.columns)): AssertionError("DataFrame does not contain required columns")

  X = df[['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET', 'FTL_OFFER_COUNT', 
            "PREDICTED_OFFER_COUNT", 'GIVEN_HOURS','EXCLUSIVE_USE_REQUESTED', 'LOAD_TO_RIDE_REQUESTED',
              'ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 'DEADLINE_WEEK', 'OFFER_WEEK', 'OFFER_IS_FTL', 
              'ESTIMATED_MODE_IS_FTL','DESTINATION_CLUTER']]
  y = df[['MIN_RATE']]

  num_feat = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET', 'FTL_OFFER_COUNT', 
            "PREDICTED_OFFER_COUNT", 'GIVEN_HOURS']
  num_transformer = Pipeline(steps=[
      ('scaler', MaxAbsScaler())
  ])

  cat_feat = ['EXCLUSIVE_USE_REQUESTED', 'LOAD_TO_RIDE_REQUESTED',
              'ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 'DEADLINE_WEEK', 'OFFER_WEEK', 'OFFER_IS_FTL', 
              'ESTIMATED_MODE_IS_FTL','DESTINATION_CLUTER']
  cat_transformer = Pipeline(steps=[
      ('onehot', OrdinalEncoder())     # output from Ordinal becomes input to OneHot
  ])

  preproc = ColumnTransformer(
    transformers=[
        ("numerical", num_transformer, num_feat),
        ("categorization", cat_transformer, cat_feat)
    ])


  pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', xgboost.XGBRegressor(max_depth = 10,
                                                                                     n_estimators = 500,
                                                                                     reg_alpha = 1,
                                                                                     reg_lambda = 1))])
  
  if split_test:
      # split train test
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= random_state)

      # train model
      pl.fit(X_train, y_train)

      y_preds = pl.predict(X_test)
      mse = mean_squared_error(y_test, y_preds)

      df["PREDICTED_MIN_RATE"] = pl.predict(df)

      return pl, df, mse

  else:
      # train model
      pl.fit(X, y)
      df["PREDICTED_MIN_RATE"] = pl.predict(X)

      return pl, df, None

  
  # pl.fit(X_train, y_train)

  # y_preds = pl.predict(X_test)
  # mse = mean_squared_error(y_test, y_preds)

  # df["PREDICTED_MIN_RATE"] = pl.predict(df)

  # return pl, df, mse