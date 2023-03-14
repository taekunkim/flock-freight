def generate_proba_minimum_pipeline(df, random_state=44, split_test = False):
  import pandas as pd
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import MaxAbsScaler
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OrdinalEncoder
  import xgboost
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split


  # check df is a DataFrame
  if not isinstance(df, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

  # checks if dataframe has required columns
  req_cols = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET',
            'GIVEN_HOURS','REMAINING_HOURS', "PREDICTED_MIN_RATE", 'EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED',
            'STRAIGHT_TRUCK_ALLOWED', 'ORDER_MONTH','ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 
            'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_WEEK', 'OFFER_IS_FTL', 
            'ESTIMATED_MODE_IS_FTL', 'DESTINATION_CLUTER', "PREDICTED_OFFER_COUNT", "N_OFFER", "IS_MINIMUM"]

  if not set(req_cols).issubset(set(df.columns)): AssertionError("DataFrame does not contain required columns")

  X = df[['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET',
            'GIVEN_HOURS','REMAINING_HOURS', "PREDICTED_MIN_RATE", 'EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED',
            'STRAIGHT_TRUCK_ALLOWED', 'ORDER_MONTH','ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 
            'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_WEEK', 'OFFER_IS_FTL', 
            'ESTIMATED_MODE_IS_FTL', 'DESTINATION_CLUTER', "PREDICTED_OFFER_COUNT", "N_OFFER"]]
  y = df[['IS_MINIMUM']]

  num_feat = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET',
            'GIVEN_HOURS','REMAINING_HOURS', "PREDICTED_MIN_RATE"]
  num_transformer = Pipeline(steps=[
      ('scaler', MaxAbsScaler())
  ])

  cat_feat = ['EXCLUSIVE_USE_REQUESTED', 'REEFER_ALLOWED','STRAIGHT_TRUCK_ALLOWED', 
              'ORDER_MONTH','ORDER_WEEK', 'DEADLINE_MONTH', 'DEADLINE_DAY', 
              'DEADLINE_DAY_OF_WEEK', 'DEADLINE_WEEK', 'OFFER_MONTH', 'OFFER_WEEK', 'OFFER_IS_FTL', 
              'ESTIMATED_MODE_IS_FTL', 'DESTINATION_CLUTER', "PREDICTED_OFFER_COUNT", "N_OFFER"]
  cat_transformer = Pipeline(steps=[
      ('onehot', OrdinalEncoder())     # output from Ordinal becomes input to OneHot
  ])

  preproc = ColumnTransformer(
    transformers=[
        ("numerical", num_transformer, num_feat),
        ("categorization", cat_transformer, cat_feat)
    ])

  pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', xgboost.XGBClassifier(max_depth = 10,
                                                                                      n_estimators = 500,
                                                                                      reg_alpha = 1,
                                                                                      reg_lambda = 1))])
  if split_test:
      # split train test
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= random_state)

      # train model
      pl.fit(X_train, y_train)

      y_preds = pl.predict(X_test)
      accuracy = accuracy_score(y_test, y_preds)

      df['PREDICTED_IS_MIN'] = pl.predict(df)

      return pl, df, accuracy

  else:
      # train model
      pl.fit(X, y)
      df['PREDICTED_IS_MIN'] = pl.predict(X)

      return pl, df, None
  
  
  # pl.fit(X_train, y_train)

  # accuracy = accuracy_score(pl.predict(X_test), y_test)

  # df['PREDICTED_IS_MIN'] = pl.predict(df)
  # df

  # return pl, df, accuracy