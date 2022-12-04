def generate_quote_regression_pipeline(df_full, max_categories=30):
  """
  Creates pipeline for regression model to predict RATE of an item having "quote" label.

  Args:
      df_full (DataFrame): dataframe to train model on
      max_categories (int, optional): maximum number of categorical values to be onehot encoded. Defaults to 30.

  Returns:
      Pipeline: pipeline of feature transformation and model
      df_X_test: DataFrame of test set features
      df_y_test: DataFrame of test set labels
  """
  import pandas as pd
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.preprocessing import MaxAbsScaler
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import train_test_split

  # BEST_MODEL = Ridge(alpha = 1)
  BEST_PARAMETERS = {"alpha": 34}

  # check df_full is a DataFrame
  if not isinstance(df_full, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

  # checks if dataframe has required columns
  req_cols = ['RATE_USD', 'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'PALLETIZED_LINEAR_FEET',
       'ORIGIN_CITY', 'DESTINATION_CITY', 'ORDER_DAY', 'ORDER_MONTH',
       'ORDER_HOUR', 'PICKUP_DAY', 'PICKUP_MONTH', 'PICKUP_HOUR',
       'REMAINIG_TIME', 'BUSINESS_HOURS', 'BUSINESS_HOURS_ORDER_PICKUP', "OFFER_TYPE", "LOAD_DELIVERED_FROM_OFFER"]

  if not set(req_cols).issubset(set(df_full.columns)): AssertionError("DataFrame does not contain required columns")

  # filter for delivered offers
  df_full = df_full[df_full["LOAD_DELIVERED_FROM_OFFER"]].reset_index(drop=True)

  # select required columns only 
  df_full = df_full[req_cols]

  # only quote
  df_full = df_full[df_full["OFFER_TYPE"] == "quote"].reset_index(drop=True)
  df_full = df_full.drop(["OFFER_TYPE"], axis=1)

  # split features and labels
  df_X = df_full.drop(["RATE_USD"], axis=1)
  df_y = df_full["RATE_USD"]

  # split train test
  df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

  # create numerical value transformer
  num_feat = ["APPROXIMATE_DRIVING_ROUTE_MILEAGE", "PALLETIZED_LINEAR_FEET", "REMAINIG_TIME", 'BUSINESS_HOURS_ORDER_PICKUP']
  num_transformer = Pipeline(steps=[
      ('scaler', MaxAbsScaler())
  ])

  cat_feat = ['ORIGIN_CITY', 'DESTINATION_CITY', 'BUSINESS_HOURS', 'ORDER_DAY', 'ORDER_MONTH', 'ORDER_HOUR', 'PICKUP_DAY', 'PICKUP_MONTH', 'PICKUP_HOUR']
  cat_transformer = Pipeline(steps=[
      ('onehot', OneHotEncoder(max_categories = max_categories, handle_unknown = 'ignore'))     # output from Ordinal becomes input to OneHot
  ])

  # combine numerical and categorical transformers
  preproc = ColumnTransformer(
    transformers=[
        ("numerical", num_transformer, num_feat),
        ("categorization", cat_transformer, cat_feat)
    ])

  # create pipeline
  pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', Ridge(alpha = BEST_PARAMETERS['alpha']))])

  # train model
  pl.fit(df_X_train, df_y_train)
  
  return pl, (df_X_test, df_y_test)