import pandas as pd
import numpy as np
from src.features import data_cleaning
import json

def get_raw_data(config_dir):
    """
    Returns raw data for EDA and Train-Test split purposes.

    Returns:
        DataFrame: tuple of orders and offers DataFrame
    """
    # read config file
    with open(config_dir, "r") as config_file:
        config = json.load(config_file)

    orders = pd.read_csv(config["data"]["raw"]["dir"]["orders"], low_memory=False)
    offers = pd.read_csv(config["data"]["raw"]["dir"]["offers"], low_memory=False)

    return orders, offers

def clean_all(orders, offers):
    offers = data_cleaning.change_to_date(offers, ["CREATED_ON_HQ"])

    orders = data_cleaning.change_to_date(orders, ["ORDER_DATETIME_PST", "PICKUP_DEADLINE_PST"])

    offers = data_cleaning.flatten_ref_num(offers)
    
    orders = data_cleaning.flatten_ref_num(orders)

    merged = data_cleaning.join_offers_orders(offers, orders)

    merged = merged.dropna()

    merged = merged.sort_values("CREATED_ON_HQ", ascending=True)

    merged = data_cleaning.is_weekend(merged, col="PICKUP_DEADLINE_PST", new_col="DEADLINE_ON_WEEKDAY")

    # time remaining from when an order was made to the deadline
    merged = data_cleaning.get_remaining_time(merged, past="ORDER_DATETIME_PST", future="PICKUP_DEADLINE_PST", new_col="GIVEN_HOURS")

    # time remaining from when an offer was made to the deadline
    merged = data_cleaning.get_remaining_time(merged, past="CREATED_ON_HQ", future="PICKUP_DEADLINE_PST", new_col="REMAINING_HOURS")

    merged = data_cleaning.get_prorated_rate(merged, overwrite=True)

    merged = data_cleaning.get_coordinates(merged, col="ORIGIN_3DIGIT_ZIP", new_col="ORIGIN")
    merged = data_cleaning.get_coordinates(merged, col="DESTINATION_3DIGIT_ZIP", new_col="DESTINATION")

    merged = data_cleaning.extract_from_date(merged, col="ORDER_DATETIME_PST", new_col="ORDER")
    merged = data_cleaning.extract_from_date(merged, col="PICKUP_DEADLINE_PST", new_col="DEADLINE")
    merged = data_cleaning.extract_from_date(merged, col="CREATED_ON_HQ", new_col="OFFER")

    merged = data_cleaning.assign_ids(merged, col="REFERENCE_NUMBER")
    merged = data_cleaning.assign_ids(merged, "CARRIER_ID")

    merged = data_cleaning.change_to_boolean(merged, col="OFFER_TYPE", new_col="OFFER_IS_FTL", true_val="quote")
    merged = data_cleaning.change_to_boolean(merged, col="TRANSPORT_MODE", new_col="ESTIMATED_MODE_IS_FTL", true_val="FTL")

    merged = merged.replace({True: 1, False: 0})

    to_drop = ["SELF_SERVE", "AUTOMATICALLY_APPROVED", "MANUALLY_APPROVED", "WAS_EVER_UNCOVERED", "COVERING_OFFER", "LOAD_DELIVERED_FROM_OFFER", "RECOMMENDED_LOAD", "VALID", "DELIVERY_TIME_CONSTRAINT"]
    merged = merged.drop(to_drop, axis=1)

    return orders, offers, merged

def remove_outliers(merged):

    def identify_anomalies(df, col, zscore=2.5, floor=0):
        """_summary_

        Args:
            df (dataframe): the dataframe to analyze
            col (str): the column of values to analyze
            zscore (float, optional): the threshold zscore used to identify anomalies. Defaults to 2.5.
            floor (int, optional): the minimum viable value. Defaults to 0.
        """
        vals = df[col]
        lower_threshold = np.max([floor, np.mean(vals) - zscore * np.std(vals)])
        upper_threshold = np.mean(vals) + zscore * np.std(vals)
        outlier_id = set(df.index[(vals < lower_threshold) | (vals > upper_threshold)])

        return outlier_id

    # create an aggregated "orders" dataframe
    orders = merged.copy()
    orders = orders.sort_values("RATE_USD", ascending=True)

    orders["OFFER_COUNT"] = orders.groupby("REFERENCE_NUMBER")["ORDER_DATETIME_PST"].transform("count")
    orders["FTL_OFFER_COUNT"] = orders.groupby("REFERENCE_NUMBER")["OFFER_IS_FTL"].transform("sum")
    orders = orders.groupby("REFERENCE_NUMBER").first().sort_values("ORDER_DATETIME_PST")

    # create sets to keep track of which offers and orderes to remove
    orders_to_drop = set() # set of order REFERENCE_NUMBERs to remove
    offers_to_drop = set() # set of indicies in the "orders" table to remove

    # identify what offers suggested abnormal rates
    offers_to_drop = offers_to_drop.union(identify_anomalies(df=merged, col="RATE_USD", zscore=2.5, floor=0))

    # identify what orders received abnormal best-rates
    orders_to_drop = orders_to_drop.union(identify_anomalies(df=orders, col="RATE_USD", zscore=2.5, floor=0))

    # identify what orders received abnormally many or few offers 
    orders_to_drop = orders_to_drop.union(identify_anomalies(df=orders, col="OFFER_COUNT", zscore=2.5, floor=0))

    # identify what orders were placed with abnormally small or greate amount of time until the pickup deadline
    orders_to_drop = orders_to_drop.union(identify_anomalies(df=orders, col="GIVEN_HOURS", zscore=2.5, floor=0))

    # identify what orders have abnormally long or short loads
    orders_to_drop = orders_to_drop.union(identify_anomalies(df=orders, col="PALLETIZED_LINEAR_FEET", zscore=2.5, floor=0))

    # remove offers from the merged dataset
    offer_id_to_keep = [merged["REFERENCE_NUMBER"].apply(lambda ref: ref not in orders_to_drop)]
    offer_id_to_keep = ([pd.Series(merged.index).apply(lambda id: id not in offers_to_drop)] and offer_id_to_keep)[0]
    merged = merged[offer_id_to_keep].reset_index(drop=True)

    return merged

def aggregate_data(merged):
    """_summary_

      Args:
          merged (dataframe): the dataframe to analyze

      Return:
          updated dataframe with OFFER_COUNT and FTL_OFFER_COUNT
    """
    orders = merged.copy().sort_values("RATE_USD", ascending=True)
    orders["OFFER_COUNT"] = orders.groupby("REFERENCE_NUMBER")["ORDER_DATETIME_PST"].transform("count")
    orders["FTL_OFFER_COUNT"] = orders.groupby("REFERENCE_NUMBER")["OFFER_IS_FTL"].transform("sum")
    orders = orders.groupby("REFERENCE_NUMBER").first().sort_values("ORDER_DATETIME_PST")
    return orders

def merge_order_offer_on_reference(offer_df, order_df, how = "left", on = "REFERENCE_NUMBER"):
  """_summary_

      Args:
          offer_df (dataframe): offer dataframe
          order_df (dataframe): order dataframe

      Return:
          merged dataframe with offer and order on REFERENCE_NUMBER
    """
  # check df is a DataFrame
  if not isinstance(offer_df, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")

  # check df is a DataFrame
  if not isinstance(order_df, pd.DataFrame): AssertionError("Parameter must be Pandas DataFrame")
  
  order_df = order_df.reset_index()

  
  # checks if dataframe has required columns
  req_cols = ['REFERENCE_NUMBER', 'OFFER_COUNT', "FTL_OFFER_COUNT"]

  if not set(req_cols).issubset(set(order_df.columns)): AssertionError("DataFrame order_df does not contain required columns")
  
  to_return = pd.merge(offer_df, order_df[req_cols], how = how, on = on)
  return to_return

def split_train_test(X, random_state = 44):
    """_summary_

      Args:
          X (dataframe): dataframe for spliting

      Return:
          trainset for X_train testset for X_test with test_size = 0.3
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(X, test_size=0.3, random_state= random_state)

    return X_train, X_test


def Kmeans_clustering(merge_df, order_df, is_test_run = False):
    """_summary_

      Args:
          merge_df (dataframe): merged offer dataframe
          order_df (dataframe): order dataframe
          is_test_run (boolean): check whether code runs for testset or not

      Return:
          order_df with clusters for origin and destination
    """
    from sklearn.cluster import KMeans
    import pandas as pd

    coords = pd.concat([merge_df[["ORIGIN_X", "ORIGIN_Y"]].rename({"ORIGIN_X": "X", "ORIGIN_Y": "Y"}, axis=1), merge_df[["DESTINATION_X", "DESTINATION_Y"]].rename({"DESTINATION_X": "X", "DESTINATION_Y": "Y"}, axis=1)], ignore_index=True)

    if not is_test_run:
      Kmean = KMeans(n_clusters=10)
      Kmean.fit(coords)
      coords["cluster"] = Kmean.predict(coords)
    else:
      Kmean = KMeans(n_clusters=1)
      Kmean.fit(coords)
      coords["cluster"] = Kmean.predict(coords)

    unique_cluster = coords.groupby(['X', 'Y']).first().reset_index()
    order_df["ORIGIN_CLUSTER"] = order_df[["ORIGIN_X","ORIGIN_Y"]].merge(unique_cluster, left_on = ["ORIGIN_X", "ORIGIN_Y"], right_on = ['X', 'Y'])['cluster'].values
    order_df["DESTINATION_CLUTER"] = order_df[["DESTINATION_X","DESTINATION_Y"]].merge(unique_cluster, left_on = ["DESTINATION_X", "DESTINATION_Y"], right_on = ['X', 'Y'])['cluster'].values
    orgin = list(map(str, order_df['ORIGIN_CLUSTER'].array))
    dest = list(map(str, order_df['DESTINATION_CLUTER'].array))


    order_df["ORGIN_DEST_COMB"] = [int(x+y) for x,y in zip(orgin,dest)]
    return order_df

def min_rate_df(df):
    """_summary_

      Args:
          df (dataframe): merged dataframe

      Return:
          df with minimum_rate
    """
    gr = df.sort_values(["REFERENCE_NUMBER", "CREATED_ON_HQ"]).groupby("REFERENCE_NUMBER")
    df["N_OFFER"] = gr.cumcount()
    min_rate = df.groupby("REFERENCE_NUMBER").agg({"RATE_USD":"min"})
    min_rate = min_rate.reset_index().rename(columns = {"RATE_USD":"MIN_RATE"})
    df = df.merge(min_rate, on = "REFERENCE_NUMBER")
    return df

def generate_is_minimum_col(df):
    """_summary_

      Args:
          df (dataframe): merged dataframe

      Return:
          df with IS_MINIMUM col that check whether current offer is minimum cost
    """
    df['IS_MINIMUM'] = df['RATE_USD'] == df['MIN_RATE']
    return df