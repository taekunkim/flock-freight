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

    orders = pd.read_csv(config["data"]["raw"]["dir"]["orders"], low_memory=False)[config["data"]["raw"]["req_cols"]["orders"]]
    offers = pd.read_csv(config["data"]["raw"]["dir"]["offers"], low_memory=False)[list(config["data"]["raw"]["req_cols"]["offers"])]

    return orders, offers

def clean_all(orders, offers):
    orders = data_cleaning.change_to_date(orders, ["ORDER_DATETIME_PST", "PICKUP_DEADLINE_PST"])
    orders = data_cleaning.parse_zipcode(orders)
    orders = data_cleaning.parse_datetime(orders)
    orders = data_cleaning.flatten_ref_num(orders)
    
    offers = data_cleaning.change_to_date(offers, ["CREATED_ON_HQ"])
    offers = data_cleaning.flatten_ref_num(offers)
    
    merged = data_cleaning.join_offers_orders(offers, orders, how="inner")
    merged = data_cleaning.get_remaining_time(merged)
    merged = data_cleaning.during_business_hours(merged)
    merged = data_cleaning.impute_mileage(merged)
    merged = data_cleaning.get_business_hours(merged)
    merged = data_cleaning.get_prorated_rate(merged)

    return orders, offers, merged

def split_train_test(merged, test_ratio=0.3):
    import numpy as np
    # here, we only consider delivered offers for both training and testing
    delivered = merged[merged["LOAD_DELIVERED_FROM_OFFER"]]

    test_size = int(delivered.shape[0]*test_ratio)
    test_ref = np.random.choice(delivered["REFERENCE_NUMBER"], size=test_size, replace=False)

    test_set = merged[merged["REFERENCE_NUMBER"].isin(test_ref)]
    train_set = merged[~merged["REFERENCE_NUMBER"].isin(test_ref)]

    test_set = test_set.sort_values(by=["REFERENCE_NUMBER", "CREATED_ON_HQ"], ascending=True)
    train_set = train_set.sort_values(by=["REFERENCE_NUMBER", "CREATED_ON_HQ"], ascending=True)
    
    return train_set, test_set
