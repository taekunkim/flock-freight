import pandas as pd
import numpy as np
import data_cleaning
import json

# read config file
with open("config/config.json", "r") as config_file:
    config = json.load(config_file)

def get_raw_data():
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

    pooled = data_cleaning.get_prorated_rate(merged)

    return orders, offers, merged, pooled