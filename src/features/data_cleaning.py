import pandas as pd
import numpy as np

def change_to_date(df, cols):
    """
    Converts values in columns to datetime objects.

    Args:
        df (DataFrame): dataframe
        cols (list): list of columns
    """

    # deep copy
    df = df.copy()

    df[cols] = df[cols].apply(pd.to_datetime, errors="coerce")
    return df

def parse_zipcode(df, cols=["ORIGIN_3DIGIT_ZIP", "DESTINATION_3DIGIT_ZIP"], new_cols=["ORIGIN_CITY", "DESTINATION_CITY"]):

    from postalcodes_ca import fsa_codes
    import json

    # deep copy
    df = df.copy()

    USZIPCODE_FILE = "../data/threeDigitZipCodes.json"

    def parse_zipcode_(zipcode):
        """
        Converts three digit zipcode to city or state.

        Args:
            zipcode (str): first three digits or letters of American zipcode or Canadian postal code
            cols (str): names of the columns to parse zipcodes
            new_cols (str): names of new columns that will contain the city names


        Returns:
            DataFrame: updated DataFrame
        """
        # covnerts zipcode to string
        if not isinstance(zipcode, str): zipcode = str(zipcode)

        if zipcode in zipcode_dict: # checks if zipcode is in America
            nearby = zipcode_dict[zipcode] # the nearest metropolitan
            return nearby["city"] + " " + nearby["state"]
        else:
            try: # checks if zipcode appears in Canadian zipcodes
                nearby = fsa_codes.get_nearby(zipcode, radius=1)[0] # the nearest metropolitan
                return nearby.name + " " + nearby.province
                
            except:
                return np.nan

    # reads in US zipcode dictionary
    with open(USZIPCODE_FILE) as json_file:
        zipcode_dict = json.load(json_file)

    # parses one column at a time
    for col, new_col in zip(cols, new_cols):
        df[new_col] = df[col].apply(lambda zipcode: parse_zipcode_(zipcode))

    return df

def flatten_ref_num(df, col="REFERENCE_NUMBER"):
    """
    Flattens dataframe by expanding a column with list values into multiple rows.

    Args:
        df (DataFrame): dataframe to flatten
        cols (_type_): columns to flatten
    """
    # deep copy
    df = df.copy()

    if col=="REFERENCE_NUMBER": 
        # REFERENCE_NUMBER column has a unique pattern of list elements where \n exists between every item
        df[col] = df[col].apply(lambda lst: lst.split("\"")[1::2])

    # create an individual row for each value in REFERENCE_NUMBER
    df = df.explode(col).reset_index(drop=True)

    return df

def join_offers_orders(offers, orders, how="left", on="REFERENCE_NUMBER"):
    """
    Returns a table with each offer paired with its order

    Args:
        offers (DataFrame): offers table
        orders (DataFrame): orders table
    """

    # deep copy
    offers = offers.copy()
    orders = orders.copy()

    return pd.merge(offers, orders, how=how, on=on)

def get_remaining_time(df, past="CREATED_ON_HQ", future="PICKUP_DEADLINE_PST", new_col="REMAINING_TIME"):
    """
    Calculates remaining delivery time from when an offer is made.

    Args:
        df (DataFrame): a merged dataset of offeres and orders
        past (str): name of column that stores later time information. Defaults to "CREATED_ON_HQ".
        future (str): name of column that stores earlier time information. Defaults to "PICKUP_DEADLINE_PST".
        new_col (str): name of column to store the remaining time. Defaults to "REMAINIG_TIME".

    """
    # deep copy
    df = df.copy()

    df[new_col] = (df[future] - df[past]).dt.total_seconds() // 3600

    # ensure that there are not negative times
    df = df[df[new_col]>=0].reset_index(drop=True)

    return df

def during_business_hours(df, col, new_col):
    """
    Adds a column that shows whether an offer was made during business hours.

    Args:
        df (DataFrame): DataFrame to study
    """

    # deep copy
    df = df.copy()

    hours = df[col].dt.hour
    hours = ((hours >= 8) | (hours <= 18))

    days = df[col].dt.dayofweek
    days = days < 5

    during_business_hours = hours & days

    df[new_col] = during_business_hours

    return df

def get_prorated_rate(df, delivered_only=False, overwrite=True):
    """
    Calculates prorated cost of pooled offers.

    Args:
        df (DataFrame): DataFrame to perform calculations on
        delivered_only (boolean): whether to filter for only offers that were chosen
        over_write(boolean): whether to overwrite RATE_USD with the prorated rate

    """

    # deep copy
    df = df.copy()

    # filters for delivered offers
    if delivered_only:
        df = df[df["LOAD_DELIVERED_FROM_OFFER"]].reset_index(drop=True).drop("LOAD_DELIVERED_FROM_OFFER", axis=1)

    # assigns unique value to each offer for future data aggregation purposes
    df["OFFER_ID"] = df["CARRIER_ID"] + " " + df["CREATED_ON_HQ"].apply(lambda x: str(x))

    # records the sum of [linear feet] of all orders inlcuded in an offer
    total_feet = df.groupby("OFFER_ID")["PALLETIZED_LINEAR_FEET"].transform("sum")

    # decides whether to overwrite
    new_col = "RATE_USD" if overwrite else "PRORATED_RATE_USD"

    # calculates the prorated rate
    df[new_col] = (df["RATE_USD"] * df["PALLETIZED_LINEAR_FEET"] / total_feet) * (df["OFFER_TYPE"] == "pool") + df["RATE_USD"] * (df["OFFER_TYPE"] == "quote")

    # drops OFFER_ID
    df = df.drop("OFFER_ID", axis=1)

    return df

def get_business_hours(df, order="ORDER_DATETIME_PST", pickup="PICKUP_DEADLINE_PST"):
    """
    Count number of hours from ORDER_DATETIME_PST to PICKUP_DEADLINE_PST that 
    are in the business hours. Create a new column called BUSINESS_HOURS_ORDER_PICKUP.

    Args:
        df (DataFrame): dataframe to add BUSINESS_HOURS_ORDER_PICKUP
        order (Series): order date and time
        pickup (Series): pickup date and time
        
    It takes about 2 minutes to run.
    """

    # deep copy
    df = df.copy()


    import datetime
    # packages to generate the rule of business hours and calculate the hours
    import businesstimedelta
    import holidays as pyholidays
    
    # Set the working hours and days
    work_hours = businesstimedelta.WorkDayRule(start_time=datetime.time(8),
                                               end_time=datetime.time(18),
                                               working_days=[0, 1, 2, 3, 4])

    # Get possible holidays
    all_holidays = pyholidays.UnitedStates()
    holidays = businesstimedelta.HolidayRule(all_holidays)

    # Create the duration of business hours
    business_hours = businesstimedelta.Rules([work_hours, holidays])
    
    def business_hrs(start, end):
        diff_hours = business_hours.difference(start, end)
        return diff_hours.hours + float(diff_hours.seconds) // float(3600)
    
    # Add a new column called BUSINESS_HOURS_ORDER_PICKUP
    df["BUSINESS_HOURS_ORDER_PICKUP"] = df.apply(
        lambda row: business_hrs(row[order], 
                                 row[pickup]), axis=1
    )
    return df

def is_weekend(df, col, new_col):
    """checks if a date in col falls on a weekend or holiday

    Args:
        df (_type_): dataframe to check 
        col (_type_): name of the column to check
        new_col (_type_): name of the new column
    """

    import holidays

    # deep copy
    df = df.copy()

    CA_holidays = holidays.US(subdiv="CA")
    df[new_col] = df[col].apply(lambda date: date not in CA_holidays)
    
    return df

def impute_mileage(df, drop=True):
    """
    Args:
        df (DataFrame): DataFrame to impute
        drop (bool, optional): Whether to drop previously unseen origin-destionation pairs. Defaults to True.
    """

    def impute_mileage_apply(df):

        # deep copy
        df = df.copy()

        if not np.isnan(df["APPROXIMATE_DRIVING_ROUTE_MILEAGE"]):
            return df["APPROXIMATE_DRIVING_ROUTE_MILEAGE"] 
        if not np.isnan(dists[str(df["ORIGIN_CITY"]) + " " + str(df["DESTINATION_CITY"])]):
            return dists[df["ORIGIN_CITY"] + " " + df["DESTINATION_CITY"]]
        if df["ORIGIN_CITY"] == df["DESTINATION_CITY"]:
            return 10
        else:
            return np.nan

    # deep copy
    df = df.copy()

    origin_dest_pair = df["ORIGIN_CITY"] + " " + df["DESTINATION_CITY"]
    mileage = df["APPROXIMATE_DRIVING_ROUTE_MILEAGE"]

    from collections import defaultdict
    dists = defaultdict(lambda : np.nan)
    for pair, dist in zip(origin_dest_pair, mileage):
        if not np.isnan(dist):
            dists[pair] = dist

    df["APPROXIMATE_DRIVING_ROUTE_MILEAGE"] = df.apply(impute_mileage_apply, axis=1)

    if drop:
        df = df[~df.isna()["APPROXIMATE_DRIVING_ROUTE_MILEAGE"]].reset_index(drop=True)

    return df

def popular_cities(df, cols, threshold=0.005):
    """
    Categorizes cities that appear in less than 0.5% of all rows as Others

    Args:
        df (DataFrame): DataFrame to manipulate
        col (list): list of names of the columns to manipulate
        threshold (float, optional): percentage threshold (0<x<1) to define what is "popular". Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    # deep copy
    df = df.copy()

    for col in cols:
        popular_cities = df[col] * (df.groupby(col)[col].transform("count") >= len(df) * threshold)
        df[col] = pd.Series(np.where((popular_cities == ""), "Other", popular_cities))

    return df

def parse_datetime(df, cols=["ORDER_DATETIME_PST", "PICKUP_DEADLINE_PST"]):

    # deep copy
    df = df.copy()

    for col in cols:
        datetime_col = df[col]
        df[col.split("_")[0]+"_DAY"] = datetime_col.dt.dayofweek
        df[col.split("_")[0]+"_MONTH"] = datetime_col.dt.month
        df[col.split("_")[0]+"_HOUR"] = datetime_col.dt.hour

    return df

def assign_ids(df, col):
    """ maps original values in col to new values ranging from 1 to the length of unique values in col

    Args:
        df (dataframe): dataframe to update
        col (str): name of the column to update
    """

    # deep copy
    df = df.copy()

    original_vals = df[col].unique()
    original_to_new = dict(zip(original_vals, range(1, len(original_vals)+1)))
    df[col] = df[col].apply(lambda original_val: original_to_new[original_val])

    return df

def change_to_boolean(df, col, new_col, true_val):
    """changes binary categorical string data to boolean

    Args:
        df (dataframe): dataframe to update
        col (str): name of the column to map
        new_col (str): name of the new column
        true_val (str): the value to be mapped to True
    """

    # deep copy
    df = df.copy()

    df[new_col] = df[col] == true_val
    df = df.drop(col, axis=1)

    return df

def get_coordinates(df, col, new_col, dir="../data/zip3/zip3.shp"):
    """appends a column of coordinates to a dataframe based on the 3-digit zipcode value in "col"

    Args:
        df (dataframe): dataframe to update
        col (str): name of the 3-digit zipcode column
        new_col (str): name of new column
        dir (str): directory of zipcode shapefile
    """

    import geopandas as gpd

    df = df.copy()

    # read 3 zipcode map data
    usa = gpd.read_file(dir)

    # set map type
    usa = usa.to_crs("EPSG:3395")

    # add centroid coordinates
    usa["x"], usa["y"] = usa.centroid.x, usa.centroid.y

    # only show continental USA
    usa = usa.query("STATE not in [\"HI\", \"AK\", \"PR\"]")
    usa = usa[usa["x"]>-1.4*10**7]
    usa = usa[usa["y"]>2.5*10**6]

    # drop irrelevant columns
    usa = usa[["ZIP3", "x", "y"]]

    # merge dataframes
    df = df.merge(usa, left_on=col, right_on="ZIP3").drop("ZIP3", axis=1).rename({"x": new_col+"_X", "y": new_col+"_Y"}, axis=1)

    return df

def extract_from_date(df, col, new_col, kind=["month", "day", "day of week", "week"]):
    """
    extracts temporal features from datetime col and adds new columns

    Args:
        df (dataframe): _description_
        col (str): name of the datetime column
        new_col (str): name of the new colum
        kind (list, optional): types of features to extract. Defaults to ["month", "day", "day of week", "week"].
    """

    df = df.copy()

    if "month" in kind: df[new_col+"_MONTH"] = df[col].dt.month
    if "day" in kind: df[new_col+"_DAY"] = df[col].dt.day
    if "day of week" in kind: df[new_col+"_DAY_OF_WEEK"] = df["CREATED_ON_HQ"].dt.dayofweek
    if "week" in kind: df[new_col+"_WEEK"] = df[col].dt.isocalendar().week

    return df