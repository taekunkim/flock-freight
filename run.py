import os
import sys
import json
import pandas as pd
import numpy as np

from src.models import estimated_minimum_cost
from src.models import offer_number_model
from src.models import is_minimum_model
from src.features import data_for_modelling
from sklearn.metrics import accuracy_score, confusion_matrix


def main(targets):

    if "clean" in targets:
        output_path = "./output"

        if os.path.exists(output_path):
            files = os.listdir(output_path)

            if len(files) > 0:
                for f in os.listdir(output_path):
                    os.remove(os.path.join(output_path, f))
            
        else: 
            os.makedirs(output_path)

    if "test" in targets: 
        config_dir = "./config/config_test.json"
        is_test_run = True
    else:
        config_dir = "./config/config.json"
        is_test_run = False

    if True:
        # get predicted values

        # get data
        orders_raw, offers_raw = data_for_modelling.get_raw_data(config_dir)

        # clean data
        _, _, merged = data_for_modelling.clean_all(orders_raw, offers_raw)

        merged = data_for_modelling.remove_outliers(merged)

        orders = data_for_modelling.aggregate_data(merged)

        orders = data_for_modelling.merge_order_offer_on_reference(merged, orders)

        merged = data_for_modelling.Kmeans_clustering(merged, orders, is_test_run = True)

        merged = data_for_modelling.min_rate_df(merged)

        merged = data_for_modelling.generate_is_minimum_col(merged)

        # split offers into train test
        merged_train, merged_test = data_for_modelling.split_train_test(merged)
        orders_train, orders_test = data_for_modelling.split_train_test(orders)
        
        offer_number_pl, offer_number_df, offer_number_mse = offer_number_model.generate_offer_number_pipeline(merged_train, orders_train, is_test_run = is_test_run)

        estimated_minimum_model, estimated_minimum_df, estimated_minimum_mse = estimated_minimum_cost.generate_estimated_minimum_cost_pipeline(offer_number_df)

        is_minimum_pl, is_minimum_df, accuracy = is_minimum_model.generate_proba_minimum_pipeline(estimated_minimum_df)

        # get testset preds
        # print("train")
        # print(merged_train['ORDER_DAY'])
        # print(merged_train['ORDER_DAY_OF_WEEK'])
        # print("test")
        # print(merged_test['ORDER_DAY'])
        # print(merged_test['ORDER_DAY_OF_WEEK'])
        merged_test["PREDICTED_OFFER_COUNT"] = offer_number_pl.predict(merged_test)
        merged_test["PREDICTED_MIN_RATE"] = estimated_minimum_model.predict(merged_test)
        merged_test["PREDICTED_IS_MIN"] = is_minimum_pl.predict(merged_test)

        preds = merged_test['PREDICTED_IS_MIN']
        preds_str = [str(pred) for pred in preds]
        preds_str = "\n".join(preds_str)

        f = open("output/predicted_is_minimum.txt", "w")
        f.write(preds_str)
        f.close()

        c_matrix = confusion_matrix(merged_test["IS_MINIMUM"], merged_test["PREDICTED_IS_MIN"])

        tp = c_matrix[0][0]
        fp = c_matrix[0][1]
        fn = c_matrix[1][0]
        tn = c_matrix[1][1]

        accuracy = str((tp + tn) / (tp + fp + fn + tp))
        sensitivity = str((tp) / (tp + fn))
        specificity = str((tn) / (tn + fp))

        f = open("output/accuracy.txt", "w")
        f.write(accuracy)
        f.close()

        f = open("output/sensitivity.txt", "w")
        f.write(sensitivity)
        f.close()

        f = open("output/specificity.txt", "w")
        f.write(specificity)
        f.close()

        print("Check ./output directory for predicted values.")
        
if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
