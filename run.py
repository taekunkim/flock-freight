import os
import sys
import json
import pandas as pd
import numpy as np

from src.models import hybrid_model
from src.features import data_for_modelling
from src.models import pooled_proba_model
from src.models import pooled_regression_model
from src.models import quote_regression_model
from src.features import data_for_modelling

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
    else:
        config_dir = "./config/config.json"

    if True:
        # get predicted values

        # get data
        orders_raw, offers_raw = data_for_modelling.get_raw_data(config_dir)

        # clean data
        _, _, merged = data_for_modelling.clean_all(orders_raw, offers_raw)

        # split offers into train test
        merged_train, merged_test = data_for_modelling.split_train_test(merged)

        # create pipelines
        pooled_proba_pl, (_, _) = pooled_proba_model.generate_probability_pipeline(merged_train)
        pooled_reg_pl, (_, _) = pooled_regression_model.generate_pooled_regression_pipeline(merged_train)
        quote_reg_pl, (_, _) = quote_regression_model.generate_quote_regression_pipeline(merged_train)

        # get testset preds
        preds = hybrid_model.get_weighted_costs(merged_test, pooled_proba_pl, pooled_reg_pl, quote_reg_pl)
        preds_str = [str(pred) for pred in preds]
        preds_str = "\n".join(preds_str)

        # get diffrence between the model's predicted cost and that of FlockFreight model
        abs_diff = np.abs(merged_test["ESTIMATED_COST_AT_ORDER"] - preds)
        abs_diff_str = [str(diff) for diff in abs_diff]
        abs_diff_str = "\n".join(abs_diff_str)


        f = open("output/predicted_cost.txt", "w")
        f.write(preds_str)
        f.close()

        f = open("output/abs_diff.txt", "w")
        f.write(abs_diff_str)
        f.close()

        f = open("output/mean_abs_diff.txt", "w")
        f.write(str(abs_diff.mean()))
        f.close()

        print("Check ./output directory for predicted values.")
        
if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
