import os
import sys
import json
import pandas as pd
from src.models import hybrid_model


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

    if "all" in targets:
        # get predicted values
        preds = hybrid_model.get_weighted_costs(config_dir)
        preds = [str(pred) for pred in preds]
        preds_str = "\n".join(preds)

        f = open("output/preds.txt", "w")
        f.write(preds_str)
        f.close()

        print("Check ./output directory for predicted values.")
        


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
