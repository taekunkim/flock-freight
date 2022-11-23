import sys
import json
import pandas as pd


def main():

    with open("config/stub.json", "r") as f:
        config = json.load(f)

    if state == "test":
        data_dir = "./test/testdata/test.csv"
    else:
        data_dir = config["data_dir"]

    data = pd.read_csv(data_dir)

    print(data)


if __name__ == "__main__":
    state = ""
    args = sys.argv[1:]
    if "test" in args:
        state = "test"
    else:
        state = "prod"

    main()
