import tensorflow as tf
import argparse
import os


def main(config_file, model_name):

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    ap.add_argument("--model_name", type=str, default="model.hdf5", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"], model_name=args["model_name"])
    main()