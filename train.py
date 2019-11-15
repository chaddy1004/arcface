from network.network import Res50
from utils.process_config import process_config
import argparse

def main(config_file: str):
    config = process_config(config_file)
    backbone = Res50(config=config).define_backbone(model_name="backbone")
    full = Res50(config=config).define_full(backbone=backbone, model_name="full")
    backbone.summary()
    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    full.summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file = args["config"])