import argparse

from dataloader.dataloader import Dataloader
from model_and_trainer_builder import build_model_and_trainer
from utils.process_config import process_config


def main(config_file: str):
    config = process_config(config_file)
    data_loader = Dataloader(config=config)
    backbone, full, trainer = build_model_and_trainer(config=config, data_loader=data_loader)
    backbone.summary()
    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    full.summary()
    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"])
