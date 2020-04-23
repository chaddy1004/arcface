from utils.latent_space_util import embedding_dump, create_embeddings, create_sprite, create_metadata
from utils.process_config import process_config
import os
from dataloader.dataloader import Dataloader
from tensorflow.python.keras.models import load_model
import argparse
import tensorflow as tf
from utils.layers import L2Normalize
from utils.losses import __arcface
from functools import partial


def load_keras_model(saved_model_path, custom_objects={}):
    # add custom layer to custom_objects
    return load_model(
        saved_model_path,
        custom_objects=custom_objects
    )


def main(config_file: str):
    config = process_config(config_file)
    data_loader = Dataloader(config=config, init_train = False).get_test_data_generator()
    arcface = partial(__arcface, m=config.model.hyperparameters.m, s=config.model.hyperparameters.s)
    custom_objects = {'tf': tf, 'L2Normalize': L2Normalize, 'arcface_loss': arcface}

    filename = os.path.join(config.exp.saved_model_dir, "model.hdf5")
    vae = load_keras_model(saved_model_path=filename, custom_objects=custom_objects)
    encoder = vae.get_layer("backbone")
    os.makedirs(config.exp.embedding_dir, exist_ok=True)
    embedding_array, data_array, label_list = create_embeddings(data_loader=data_loader, encoder=encoder, embeddings_dir=config.exp.embedding_dir)
    create_sprite(data_array=data_array, embeddings_dir=config.exp.embedding_dir)
    create_metadata(label_list=label_list, embeddings_dir=config.exp.embedding_dir)
    embedding_dump(embeddings=embedding_array, embeddings_dir=config.exp.embedding_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"])