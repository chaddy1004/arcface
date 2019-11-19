import tensorflow as tf
import argparse
import os
from utils.process_config import process_config
from utils.layers import L2Normalize


def main(config_file, model_name):
    config = process_config(config_file)
    model_file = os.path.join(config.exp.saved_model_dir, model_name)
    custom_objects = {"tf": tf, "L2Normalize": L2Normalize}
    model = tf.keras.models.load_model(filepath=model_file, custom_objects=custom_objects, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    ap.add_argument("--model_name", type=str, default="model.hdf5", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"], model_name=args["model_name"])
