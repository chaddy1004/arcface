import tensorflow as tf
import argparse
import os
from utils.process_config import process_config
from utils.layers import L2Normalize
from utils.quantization import quantize, save_quantized_model


def main(config_file, model_name):
    config = process_config(config_file)
    model_file = os.path.join(config.exp.saved_model_dir, model_name)
    custom_objects = {"tf": tf, "L2Normalize": L2Normalize}

    model = tf.keras.models.load_model(filepath=model_file, custom_objects=custom_objects, compile=False)
    tflite_model = quantize(model=model)
    quantized_model_file = os.path.join(config.exp.saved_model_dir, f"{model_name}_quantized")
    save_quantized_model(tflite_model=tflite_model, filename=quantized_model_file)
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="config path to use")
    ap.add_argument("--model_name", type=str, default="model.hdf5", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"], model_name=args["model_name"])
