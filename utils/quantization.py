import tensorflow as tf


def quantize(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model


def save_quantized_model(tflite_model, filename):
    open(filename, "wb").write(tflite_model)
    print(f"Quantized model saved to {filename}")
