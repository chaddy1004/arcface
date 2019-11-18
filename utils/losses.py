import tensorflow as tf
from tensorflow.python.keras import backend as K


def _arcface(y_true, y_pred, m, s):
    one_hot = tf.identity(y_pred)
    m_vector = m * one_hot
    y_true = tf.acos(y_true)
    margin_added = y_true + m_vector
    rescaled = s * margin_added
    return K.categorical_crossentropy(target=y_true, output=rescaled, from_logits=True)
