import tensorflow as tf
from tensorflow.python.keras import backend as K


def _arcface(y_true, y_pred, m, s):
    one_hot = tf.identity(y_true)
    m_vector = m * one_hot
    y_pred = tf.acos(y_pred)
    margin_added = y_pred + m_vector
    logit = tf.cos(margin_added)
    rescaled = s * logit
    return K.categorical_crossentropy(target=y_true, output=rescaled, from_logits=True)
