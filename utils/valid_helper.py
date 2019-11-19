import tensorflow as tf
import numpy as np


def cos_similarity(src_feature, tar_feature):
    return tf.keras.backend.batch_dot(src_feature, tar_feature)


def face_match(src_feature, tar_feature, threshold=0.3):
    sim = cos_similarity(src_feature=src_feature, tar_feature=tar_feature)
    return sim <= threshold # for cosine similarity, smaller is better
