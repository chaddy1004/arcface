import tensorflow as tf
from tensorflow.keras.layers import Wrapper, Layer


class L2Normalize(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.layer = layer
        # self.name = self.layer.name + "_l2normed"

    def build(self, input_shape=None):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        self._update_kernel()
        return self.layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def _update_kernel(self):
        w = self.layer.kernel
        self.layer.kernel = tf.math.l2_normalize(w)


class Arccos(Layer):
    def __init__(self, index, m, s, **kwargs):
        self.index = m
        self.m = m
        self.s = s
        super(Arccos, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Arccos, self).build(input_shape)

    def call(self, inputs, **kwargs):
        m_one_hot = tf.one_hot(indices=self.index, on_value=self.m, off_value=0)
        x = tf.math.acos(inputs)
        x = tf.math.add(x, m_one_hot)  # only add to the index of the current variable
        x = tf.math.scalar_mul(self.s, x)
        return x
