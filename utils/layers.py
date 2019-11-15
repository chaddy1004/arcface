import tensorflow as tf
from tensorflow.keras.layers import Wrapper, Layer




class L2Normalize(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.layer = layer
        # self.name = self.layer.name + "_l2normed"

    def build(self, input_shape=None):
        self.layer.build(input_shape)

    def call(self, inputs, **kwargs):
        self._update_kernel()
        return self.layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def _update_kernel(self):
        w = self.layer.kernel
        self.layer.kernel = tf.math.l2_normalize(w)

class Arccos(Layer):
    def __init__(self, m, s,  **kwargs):
        self.m = m
        self.s = s
        super(Arccos, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Arccos, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.math.acos(inputs)
        x = tf.math.add(x, self.m)
        x = tf.math.scalar_mul(self.m,x)
        return x

