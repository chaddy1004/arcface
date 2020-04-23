import tensorflow as tf
from tensorflow.keras.layers import Wrapper, Layer


class L2Normalize(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.layer = layer
        # self.name = self.layer.name + "_l2normed"

    def build(self, input_shape=None):
        super().build(input_shape)
        self.layer_depth = int(self.layer.kernel.shape[-1])
        self.kernel_norm_axes = list(range(self.layer.kernel.shape.rank - 1))

    def call(self, inputs, **kwargs):
        self._update_kernel() # THIS IS THE PROBLEM! WHEN THIS IS COMMENTED IT WORKS!!!
        output = self.layer(inputs)
        print(f"output:{output.shape}", tf.norm(output[0]), tf.norm(output[10]))
        return output
        # # self._update_kernel()
        # print("asdf", tf.norm(inputs[0]), inputs[0].shape, tf.norm(inputs[10]), inputs[10].shape)
        # with tf.name_scope('compute_weights'):
        #     # Replace kernel by normalized weight variable.
        #     # self.layer.kernel = tf.nn.l2_normalize(
        #     #     self.v, axis=self.kernel_norm_axes)
        #     self.layer.kernel = tf.nn.l2_normalize(self.v)
        #     print("input", tf.norm(inputs))
        #     # Ensure we calculate result after updating kernel.
        #     update_kernel = tf.identity(self.layer.kernel)
        #     # with tf.control_dependencies([update_kernel]):
        #     outputs = self.layer(inputs)
        #     print("output", tf.norm(outputs[0]), tf.norm(outputs[10]))
        #     return outputs

    def _update_kernel(self):
        with tf.name_scope('compute_weights'):
            v = tf.identity(self.layer.kernel)
            self.layer.kernel = tf.nn.l2_normalize(v, -1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


class _L2Normalize(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.layer = layer
        # self.name = self.layer.name + "_l2normed"

    def build(self, input_shape=None):
        super().build(input_shape)
        self.layer_depth = int(self.layer.kernel.shape[-1])
        self.kernel_norm_axes = list(range(self.layer.kernel.shape.rank - 1))
        self.v = self.layer.kernel

    def call(self, inputs, **kwargs):
        # self._update_kernel()
        # print("asdf", tf.norm(inputs[0]), inputs[0].shape, tf.norm(inputs[10]), inputs[10].shape)
        with tf.name_scope('compute_weights'):
            # Replace kernel by normalized weight variable.
            # self.layer.kernel = tf.nn.l2_normalize(
            #     self.v, axis=self.kernel_norm_axes)
            self.layer.kernel = tf.nn.l2_normalize(self.v)
            # Ensure we calculate result after updating kernel.
            update_kernel = tf.identity(self.layer.kernel)
            # with tf.control_dependencies([update_kernel]):
            outputs = self.layer(inputs)
            print(f"output{outputs.shape}", tf.norm(outputs[0]), tf.norm(outputs[10]))
            return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
