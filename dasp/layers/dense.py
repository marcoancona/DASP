import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from lpdn import filter_activation


class ProbDenseInput(Dense):
    def __init__(self, units, **kwargs):
        super(ProbDenseInput, self).__init__(units, **kwargs)

    def build(self, input_shape):
        return super(ProbDenseInput, self).build(input_shape[0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbDenseInput, self).compute_output_shape(input_shape[0])
        return original_output_shape + (4,)

    def assert_input_compatibility(self, inputs):
        return super(ProbDenseInput, self).assert_input_compatibility(inputs[0])

    def call(self, inputs):

        inputs, mask, k = inputs

        ghost = tf.ones_like(inputs) * (1.0 - mask)
        inputs_i = inputs * (1.0 - mask)

        dot = K.dot(inputs, self.kernel)
        dot_i = K.dot(inputs_i, self.kernel)
        dot_mask = K.dot(ghost, tf.ones_like(self.kernel))
        dot_v = K.dot(inputs_i**2, self.kernel**2)
        # Compute mean without feature i
        mu = dot_i / dot_mask
        v = dot_v / dot_mask - mu ** 2
        # Compensate for number of players in current coalition
        mu1 = mu * k
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (dot - dot_i)
        # Compensate for number or players in the coalition
        v1 = v * k * (1.0 - (k-1) / (dot_mask - 1))
        # Set something different than 0 if necessary
        v1 = tf.maximum(0.00001, v1)
        # Since player i is only a bias, at this point the variance of the distribution that
        # includes it is the same
        v2 = tf.identity(v1)

        if self.use_bias:
            mu1 = K.bias_add(mu1, self.bias, data_format='channels_last')
            mu2 = K.bias_add(mu2, self.bias, data_format='channels_last')

        mu1, v1 = filter_activation(self.activation.__name__, mu1, v1)
        mu2, v2 = filter_activation(self.activation.__name__, mu2, v2)
        return tf.stack([mu1, v1, mu2, v2 ], -1)
