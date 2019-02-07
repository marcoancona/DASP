import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Conv1D
from lpdn import filter_activation


class ProbConv2DInput(Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ProbConv2DInput, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        print (input_shape)
        return super(ProbConv2DInput, self).build(input_shape[0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbConv2DInput, self).compute_output_shape(input_shape[0])
        return original_output_shape + (4,)

    def assert_input_compatibility(self, inputs):
        return super(ProbConv2DInput, self).assert_input_compatibility(inputs[0])

    def _conv2d(self, input, kernel):
        return K.conv2d(input, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)

    def call(self, inputs):
        inputs, mask, k = inputs
        print ("Hello")
        n_players =  tf.reduce_sum(tf.ones_like(inputs[0]))
        size_coalition = tf.reduce_sum(mask[0])
        n_players = n_players / size_coalition
        #z = tf.Print(size_coalition, [size_coalition])
        #z2 = tf.Print(n_players, [n_players])

        # When I say k, I actually mean k coalition-players so need to compensate for it
        k = tf.expand_dims(tf.expand_dims(k, -1), -1)

        ghost = tf.ones_like(inputs) * (1.0 - mask)
        inputs_i = inputs * (1.0 - mask) #+ 0.0*(z+z2)

        conv = self._conv2d(inputs, self.kernel)
        conv_i = self._conv2d(inputs_i, self.kernel)
        conv_count = self._conv2d(ghost, tf.ones_like(self.kernel))
        conv_v = self._conv2d(inputs_i**2, self.kernel**2)

        # Compute mean without feature i
        #conv_mask = tf.maximum(0.01, conv_mask)
        mu_t = conv_i / conv_count #conv_count
        # Compensate for number of players in current coalition
        mu1 = mu_t * conv_count * (k / n_players)
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (conv - conv_i)
        # Compute variance without player i
        v1 = conv_v / conv_count - mu_t ** 2

        # Compensate for number or players in the coalition
        k = conv_count * (k / n_players)
        v1 = v1 * k  * (1.0 - (k-1) / (conv_count - 1))
        # Set something different than 0 if necessary
        #v1 = K.maximum(0.00001, v1)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if self.use_bias:
            mu1 = K.bias_add(
                mu1,
                self.bias,
                data_format=self.data_format)
            mu2 = K.bias_add(
                mu2,
                self.bias,
                data_format=self.data_format)

        print(self.activation.__name__)

        mu1, v1 = filter_activation(self.activation.__name__, mu1, v1)
        mu2, v2 = filter_activation(self.activation.__name__, mu2, v2)
        return tf.stack([mu1, v1, mu2, v2], -1)


class ProbConv1DInput(Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ProbConv1DInput, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        print (input_shape)
        return super(ProbConv1DInput, self).build(input_shape[0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbConv1DInput, self).compute_output_shape(input_shape[0])
        return original_output_shape + (4,)

    def assert_input_compatibility(self, inputs):
        return super(ProbConv1DInput, self).assert_input_compatibility(inputs[0])

    def _conv1d(self, input, kernel):
        return K.conv1d(input, kernel, self.strides, self.padding, self.data_format)

    def call(self, inputs):
        inputs, mask, k = inputs

        n_players =  tf.reduce_sum(tf.ones_like(inputs[0]))
        size_coalition = tf.reduce_sum(mask[0])
        n_players = n_players / size_coalition
        z = tf.Print(size_coalition, [size_coalition])
        z2 = tf.Print(n_players, [n_players])

        ghost = tf.ones_like(inputs) * (1.0 - mask)
        inputs_i = inputs * (1.0 - mask) + 0.0*(z+z2)

        conv = self._conv1d(inputs, self.kernel)
        conv_i = self._conv1d(inputs_i, self.kernel)
        conv_count = self._conv1d(ghost, tf.ones_like(self.kernel))
        conv_v = self._conv1d(inputs_i**2, self.kernel**2)

        k = tf.expand_dims(k, -1)

        # Compute mean without feature i
        #conv_mask = tf.maximum(0.01, conv_mask)
        mu_t = conv_i / conv_count #conv_count
        # Compensate for number of players in current coalition
        mu1 = mu_t * conv_count * (k / n_players)
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (conv - conv_i)
        # Compute variance without player i
        v1 = conv_v / conv_count - mu_t ** 2

        # Compensate for number or players in the coalition
        k = conv_count * (k / n_players)
        v1 = v1 * k  * (1.0 - (k-1) / (conv_count - 1))
        # Set something different than 0 if necessary
        #v1 = K.maximum(0.00001, v1)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if self.use_bias:
            mu1 = K.bias_add(
                mu1,
                self.bias,
                data_format=self.data_format)
            mu2 = K.bias_add(
                mu2,
                self.bias,
                data_format=self.data_format)

        mu1, v1 = filter_activation(self.activation.__name__, mu1, v1)
        mu2, v2 = filter_activation(self.activation.__name__, mu2, v2)
        print (mu1)
        return tf.stack([mu1, v1, mu2, v2], -1)