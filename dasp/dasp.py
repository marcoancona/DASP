import logging
import numpy as np
from .coalition_policies.default import DefaultPlayerIterator
from .layers.convolution import ProbConv2DInput
from .layers.dense import ProbDenseInput
from keras.models import  Model
from keras.layers import Dense, Input, Conv2D, Conv1D, Lambda
from lpdn import convert_to_lpdn


class DASP():
    def __init__(self, keras_model, player_generator=None, input_shape=None):
        self.keras_model = keras_model
        self.player_generator = player_generator
        self.input_shape = input_shape
        self.inputs = None
        if self.input_shape is None:
            self.input_shape = keras_model.layers[0].input_shape[1:]
            logging.info("Inferred input shape: " + str(self.input_shape))
        self.dasp_model = self._build_dasp_model()

    def model_summary(self):
        if self.dasp_model is not None:
            self.dasp_model.summary()

    def run(self, x, steps=None):
        player_generator = self.player_generator
        if player_generator is None:
            player_generator = DefaultPlayerIterator(x)
        player_generator.set_n_steps(steps)
        ks = player_generator.get_steps_list()

        logging.info("DASP: testing %d coalitions sizes:" % len(ks))
        if len(ks) < 10:
            logging.info(ks)
        else:
            logging.info("%d %d ... %d %d" % (ks[0], ks[1], ks[-2], ks[-1]))

        result = None
        tile_input = [len(ks)] + (len(x.shape) - 1) * [1]
        tile_mask = [len(ks)*x.shape[0]] + (len(x.shape) - 1) * [1]

        for i, (mask, mask_output) in enumerate(player_generator):
            # Workaround: Keras does not seem to support scalar inputs
            inputs = [np.tile(x, tile_input), np.tile(mask, tile_mask), np.repeat(ks, x.shape[0])]
            print (tile_input)
            print(tile_mask)
            print ([x.shape for x in inputs])
            y1, y2 = self.dasp_model.predict(inputs)
            y1 = y1.reshape(len(ks), x.shape[0], -1, 2)
            y2 = y2.reshape(len(ks), x.shape[0], -1, 2)
            y = np.mean(y2[..., 0] - y1[..., 0], 0)
            if np.isnan(y).any():
                raise RuntimeError('Result contains nans! This should not happen...')

            # Compute Shapley Values as mean of all coalition sizes
            if result is None:
                result = np.zeros(y.shape + mask_output.shape)

            shape_mask = [1] * len(y.shape)
            shape_mask += list(mask_output.shape)

            shape_out = list(y.shape)
            shape_out += [1] * len(mask_output.shape)

            result += np.reshape(y, shape_out) * mask_output
        return result

    def _build_dasp_model(self):
        # Create an equivalent probabilistic model.
        prob_input = Input(shape=self.input_shape)
        player_mask = Input(shape=self.input_shape)
        coalition_size = Input(shape=(1,))

        self.inputs = [prob_input, player_mask, coalition_size]

        # Find first layer on input model
        first_hidden_layer_name = None
        first_layer_output = None
        for li, l in enumerate(self.keras_model.layers):
            if isinstance(l, Dense):
                first_layer_output = ProbDenseInput(l.units, activation=l.activation, name=l.name)(self.inputs)
            elif isinstance(l, Conv2D):
                first_layer_output = ProbConv2DInput(l.filters, l.kernel_size, padding=l.padding,
                                    activation=l.activation, name=l.name)(self.inputs)
            elif isinstance(l, Conv1D):
                pass
            if first_layer_output is not None:
                first_hidden_layer_name = l.name
                break

        # Get full LPDN model
        lpdn_model = convert_to_lpdn(self.keras_model)

        # Disassemble layers and remove first hidden one
        lpdn_layers = [l for l in lpdn_model.layers]
        while True:
            l = lpdn_layers.pop(0)
            if l.name == first_hidden_layer_name:
                break

        # Now stack everything back, adding new the new DASP input layer
        y1 = Lambda(lambda x: x[..., 0:2])(first_layer_output)
        y2 = Lambda(lambda x: x[..., 2:4])(first_layer_output)
        for i in range(len(lpdn_layers)):
            y1 = lpdn_layers[i](y1)
        for i in range(len(lpdn_layers)):
            y2 = lpdn_layers[i](y2)

        # Declare the model
        model = Model(inputs=self.inputs, outputs=[y1,y2])

        # Load weights of original Keras model.
        # Easiest way is to save them to file and reload them.
        self.keras_model.save_weights('_dasp_tmp.h5')
        model.load_weights('_dasp_tmp.h5', by_name=True)
        return model
