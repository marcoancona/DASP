import numpy as np
from keras_applications import inception_v3
# Implementation of "Polynomial calculation of the Shapley value based on sampling"
# https://www.sciencedirect.com/science/article/pii/S0305054808000804

def _finalize(result, runs, shape):
    shapley = result.copy() / runs
    return shapley.reshape(shape)


def run_shapley_sampling(model, xs, ys, feat_dims, runs=20, callback=None):
    """

    :param model: Keras model
    :param xs: input to test
    :param ys: gt labels (one hot encoded)
    :param feat_dims: list of dimensions representing features. eg [1, 2] to remove pixels when input is images
    :param runs: number of network runs per feature
    :return:
    """

    xs_shape = list(xs.shape)
    print("Data shape: ", xs_shape)
    print ("Runs", runs)
    n_features = np.prod(xs.shape[1:])
    if feat_dims:
        n_features = np.prod([xs.shape[i] for i in feat_dims])
    result = np.zeros((xs_shape[0], n_features))

    run_shape = xs_shape.copy()
    run_shape = np.delete(run_shape, feat_dims).tolist()
    run_shape.insert(1, -1)
    print("Shape to mask features: ", run_shape)

    reconstruction_shape = [xs_shape[0]]
    for j in feat_dims:
        reconstruction_shape.append(xs_shape[j])

    next_callback = 1
    for r in range(runs):
        # Select randomly 1 feature to test

        p = np.random.permutation(n_features)
        x = xs.copy().reshape(run_shape)

        y = None
        for i in p:
            if y is None:
                y = model.predict(x.reshape(xs_shape))
            x[:, i] = 0
            y0 = model.predict(x.reshape(xs_shape))
            assert y0.shape == ys.shape, y0.shape
            prediction_delta = np.sum((y - y0) * ys, 1)
            result[:, i] += prediction_delta
            y = y0

        if (r+1) == next_callback and callback is not None:
            next_callback = next_callback * 2
            callback((r+1, _finalize(result, r+1, reconstruction_shape)))

    return _finalize(result, runs, reconstruction_shape)
