import numpy as np
from itertools import chain, combinations
import scipy.special
fact = scipy.special.factorial


def f_max(inputs):
    return np.max(inputs)


def f_linear_relu(x, w, b):
    y = np.sum(x*w, -1) + b
    return np.maximum(0, y)


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


def compute_shapley(inputs, f, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(inputs)
    results = np.zeros(inputs.shape)
    n = inputs.shape[0]
    assert inputs.shape == (n,), inputs.shape
    # Create powerset binary mask with shape (2**n, n)
    # Note: we first exclude column with index index, then we add it
    mask = vec_bin_array(np.arange(2 ** (n-1)), n-1)
    assert mask.shape == (2**(n-1), n-1)
    # assert mask.shape == (2**(n-1), n-1), 'Mask shape does not match'
    coeff = (fact(mask.sum(1)) * fact(n - mask.sum(1) - 1)) / fact(n)

    for index in range(n):
        # Copy mask and set the current player active
        mask_wo_index = np.insert(mask, index, np.zeros(2 ** (n-1)), axis=1)
        mask_wi_index = np.insert(mask, index, np.ones(2 ** (n-1)), axis=1)
        # print(mask_wo_index.shape)
        assert mask_wo_index.shape == (2 ** (n - 1), n), 'Mask shape does not match'
        assert np.max(mask_wo_index) == 1, np.max(mask_wo_index)
        assert np.min(mask_wo_index) == 0, np.min(mask_wo_index)

        run_wo_i = f(inputs * mask_wo_index + baseline * (1-mask_wo_index))  # run all masks at once
        run_wi_i = f(inputs * mask_wi_index + baseline * (1-mask_wi_index))  # run all masks at once
        r = (run_wi_i - run_wo_i) * coeff
        results[index] = r.sum()

    return results
