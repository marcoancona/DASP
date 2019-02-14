import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import scipy.misc

CORRUPTION_MAX_RATIO = 1
REMOVE_VALUE = 0
RANDOM_TESTS = 3

SAVE_IMAGES = 0
path = None


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _run_model(x, y, model, mode):
    if mode == 'accuracy':
        raise RuntimeError('Not supported')
    elif mode == 'prediction':
        p = model.predict(x, batch_size=64, verbose=0)
        p = np.sum(p*y, 1)
        return p
    else:
        raise RuntimeError('Mode non valid')


def _corrupt_loop(eval_x, eval_y, rank, model, feat_per_step, mode, method_name=None):
    x_neg = REMOVE_VALUE
    x = eval_x
    input_shape = eval_x.shape
    batch_size = input_shape[0]

    map_size = rank.shape
    features_count = np.prod(map_size[1:])
    metric = [_run_model(x, eval_y, model, mode=mode)]
    rank_flatten = rank.reshape((batch_size, -1))

    steps = int(features_count * CORRUPTION_MAX_RATIO / feat_per_step)

    for i in range(steps):
        batch_mask = np.zeros_like(rank_flatten, dtype=np.bool)
        batch_mask[[(rank_flatten >= i * feat_per_step) * (rank_flatten < (i+1)*feat_per_step)]] = True
        batch_mask = batch_mask.reshape(map_size)
        x = ~batch_mask * x + batch_mask * x_neg
        if i < SAVE_IMAGES:
            scipy.misc.imsave(path+'/img_%s_%3d.png' % (method_name if method_name is not None else '', i), x[0])
        metric.append(_run_model(x, eval_y, model, mode=mode))
    return np.array(metric)


def run_robustness_test(model, x, y, original_maps, names, task_name, feat_per_step,
                        result_path='.', mode='accuracy', reduce_dim=None):
    print('Running robustness test...')
    assert len(original_maps) > 0
    maps = [np.copy(m) for m in original_maps]

    global path
    path = result_path + '/' + task_name
    _ensure_dir(path)

    batch_size = len(x)
    map_shape = maps[0].shape
    saliency_length = np.prod(map_shape[1:])

    # do reduce_sum in the beginning
    if reduce_dim is not None:
        print ('--> using reduce_sum of dimension {} of heatmap'.format(reduce_dim))
        maps = [np.sum(m, reduce_dim, keepdims=True) for m in maps]
        map_shape = maps[0].shape
        saliency_length = np.prod(map_shape[1:])

    if len(x.shape) != len(map_shape):
        print ('Input shape (%s) is not equal to map shape (%s)' % (x.shape, map_shape))
        if len(x.shape) == len(map_shape) - 1 and map_shape[-1] == 1:
            print ('Trying to squeeze maps...')
            maps = [m.squeeze(-1) for m in maps]
            map_shape = maps[0].shape


    plot_data_max = {}
    colors_max = ['#ee1234', '#CD4075', '#cc6600', '#d07c7c', '#5f0f40', '#6E2A00', '#3e1591', '#e5ac10']
    colors_min = ['#03797c', '#5b5aac', '#009fcb', '#676767', '#4285f4', '#141a30']
    markers = ['^', 'v', 'o', 's', '*', '', '', '', '']
    msizes = [4, 4, 3, 3, 5, 3, 3, 3, 3]

    plot_data_min = {}
    plot_data_other = {}

    for map, name in zip(maps, names):
        map = map.reshape(map_shape)
        map_flat = map.reshape(batch_size, -1)
        rank_max = np.argsort(np.argsort(map_flat * -1.0, axis=1), axis=1).reshape(map_shape)
        plot_data_max[name] = _corrupt_loop(x, y, rank_max, model, feat_per_step, mode=mode, method_name=name)

    # Do random
    rand_results = []
    for j in range(RANDOM_TESTS):
        rank_rand = np.stack([np.random.permutation(np.arange(0, saliency_length)) for i in range(len(x))]).reshape(map_shape)
        rand_results.append(_corrupt_loop(x, y, rank_rand, model, feat_per_step, mode=mode, method_name='rand'))
    plot_data_other['rand'] = np.mean(rand_results, 0)
    plot_data_other['_rand_std'] = np.std(rand_results, 0)

    n_ticks = len(plot_data_other['rand'])
    x_ticks = np.linspace(0, 100, n_ticks)

    fig = plt.figure(figsize=(14, 6))

    for idx, map_name in enumerate(plot_data_max):
        y = plot_data_max[map_name].mean(1) - plot_data_max[map_name].mean(1)[0]
        plt.plot(x_ticks, y, color=colors_max[idx],
                label=map_name,
                linewidth=2,
                marker=markers[idx],
                markersize=msizes[idx],
                markevery=40 + np.random.randint(-5, 5),
                alpha=1)

    for map_name in plot_data_other:
        if map_name[0] != '_':
            y = plot_data_other[map_name].mean(1) - plot_data_other[map_name].mean(1)[0]
            plt.plot(x_ticks, y, '--', color='green',
                     label=map_name,
                     linewidth=3,
                     alpha=1)

    frm = ScalarFormatter()
    frm.set_scientific(False)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(frm)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True)
    plt.subplots_adjust(bottom=0.30)
    fig.text(0.5, 0.22, 'Features removed (%)', ha='center', va='center', fontsize=13)
    fig.text(0.08, 0.55, '$Δf_c(x)$', ha='center', va='center', rotation='vertical', fontsize=13)
    fig.legend(names + ['Random'], loc='lower center', fontsize=12, ncol=5, bbox_to_anchor=(0, 0, 1, 1),
                         bbox_transform=plt.gcf().transFigure)
    plt.show()
