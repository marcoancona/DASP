from skimage import feature, transform
import numpy as np
import matplotlib.pyplot as plt
import scipy
import datetime
import os
from keras.applications import vgg16
from matplotlib.ticker import FormatStrFormatter


COLORS = ["#4e79a7",
          "#59a14f",
          "#9c755f",
          "#edc948",
          "#bab0ac",
          "#e15759",
          "#b07aa1",
          "#f28e2b"]

def color_for_label(label):
    l = label.lower()
    if "integrated" in l:
        return COLORS[0]
    elif "occlusion" in l:
        return COLORS[1]
    elif "revcancel" in l:
        return COLORS[2]
    elif "mix" in l:
        return "#666666"
    elif "rescale" in l:
        return COLORS[5]
    elif "ours" in l or "dasp" in l:
        return "orange"
    elif "sampling" in l:
        return COLORS[6]
    else:
        return "#FFFFFF"


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_plot_filename(plot_name, experiment_name):
    folder = "results/" + experiment_name + "/"
    _ensure_dir(folder)
    t = datetime.datetime.now().isoformat()[:-7]
    return folder + plot_name + "_" + experiment_name + "_" + t + ".pdf"

def _isiterable(p_object):
    try:
        it = iter(p_object)
    except TypeError:
        return False
    return True

def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


def plot_attribution_maps(name, xs, attributions, names, idxs=None, percentile=100, dilation=3.0, alpha=0.8, show_original=False):
    if idxs is None:
        idxs = list(range(len(attributions[0])))
    if show_original:
        names.insert(0, 'Input')
    rows = len(idxs)
    cols = len(names)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 2*rows))
    for ax, col in zip(axes[0] if _isiterable(axes[0]) else axes, names):
        ax.set_title(col, fontsize= 14)


    for i, idx in enumerate(idxs):
        if show_original:
            ax = axes[i, 0] if _isiterable(axes[0]) else axes[0]
            ax.axis('off')
            ax.imshow(xs[idx])

        for j, attribution in enumerate(attributions):
            if show_original:
                j += 1
            ax = axes[i, j] if _isiterable(axes[0]) else axes[j]
            attribution_map = attribution[idx]
            original_sample = xs[idx]
            plot(attribution_map, original_sample, axis=ax, percentile=percentile, dilation=dilation, alpha=alpha)
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.2)

    plt.show()
    fig.savefig(get_plot_filename('maps', name), bbox_inches = "tight")


def _compare_attributions(attributions, metric='mse'):
    n = attributions[0].shape[0]
    values = np.zeros((len(attributions), len(attributions), n))

    for i, a1 in enumerate(attributions):
        for j, a2 in enumerate(attributions):
            for idx in range(n):
                if i >= j:
                    x1 = np.array(a1[idx]).reshape(-1)
                    x2 = np.array(a2[idx]).reshape(-1)
                    v = np.nan
                    if metric == 'mse':
                        v = (np.mean((x1-x2)**2.0))**0.5
                    elif metric == 'corr':
                        v = scipy.stats.spearmanr(x1, x2)[0]
                    values[i, j, idx] = v
                    values[j, i, idx] = values[i, j, idx, ]
    return values


def _plot_boxplot(plot_data, permuted_names, yaxis, experiment_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    bplot = ax.boxplot(plot_data, showfliers=False, patch_artist=True, medianprops=dict(color="#FF0000AA"), )
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # fill with colors
    for bplot in (bplot, ):
        for patch, name in zip(bplot['boxes'], permuted_names):
            patch.set_facecolor(color_for_label(name))

    ## change color and linewidth of the whiskers
    for whisker in bplot['whiskers']:
        whisker.set(color='#222222', linewidth=1)

    ## change color and linewidth of the caps
    for cap in bplot['caps']:
        cap.set(color='#222222', linewidth=1)

    # Set the borders to a given color...
    ax.tick_params(color="#666666", labelcolor="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")

    plt.ylabel(yaxis)
    plt.tick_params(color="#222222", labelcolor="#222222")
    ax.set_xticklabels(permuted_names,rotation=0, fontsize=12)
    plt.subplots_adjust(left=0.1, bottom=0.2, right=1, top=0.95, wspace=0.0, hspace=0.0)
    fig.savefig(get_plot_filename(yaxis.split(" ")[0].lower(), experiment_name))



def plot_mse_comparison(name, attributions, names, gt_idx=None):

    plot_data = _compare_attributions(attributions, metric='mse')
    plot_data = plot_data[gt_idx, :, :]

    permutation = np.argsort(-np.mean(plot_data, 1))

    plot_data = plot_data[permutation]

    permuted_names = [names[i] for i in permutation]
    gt_permuted_index = permuted_names.index(names[gt_idx])

    plot_data = plot_data.tolist()
    plot_data.pop(gt_permuted_index)
    permuted_names.pop(gt_permuted_index)

    _plot_boxplot(plot_data, permuted_names, 'RMSE', name)


def plot_kendall_comparison(name, attributions, names, gt_idx=None):

    plot_data = _compare_attributions(attributions, metric='corr')
    plot_data = plot_data[gt_idx, :, :]

    permutation = np.argsort(np.mean(plot_data, 1))

    plot_data = plot_data[permutation]

    permuted_names = [names[i] for i in permutation]
    gt_permuted_index = permuted_names.index(names[gt_idx])

    plot_data = plot_data.tolist()
    plot_data.pop(gt_permuted_index)
    permuted_names.pop(gt_permuted_index)

    _plot_boxplot(plot_data, permuted_names, "Spearman's rank correlation", name)
