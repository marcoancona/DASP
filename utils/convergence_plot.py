import numpy as np
import matplotlib.pyplot as plt
from .utils import color_for_label, get_plot_filename, _compare_attributions


def convergence_comparison_rmse(experiment_name, series, series_names, x_ticks, gt, max_tick=None):
    return convergence_comparison(experiment_name, series, series_names, x_ticks, gt, 'mse', max_tick)


def convergence_comparison_corr(experiment_name, series, series_names, x_ticks, gt, max_tick=None):
    return convergence_comparison(experiment_name, series, series_names, x_ticks, gt, 'corr', max_tick)


def convergence_comparison(experiment_name, series, series_names, x_ticks, gt, metric, max_tick=None):
    fig = plt.figure(figsize=(6, 3))

    if max_tick is None:
        max_tick = np.max([np.max(x) for x in x_ticks])

    legend_handles = []
    for attributions, name, ticks in zip(series, series_names, x_ticks):
        _attributions = attributions.copy()
        _attributions.append(gt)
        comparison = _compare_attributions(_attributions, metric)[-1, :-1, :]

        y = np.mean(comparison, 1)
        e = np.std(comparison, 1)

        y2 = np.array([y[-1], y[-1]])
        e2 = np.array([e[-1], e[-1]])
        t2 = np.array([ticks[-1], max_tick])

        plt.fill_between(ticks, y + e / 2, y - e / 2, alpha=.2, color=color_for_label(name))
        h, = plt.plot(ticks, y, color=color_for_label(name), label=name)
        legend_handles.append(h)
        if max_tick > ticks[-1]:
            plt.fill_between(t2, y2 + e2 / 2, y2 - e2 / 2, alpha=.2, color=color_for_label(name))
            if 'Integrated' not in name:
                plt.plot(t2, y2, '--', color=color_for_label(name))
            else:
                plt.plot(t2, y2, color=color_for_label(name))

    plt.xscale("log")
    plt.xlim(1, max_tick)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.5, 1))
    plt.box(False)
    plt.tick_params(color="#222222", labelcolor="#222222")
    plt.xlabel("Adjusted number of evaluations")
    if metric == "corr":
        plt.ylabel("Spearman's rank correlation")
    else:
        plt.ylabel("RMSE")
    plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    plt.show()
    fig.savefig(get_plot_filename('convergence_'+metric, experiment_name), bbox_inches = "tight")


