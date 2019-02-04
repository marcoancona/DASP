import numpy as np
import matplotlib.pyplot as plt


CORRUPTION_MAX_RATIO = 1
REMOVE_VALUE = 0
RANDOM_TESTS = 3

SAVE_IMAGES = 0
path = None


def plot_max_variation(deltas, names):
    plot_data = []

    for idx, map_name in enumerate(deltas):
        if map_name is not 'Random':
            #print (map_name)
            delta = deltas[map_name]
            #print (delta.shape)

            # perturbation_steps x samples
            original = delta[0]
            D = delta - original
            delta_max = np.max(np.abs(D), 0)
            #print (delta_max)
            #print (original)
            plot_data.append(delta_max)

    plot_data = np.array(plot_data)
    plot_data -= np.min(plot_data, 0)
    permutation = np.argsort(np.mean(plot_data, 1))
    plot_data = plot_data[permutation]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.boxplot(plot_data.tolist())

    ax.set_xticklabels([names[i] for i in permutation],rotation=45, fontsize=12)


