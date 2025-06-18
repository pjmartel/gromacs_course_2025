## Functions to plot xvg files with Matplotlib
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
  if window_size < 1:
    raise ValueError("Window size must be at least 1.")
  return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def parse_xvg(filename):
    legends = {}
    labels = {}
    data_lines = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('@'):
                parts = line.split()
                if parts[1] == 'title':
                    labels['title'] = ' '.join(parts[2:]).strip('"')
                elif parts[1] == 'xaxis' and parts[2] == 'label':
                    labels['xlabel'] = ' '.join(parts[3:]).strip('"')
                elif parts[1] == 'yaxis' and parts[2] == 'label':
                    labels['ylabel'] = ' '.join(parts[3:]).strip('"')
                elif parts[1].startswith('s') and parts[2] == 'legend':
                    index = int(parts[1][1:])
                    legends[index] = ' '.join(parts[3:]).strip('"')
            else:
                data_lines.append(line)

    # Parse data into columns
    data = [list(map(float, line.split())) for line in data_lines]
    data_by_columns = list(zip(*data))  # transpose rows to columns

    return data_by_columns, legends, labels

def plot_xvg(filename, show_moving_avg=False, window_size=10):

    data_columns, legends, labels = parse_xvg(filename)
    x = data_columns[0]
    num_datasets = len(data_columns) - 1

    plt.figure(figsize=(10, 6))

    for i in range(num_datasets):
        y = data_columns[i + 1]
        label = legends.get(i, f'Dataset {i}')
        plt.plot(x, y, label=label)

        # Apply moving average only if there's a single dataset and the user requested it
        if show_moving_avg and num_datasets == 1:
            y_avg = moving_average(y, window_size)
            x_avg = x[:len(y_avg)]  # match lengths
            plt.plot(x_avg, y_avg, label=f'{label} (Moving Avg, window={window_size})', linestyle='--')

    plt.xlabel(labels.get('xlabel', 'X-axis'))
    plt.ylabel(labels.get('ylabel', 'Y-axis'))
    plt.title(labels.get('title', ''))
    if num_datasets > 1 or legends:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# This version supports chaining multiple plots on the same axes, based on a
# shared ax variable (should replace the plot_xvg, but it is not completely tested)
def plot_xvg_multi(filename, show_moving_avg=False, window_size=10, ax=None, custom_legend=None):

    data_columns, legends, labels = parse_xvg(filename)
    x = data_columns[0]
    num_datasets = len(data_columns) - 1

    # Create new Axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(num_datasets):
        y = data_columns[i + 1]
        #label = legends.get(i, f'Dataset {i}')
        label = custom_legend if custom_legend and num_datasets == 1 else legends.get(i, f'Dataset {i}')
        ax.plot(x, y, label=label)

        if show_moving_avg and num_datasets == 1:
            y_avg = moving_average(y, window_size)
            x_avg = x[:len(y_avg)]
            ax.plot(x_avg, y_avg, linestyle='--',
                    label=f'{label} (Moving Avg, window={window_size})')

    ax.set_xlabel(labels.get('xlabel', 'X-axis'))
    ax.set_ylabel(labels.get('ylabel', 'Y-axis'))
    ax.set_title(labels.get('title', ''))

    if num_datasets > 1 or legends or custom_legend:
        ax.legend()
    ax.grid(True)

    return ax



