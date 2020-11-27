import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
from brokenaxes import brokenaxes

import json
from os import read, walk
import numpy as np
from scipy.stats import ttest_ind
from utils import smooth, create_dir_for_file


def get_plot_components(data_matrix):
    data_mean = np.mean(data_matrix, axis=0)
    data_var = np.var(data_matrix, axis=0)
    upper = data_mean + data_var
    lower = data_mean - data_var

    return data_mean, upper, lower


def read_topo_sim_file(fpath: str) -> list:
    topo_sim_list = []
    data_file = open(fpath, 'r')
    for line in data_file.readlines():
        result = json.loads(line.strip())
        topo_sim_list.append(float(result['topsim']))
    return topo_sim_list


def read_topo_sim_dir(dpath: str) -> list:
    file_name_list = []
    for (_, _, filenames) in walk(dpath):
        file_name_list.extend(filenames)
        break

    data_matrix = []
    for file_name in file_name_list:
        data_list = read_topo_sim_file(dpath+'/'+file_name)
        data_matrix.append(data_list)

    return data_matrix


def plot_both_games_curves(
        num_epochs=5000, 
        window_size=200,
        log_path='./log/',
    ):
    recon_matrix = np.asarray(read_topo_sim_dir(log_path+'recon_topsim/'))[:, :num_epochs]
    refer_matrix = np.asarray(read_topo_sim_dir(log_path+'refer_topsim/'))[:, :num_epochs]

    recon_mean, recon_upper, recon_lower = get_plot_components(recon_matrix)
    refer_mean, refer_upper, refer_lower = get_plot_components(refer_matrix)

    print('topo-sim:', ttest_ind(recon_mean[-29:], refer_mean[-29:]))

    recon_mean, recon_upper, recon_lower = tuple(
        [smooth(x, window_size)[:num_epochs] for x in [recon_mean, recon_upper, recon_lower]])
    refer_mean, refer_upper, refer_lower = tuple(
        [smooth(x, window_size)[:num_epochs] for x in [refer_mean, refer_upper, refer_lower]])

    # Start plotting the two lines with variance
    x_axis = np.arange(recon_mean.size) + 1

    plt.plot(x_axis, recon_mean, label='reconstruction language', linewidth=0.5)
    plt.fill_between(x_axis, recon_upper, recon_lower, color='blue', alpha=0.2)

    plt.plot(x_axis, refer_mean, label='referential language', linewidth=0.5)
    plt.fill_between(x_axis, refer_upper, refer_lower,
                     color='orange', alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Topological Similarity')
    plt.legend()
    plt.grid()
    plt.ylim([0.28, 0.36])

    fig_file = './result/compare_topo_sim.pdf'
    create_dir_for_file(fig_file)
    plt.savefig(fig_file, format='pdf', bbox_inches='tight')


def read_task_transfer_file(fpath: str, **kwargs) -> list:
    train_list = []
    test_list = []
    with open(fpath, 'r') as data_file:
        for line in data_file.readlines():
            train_data, test_data = line.strip().split(',')
            train_list.append(float(train_data))
            test_list.append(float(test_data))
    return train_list, test_list


def read_task_train_file(fpath: str, data_key='loss') -> list:
    train_list = []
    test_list = []
    with open(fpath, 'r') as data_file:
        for line in data_file.readlines():
            data = json.loads(line.strip())
            if data['mode'] == 'train':
                train_list.append(data[data_key])
            elif data['mode'] == 'test':
                test_list.append(data[data_key])
            else:
                raise ValueError("Unrecognised mode.")

    return train_list, test_list


def read_dir(dpath: str, read_file_function=read_task_transfer_file, data_key='loss') -> list:
    file_name_list = []
    for (_, _, filenames) in walk(dpath):
        file_name_list.extend(filenames)
        break

    train_matrix = []
    test_matrix = []
    for file_name in file_name_list:
        train_list, test_list = read_file_function(
            dpath+'/'+file_name, data_key=data_key)
        if len(train_list) > 5000:
            train_list = train_list[::2][:5000]
        if len(test_list) > 5000:
            test_list = test_list[::2][:5000]
        train_matrix.append(train_list)
        test_matrix.append(test_list)

    return train_matrix, test_matrix


def get_language_test_results(log_path='./log/', direction='recon_to_refer'):
    # Read the train and test performance in *first* training
    train_log_dir = direction.split('_')[0] + '_train'
    data_key = 'loss' if direction.split('_')[0] == 'recon' else 'acc'
    train_matrix, test_matrix = read_dir(log_path + train_log_dir, read_task_train_file, data_key)

    original_task_train_performace = np.asarray(train_matrix)
    original_task_test_performance = np.asarray(test_matrix)
    # then read the train and test performance in second training
    _, test_matrix = read_dir(log_path + direction, read_file_function=read_task_transfer_file)
    transfer_task_test_performance = np.asarray(test_matrix)

    return original_task_train_performace, original_task_test_performance, transfer_task_test_performance


def plot_task_transfer_curves(num_epochs=5000, window_size=200, log_path='./log/'):

    # the first refer/recon refers to the language, the second refer/recon refers to the task
    # e.g. "recon_on_refer" means
    # the generalisation performance of recon-language on referential task
    recon_train, recon_on_recon, recon_on_refer = get_language_test_results(log_path, 'recon_to_refer')
    refer_train, refer_on_refer, refer_on_recon = get_language_test_results(log_path, 'refer_to_recon')

    # 1. Process the data on reconstruction game
    recon_on_recon_mean, recon_on_recon_up, recon_on_recon_low = get_plot_components(recon_on_recon)
    refer_on_recon_mean, refer_on_recon_up, refer_on_recon_low = get_plot_components(refer_on_recon)
    x_axis = np.arange(num_epochs) + 1

    # significance test between generalisation performance of both languages on reconstruction game
    print(ttest_ind(recon_on_recon_mean[num_epochs-29:num_epochs], refer_on_recon_mean[num_epochs-29:num_epochs]))

    # plot the generalisation performance on reconstruction task
    plt.clf()

    recon_train_mean = 1 - np.mean(recon_train[:, 1000:num_epochs])
    plt.plot(x_axis,
             [recon_train_mean] * len(x_axis),
             linestyle='--',
             color='black',
             label='Converged training performance')
    plt.plot(x_axis,
             1. - smooth(recon_on_recon_mean, window_size)[:num_epochs],
             label='Reconstruction Language', linewidth=0.5, color='blue')
    plt.fill_between(x_axis, 
                     1. - smooth(recon_on_recon_up, window_size)[:num_epochs], 
                     1. - smooth(recon_on_recon_low, window_size)[:num_epochs], 
                     color='blue', alpha=0.2)

    plt.plot(x_axis, 1. - smooth(refer_on_recon_mean)[:num_epochs], 
             label='Referential Language', linewidth=0.5, color='orange')
    plt.fill_between(x_axis, 
                     1. - smooth(refer_on_recon_up)[:num_epochs], 
                     1. - smooth(refer_on_recon_low)[:num_epochs], 
                     color='orange', alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('1 - MSE')
    plt.legend(loc='upper right')
    plt.grid()
    plt.ylim([0.87, 0.93])

    _fig_file = './result/generalisation_on_recon.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')

    # 2. Process the data on referential game
    recon_on_refer_mean, recon_on_refer_up, recon_on_refer_low = get_plot_components(recon_on_refer)
    refer_on_refer_mean, refer_on_refer_up, refer_on_refer_low = get_plot_components(refer_on_refer)
    x_axis = np.arange(num_epochs) + 1

    # significance test between generalisation performance of both languages on referential game
    print(ttest_ind(recon_on_refer_mean[num_epochs-29:num_epochs], refer_on_refer_mean[num_epochs-29:num_epochs]))

    # plot the generalisation performance on referential task
    plt.clf()

    refer_train_mean = np.mean(refer_train[:, 1000:num_epochs])
    bax=brokenaxes(ylims=((0.89,0.93),(0.98,0.99)), hspace=0.2)

    bax.plot(x_axis,
             [refer_train_mean] * len(x_axis),
             linestyle='--',
             color='black',
             label='Converged training performance')
    bax.plot(x_axis, 
             smooth(recon_on_refer_mean, window_size)[:num_epochs], 
             label='Reconstruction Language', linewidth=0.5, color='blue')
    bax.fill_between(x_axis, 
                     smooth(recon_on_refer_up, window_size)[:num_epochs], 
                     smooth(recon_on_refer_low, window_size)[:num_epochs], 
                     color='blue', alpha=0.2)

    bax.plot(x_axis, 
             smooth(refer_on_refer_mean, window_size)[:num_epochs], 
             label='Referential Language', linewidth=0.5, color='orange')
    bax.fill_between(x_axis, 
                     smooth(refer_on_refer_up, window_size)[:num_epochs], 
                     smooth(refer_on_refer_low, window_size)[:num_epochs], 
                     color='orange', alpha=0.2)

    bax.set_xlabel('Epochs')
    bax.set_ylabel('Accuracy\n')
    bax.legend(loc=1)
    bax.grid()
    # plt.ylim([0.89, 0.99])

    _fig_file = './result/generalisation_on_refer.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs_fig1', type=int, default=10000, help='number of epochs shown in Figure 1.')
    parser.add_argument('--num_epochs_fig2', type=int, default=5000, help='number of epochs shown in Figure 2.')
    parser.add_argument('--smooth_window_size', type=int, default=100, help='window size for smoothing the curves.')
    args = parser.parse_args()

    plot_both_games_curves(args.num_epochs_fig1, args.smooth_window_size, './log/')
    plot_task_transfer_curves(args.num_epochs_fig2, args.smooth_window_size, './log/')
