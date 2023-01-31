import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_loss(history, save=False, name='ouput'):
    # list all data in history
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        plt.savefig(f'loss_figs/{name}_loss_{time}.png')
        plt.close()
    else:
        plt.show()

def plot_trajectories(res, actual, horizon, save=False, name='output', regression=False):
    """ Plot the trajectories: predicted and actual of all 3 bodies
    :param res: predicted results
    :param actual: actual data
    :param regression: if model is regression or not, since the res is already in correct shape for regr
    :return: show plot
    """
    if not regression:
        res = np.array(res).reshape((horizon, 12))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    ax.plot(res[:, 0], res[:, 1], color='C0', alpha=1)
    ax.plot(res[:, 4], res[:, 5], color='C1', alpha=1)
    ax.plot(res[:, 8], res[:, 9], color='C2', alpha=1)

    ax.plot(actual[:, 0], actual[:, 1], color='C0', alpha=0.2)
    ax.plot(actual[:, 4], actual[:, 5], color='C1', alpha=0.2)
    ax.plot(actual[:, 8], actual[:, 9], color='C2', alpha=0.2)

    ax.scatter(res[-1, 0], res[-1, 1], marker="o", s=100,
               label="Point Mass A", color='C0', alpha=1)
    ax.scatter(res[-1, 4], res[-1, 5], marker="o", s=100,
               label="Point Mass B", color='C1', alpha=1)
    ax.scatter(res[-1][8], res[-1][9], marker="o", s=100,
               label="Point Mass C", color='C2', alpha=1)

    ax.scatter(actual[-1, 0], actual[-1, 1], marker="o",
               s=100, color='C0', alpha=0.2)
    ax.scatter(actual[-1, 4], actual[-1, 5], marker="o",
               s=100, color='C1', alpha=0.2)
    ax.scatter(actual[-1, 8], actual[-1, 9], marker="o",
               s=100, color='C2', alpha=0.2)

    ax.legend()

    if save:
        time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        fig.savefig(f'pred_figs/{name}_pred_{time}.png')
        plt.close()
    else:
        fig.show()
