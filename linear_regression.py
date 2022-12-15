import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class LinReg:
    def __init__(self, train_x, train_y, test_x, test_y, horizon=5, make_plot=True):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.horizon = horizon
        self.make_plot = make_plot
        self.model = LinearRegression()
        self.predictions_y = None

    def fit_regression(self):
        """ Do the regression """
        x = self.train_x
        nsamples, nx, ny = x.shape
        x_reshaped = x.reshape((nsamples, nx * ny))
        self.model.fit(x_reshaped, self.train_y)

    def predict(self, time):
        """ Do the prediction to test the estimator
        :param time: how long in the future to predict
        """
        y_pred = np.empty(shape=(time, 12))
        x_in = self.test_x[0]

        for i in range(time):
            prediction = self.model.predict(x_in.ravel().reshape(1, -1))
            y_pred[i] = prediction[0]
            x_in = np.vstack([x_in[1:], np.array(prediction)])

        self.predictions_y = y_pred


def plot_trajectories(res, actual):
    """ Plot the trajectories: predicted and actual of all 3 bodies
    :param res: predicted results
    :param actual: actual data
    :return: show plot
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    ax.plot(res[:, 0], res[:, 1], color='C0', alpha=1)
    ax.plot(res[:, 2], res[:, 3], color='C1', alpha=1)
    ax.plot(res[:, 4], res[:, 5], color='C2', alpha=1)

    ax.plot(actual[:, 0], actual[:, 1], color='C0', alpha=0.2)
    ax.plot(actual[:, 2], actual[:, 3], color='C1', alpha=0.2)
    ax.plot(actual[:, 4], actual[:, 5], color='C2', alpha=0.2)

    ax.scatter(res[-1, 0], res[-1, 1], marker="o", s=100,
               label="Point Mass A", color='C0', alpha=1)
    ax.scatter(res[-1, 2], res[-1, 3], marker="o", s=100,
               label="Point Mass B", color='C1', alpha=1)
    ax.scatter(res[-1][4], res[-1][5], marker="o", s=100,
               label="Point Mass C", color='C2', alpha=1)

    ax.scatter(actual[-1, 0], actual[-1, 1], marker="o",
               s=100, color='C0', alpha=0.2)
    ax.scatter(actual[-1, 2], actual[-1, 3], marker="o",
               s=100, color='C1', alpha=0.2)
    ax.scatter(actual[-1, 4], actual[-1, 5], marker="o",
               s=100, color='C2', alpha=0.2)

    ax.legend()
    fig.show()


def read_data(length, datapath='TBP_dataset.csv'):
    """ Get the generated dataset.
    :param length: number of observations from dataset
    :param datapath: absolute path to dataset
    :return: dataset
    """
    df = pd.read_csv(datapath)
    df_short = df[["r1_x", "r1_y", "r2_x", "r2_y", "r3_x", "r3_y", "v1_x",
                   "v1_y", "v2_x", "v2_y", "v3_x", "v3_y"]][:length]

    return df_short


def splitxy(data, timestep=100):
    """ Split the given data into x and y
    :param data: the dataset
    :param timestep: how many time steps to be used for predicting (horizon)
    :return: the dataset split into x and y
    """
    data_x = []
    data_y = []

    for i in range(len(data) - timestep - 1):
        x = data.loc[i:i + timestep]
        y = data.loc[i + timestep + 1]
        data_x.append(x.reset_index(drop=True))
        data_y.append(y.reset_index(drop=True))

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def main():
    # getting train and test data
    df_length = 100000
    dataset = read_data(df_length)
    train = dataset[:50000].reset_index(drop=True)
    test = dataset[50000:].reset_index(drop=True)

    # splitting data based on horizon
    horizon = 5
    time_prediction = 1000
    train_x, train_y = splitxy(train, horizon)
    test_x, test_y = splitxy(test, horizon)

    # fitting
    lr = LinReg(train_x, train_y, test_x, test_y, horizon, True)
    lr.fit_regression()
    lr.predict(time_prediction)

    # results
    print(mean_squared_error(lr.test_y[:time_prediction], lr.predictions_y))
    plot_trajectories(np.insert(lr.predictions_y, 0, lr.test_x[0], axis=0),
                      np.insert(lr.test_y[:time_prediction], 0, lr.test_x[0], axis=0))


if __name__ == "__main__":
    main()
