import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings
import pickle
import plot_utils
from data_process import Data

warnings.filterwarnings('ignore')


class LinReg:
    def __init__(self, train_x, train_y, test_x, test_y, alpha, lookback=5, make_plot=True):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.lookback = lookback
        self.make_plot = make_plot
        self.model = Ridge(alpha=alpha)
        self.predictions_y = None

    def train_model(self, filename):
        """ Do the regression """
        x = self.train_x
        # sklearn expects 2D arr for training dataset so reshape
        nsamples, nx, ny = x.shape
        x_reshaped = x.reshape((nsamples, nx * ny))
        self.model.fit(x_reshaped, self.train_y)
        pickle.dump(self.model, open(filename, 'wb'))

    def make_new_predictions(self, horizon):
        """ Do the prediction to test the estimator
        :param horizon: how long in the future to predict
        """
        y_pred = np.empty(shape=(horizon, 12))
        x_in = self.test_x[0]

        for i in range(horizon):
            # reshape to 2D arr
            prediction = self.model.predict(x_in.ravel().reshape(1, -1))
            y_pred[i] = prediction[0]
            # new input so forget previous first one and add new one at top
            x_in = np.vstack([x_in[1:], np.array(prediction)])

        self.predictions_y = y_pred
        if self.make_plot:
            plot_utils.plot_trajectories(self.predictions_y, self.test_y[:horizon], 
                                         horizon, save=True, regression=True)

    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

if __name__ == "__main__":
    lookback = 5
    time_prediction = 10000
    alpha = 0.6

    data = Data(train_perc=0.9, lookback=lookback)

    lr = LinReg(data.train_X, data.train_Y, data.test_X, data.test_Y, alpha, lookback, True)
    lr.train_model("regr_model.pkl")
    lr.make_new_predictions(time_prediction)
    print(f"MSE : {mean_squared_error(lr.test_y[:time_prediction], lr.predictions_y)}")
