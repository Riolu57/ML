# https://towardsdatascience.com/one-step-predictions-with-lstm-forecasting-hotel-revenues-c9ef0d3ef2df
# packages
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import plot_utils

class Data:
    def __init__(self, train_perc, lookback):
        self.train_perc = train_perc
        self.lookback = lookback

        self.data = self.read_data()
        self.train_data, self.test_data = self.train_test_split()
        self.train_X, self.train_Y = self.timeseries_conversion(self.train_data)
        self.test_X, self.test_Y = self.timeseries_conversion(self.test_data)


    def read_data(self, path='TBP_dataset.csv'):

        df = pd.read_csv(path)
        data = df[['r1_x', 'r1_y', 'v1_x', 'v1_y', 'r2_x', 'r2_y', 'v2_x', 'v2_y', 'r3_x', 'r3_y', 'v3_x',
                   'v3_y']].values.tolist()
        return data[0:1000]


    def train_test_split(self):
        train_size = int(len(self.data) * self.train_perc)

        train_data, test_data = self.data[0:train_size], self.data[train_size:(len(self.data) + 1)]
        return train_data, test_data


    def timeseries_conversion(self, arr):
        dataX, dataY = [], []
        for i in range(len(arr)-self.lookback-1):
            a = arr[i:(i+self.lookback)]
            dataX.append(a)
            dataY.append(arr[i + self.lookback])
        return np.array(dataX), np.array(dataY)


class LSTM:
    def __init__(self, train_X, train_Y, test_X, test_Y, loss='mean_squared_error',
                optimizer='adam', validation_split=0.2, epochs=10000, batch_size=1,
                patience=5, lookback=5, plot=True):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.loss = loss
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lookback = lookback

        self.plot = plot
        self.model = self.train_model()

    def train_model(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)
        model = tf.keras.Sequential()
        model.add(layers.LSTM(2))
        model.add(layers.Dense(12))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        history = model.fit(self.train_X, self.train_Y,
                            validation_split=self.validation_split,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            callbacks=[callback],
                            verbose=2)
        print("MSE on test data: ", model.evaluate(self.test_X, self.test_Y))

        if self.plot:
            plot_utils.plot_loss(history)

        return model

    def make_new_predictions(self, horizon):
        input = self.test_X[0]
        preds = []
        for _ in range(horizon):
            pred = self.model.predict(np.reshape(input, (1, self.lookback, 12)))
            preds.append(pred)
            input = np.concatenate((input[1:len(input)], pred))

        if self.plot:
            plot_utils.plot_predictions(preds, self.test_Y[0:horizon], horizon)

    def evaluate_predictions_on_test(self, horizon):
        horizon = 100
        preds = []
        for i in range(horizon):
            pred = self.model.predict(np.reshape(self.test_X[i], (1, self.lookback, 12)))
            preds.append(pred)

        if self.plot:
            plot_utils.plot_predictions(preds, self.test_Y[0:horizon], horizon)



if __name__ == "__main__":
    data = Data(train_perc=0.8, lookback=5)
    lstm = LSTM(data.train_X, data.train_Y, data.test_X, data.test_Y)
    lstm.make_new_predictions(100)
    lstm.evaluate_predictions_on_test(100)


