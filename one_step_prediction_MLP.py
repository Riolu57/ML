# packages
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import plot_utils


class Data:
    def timeseries_conversion(self, arr):
        import copy
        dataX, dataY = [], []
        for i in range(len(arr) - self.lookback - 1):
            a = copy.deepcopy(arr[i])
            for lb in range(1, self.lookback):
                a += arr[i + lb]
            dataX.append(a)
            dataY.append(arr[i + self.lookback])
        return np.array(dataX), np.array(dataY)


class MLP:
    def __init__(self, train_X, train_Y, test_X, test_Y, layer_sizes, loss='mean_squared_error',
                 optimizer='adam', validation_split=0.2, epochs=10000, batch_size=1,
                 patience=5, lookback=5, plot=True):
        self.loss = loss
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lookback = lookback

        self.train_X = self.adjust_data(train_X)
        self.train_Y = train_Y
        self.test_X = self.adjust_data(test_X)
        self.test_Y = test_Y

        self.plot = plot
        self.model = self.train_model(layer_sizes)

    def train_model(self, layer_sizes):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)
        model = tf.keras.Sequential()

        for size in layer_sizes:
            model.add(layers.Dense(size))

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
            pred = self.model.predict(np.reshape(input, (1, 12*self.lookback)))
            preds.append(pred)
            input = np.concatenate((input[12:], pred[0]))

        if self.plot:
            plot_utils.plot_predictions(preds, self.test_Y[0:horizon], horizon)

    def evaluate_predictions_on_test(self, horizon):
        preds = []
        for i in range(horizon):
            pred = self.model.predict(np.reshape(self.test_X[i], (1, 12*self.lookback)))
            preds.append(pred)

        if self.plot:
            plot_utils.plot_predictions(preds, self.test_Y[0:horizon], horizon)

    def adjust_data(self, data):
        first = np.reshape(data[0], (1, self.lookback*12))

        for i in range(1, data.shape[0]):
            first = np.concatenate((first, np.reshape(data[i], (1, self.lookback*12))))

        return first


if __name__ == "__main__":
    look = 5
    print("Prepare data")
    data = Data(train_perc=0.8, lookback=look)
    print("Train MLP")
    mlp = MLP(data.train_X, data.train_Y, data.test_X, data.test_Y, lookback=look)
    print("New predictions")
    mlp.make_new_predictions(100)
    print("Predictions test")
    mlp.evaluate_predictions_on_test(100)
