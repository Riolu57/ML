# https://towardsdatascience.com/one-step-predictions-with-lstm-forecasting-hotel-revenues-c9ef0d3ef2df
# packages
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import plot_utils
from data_process import Data

class LSTM:
    def __init__(self, train_X, train_Y, test_X, test_Y, units, loss='mean_squared_error',
                optimizer='adam', regularizer='l2', validation_split=0.2, epochs=10000, batch_size=1,
                patience=5, lookback=5, plot=True):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lookback = lookback

        self.plot = plot
        self.model = self.train_model(units)

    def train_model(self, units):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)
        model = tf.keras.Sequential()
        model.add(layers.LSTM(units, kernel_regularizer=self.regularizer))
        model.add(layers.Dense(12))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        history = model.fit(self.train_X, self.train_Y,
                            validation_split=self.validation_split,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            callbacks=[callback],
                            shuffle=False,
                            verbose=2)
        print("MSE on test data: ", model.evaluate(self.test_X, self.test_Y))

        if self.plot:
            plot_utils.plot_loss(history, True, 'RNN')

        return model

    def make_new_predictions(self, horizon, with_memory=True):
        input = self.test_X[0]
        preds = []

        if with_memory:
            for _ in range(horizon):
                pred = self.model.predict(np.reshape(input, (1, self.lookback, self.train_X[0].shape[1])))
                preds.append(pred)
                input = np.concatenate((input[1:len(input)], pred))

        else:
            base = np.zeros((self.lookback - 1, self.train_X.shape[2]))
            inp = np.concatenate(
                    (base, input[len(input) - 1:len(input)].reshape((1,self.train_X.shape[2])))).reshape(1,
                                                                                                       self.lookback,
                                                                                                       self.train_X.shape[2])

            for _ in range(horizon):
                pred = self.model.predict(inp)
                preds.append(pred)
                inp = np.concatenate((base, pred.reshape((1, self.train_X.shape[2])))).reshape(1, self.lookback, self.train_X.shape[2])

        if self.plot:
            plot_utils.plot_trajectories(preds, self.test_Y[0:horizon], horizon, True, 'RNN')



if __name__ == "__main__":
    data = Data(train_perc=0.8, lookback=5, max_size=10_000)
    lstm = LSTM(data.train_X, data.train_Y, data.test_X, data.test_Y)
    lstm.make_new_predictions(100)


