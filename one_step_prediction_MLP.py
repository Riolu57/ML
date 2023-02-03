# packages
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import plot_utils


class MLP:
    def __init__(self, train_X, train_Y, test_X, test_Y, layer_sizes, loss='mean_squared_error',
                 optimizer='adam', regularizer = 'l2', validation_split=0.2, epochs=10000, batch_size=1,
                 patience=5, lookback=5, plot=True):
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lookback = lookback

        self.old_shape = train_X.shape
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

        model.add(layers.Dense(12, kernel_regularizer=self.regularizer))
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
            plot_utils.plot_loss(history, True, 'MLP')

        return model

    def make_new_predictions(self, horizon):
        input = self.test_X[0].reshape(1, self.test_X[0].shape[0])
        preds = []
        for _ in range(horizon):
            pred = self.model.predict(input)
            preds.append(pred)
            input = np.concatenate((input[:, self.old_shape[2]:].reshape(1, input.shape[0] - self.old_shape[2]), pred),
                                   axis=1)

        if self.plot:
            plot_utils.plot_trajectories(preds, self.test_Y[0:horizon], horizon, True, 'MLP')

    def adjust_data(self, data):
        first = np.reshape(data[0], (1, self.lookback*data[0].shape[1]))

        for i in range(1, data.shape[0]):
            first = np.concatenate((first, np.reshape(data[i], (1, self.lookback*data[i].shape[1]))))

        return first


