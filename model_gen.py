import numpy as np
import pandas as pd
import datetime
from enum import Enum

from one_step_prediction_MLP import MLP
from one_step_prediction_RNN import LSTM
from data_process import Data

import random
import tensorflow as tf

seed = 69
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


class ModelType(Enum):
    RNN = 1
    MLP = 2


def get_models(**params):
    max_size = 10_000
    data_processor = Data(params.pop('train_test_split'), params['lookback'], params.pop('file_path'),
                          max_size=max_size, expand=params['expand'])

    time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    f = open(f'logs/log_{time}.txt', 'a')
    f.write(str(params) + '\n' + f'size dataset: {max_size}' + '\n' + f'seed: {seed}')
    f.close()

    print("Data has been prepared!")

    RNN_model = LSTM(
        data_processor.train_X,
        data_processor.train_Y,

        data_processor.test_X,
        data_processor.test_Y,

        params.pop('lstm_units'),

        loss=params['loss'],
        optimizer=params['optimizer'],
        regularizer = params['regularizer'],
        validation_split=params['validation_split'],
        epochs=params['epochs'],
        batch_size=params['batch_size_rnn'],
        patience=params['patience'],
        lookback=params['lookback'],

        plot=params['plot']
    )

    RNN_model.make_new_predictions(len(data_processor.test_X), with_memory=True)
    RNN_model.make_new_predictions(len(data_processor.test_X), with_memory=False)

    MLP_model = MLP(
        data_processor.train_X,
        data_processor.train_Y,

        data_processor.test_X,
        data_processor.test_Y,

        layer_sizes=params.pop('layer_sizes'),

        loss=params['loss'],
        optimizer=params['optimizer'],
        validation_split=params['validation_split'],
        epochs=params['epochs'],
        batch_size=params['batch_size_mlp'],
        patience=params['patience'],
        lookback=params['lookback'],

        plot=params['plot']
    )

    MLP_model.make_new_predictions(len(data_processor.test_X))

    return RNN_model, MLP_model


if __name__ == "__main__":
    # Data is generated and split in the get_models function internally
    # Thus, only hyper-parameters need to be determined
    # train_test_split% of data will be used for the model, 10 percent will be kept for testing purposes
    params = {
        'train_test_split': 0.9,
        'layer_sizes': [7, 8, 11, 11, 8],
        'lstm_units': 9,
        'file_path': 'training_data.pkl',
        'loss': 'mean_squared_error',
        'optimizer': 'adam',
        'regularizer': 'l2',
        'validation_split': 0.2,
        'epochs': 10000,
        'batch_size_rnn': 256,
        'batch_size_mlp': 1,
        'patience': 5,
        'lookback': 5,
        'plot': True,
        'expand': True
    }

    rnn, mlp = get_models(**params)
    rnn.model.summary()
    mlp.model.summary()
    rnn.model.save(f'models/rnn_{str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}')
    mlp.model.save(f'models/mlp_{str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))}')
