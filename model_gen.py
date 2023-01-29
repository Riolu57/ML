import numpy as np
import pandas as pd

from enum import Enum

from one_step_prediction_MLP import MLP
from one_step_prediction_RNN import LSTM
from data_process import Data


class ModelType(Enum):
    RNN = 1
    MLP = 2


def get_models(**params):
    data_processor = Data(params.pop('train_test_split'), params['lookback'], params.pop('file_path'), max_size=10_000)
    print("Data has been prepared!")
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
        batch_size=params['batch_size'],
        patience=params['patience'],
        lookback=params['lookback'],

        plot=params['plot']
    )

    RNN_model = LSTM(
        data_processor.train_X,
        data_processor.train_Y,

        data_processor.test_X,
        data_processor.test_Y,

        params.pop('lstm_units'),

        loss=params['loss'],
        optimizer=params['optimizer'],
        validation_split=params['validation_split'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        patience=params['patience'],
        lookback=params['lookback'],

        plot=params['plot']
    )

    return MLP_model, RNN_model


if __name__ == "__main__":
    # Data is generated and split in the get_models function internally
    # Thus, only hyper-parameters need to be determined
    # train_test_split% of data will be used for the model, 10 percent will be kept for testing purposes
    params = {
        'train_test_split': 0.9,
        'layer_sizes': [10, 11, 9, 10, 9],
        'lstm_units': 9,
        'file_path': 'TBP_dataset.csv',
        'loss': 'mean_squared_error',
        'optimizer': 'adam',
        'validation_split': 0.2,
        'epochs': 10000,
        'batch_size': 1,
        'patience': 5,
        'lookback': 5,
        'plot': True,
    }

    mlp, rnn = get_models(**params)
    mlp.model.summary()
    rnn.model.summary()
