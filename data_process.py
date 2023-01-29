import numpy as np
import pandas as pd


class Data:
    def __init__(self, train_perc, lookback, file_path='TBP_dataset.csv', max_size=False):
        self.train_perc = train_perc
        self.lookback = lookback
        self.path = file_path
        self.data = self.read_data()

        self.train_data, self.test_data = self.train_test_split(max_num=max_size)
        self.train_X, self.train_Y = self.timeseries_conversion(self.train_data)
        self.test_X, self.test_Y = self.timeseries_conversion(self.test_data)

    def read_data(self):
        df = pd.read_csv(self.path)
        data = df[['r1_x', 'r1_y', 'v1_x', 'v1_y', 'r2_x', 'r2_y', 'v2_x', 'v2_y', 'r3_x', 'r3_y', 'v3_x',
                   'v3_y']].values.tolist()
        return data

    def train_test_split(self, max_num=False):
        if max_num:
            size = max_num
        else:
            size = len(self.data)
        train_size = int(size * self.train_perc)
        train_data, test_data = self.data[0:train_size], self.data[train_size:size]
        return train_data, test_data

    def timeseries_conversion(self, arr):
        dataX, dataY = [], []
        for i in range(len(arr)-self.lookback-1):
            a = arr[i:(i+self.lookback)]
            dataX.append(a)
            dataY.append(arr[i + self.lookback])
        return np.array(dataX), np.array(dataY)
