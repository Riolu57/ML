import numpy as np
from enum import Enum

import pandas as pd


class Methods(Enum):
    EULER = 1
    VERLET = 2


class DataGenerator:
    def __init__(self, pos, vel, masses, step_size=0.001, G=1, num_steps=10000, method=Methods.EULER, record=False):
        self.pos = np.asarray(pos)
        self.vel = np.asarray(vel)
        self.masses = np.asarray(masses)
        self.step_size = step_size
        self.G = G
        self.num_steps = num_steps
        self.method = method
        self.record = record

        if record:
            self.all_pos = np.zeros((self.num_steps, *self.pos.shape))
            self.all_vel = np.zeros((self.num_steps, *self.vel.shape))

    def comp_acc(self):
        res = np.zeros((3, 2))
        for idx, cur_pos in enumerate(self.pos):
            delta_x = cur_pos[0] - self.pos[:, 0]
            delta_y = cur_pos[1] - self.pos[:, 1]

            root = np.sqrt(delta_x**2 + delta_y**2)**3
            bodies = self.masses*self.G

            ax = (-delta_x * bodies) / root
            ay = (-delta_y * bodies) / root

            ax[np.isnan(ax)] = 0
            ay[np.isnan(ay)] = 0

            res[idx, 0] = sum(ax)
            res[idx, 1] = sum(ay)

        return res

    def step_euler(self):
        self.pos, self.vel = self.pos + self.step_size*self.vel, self.vel + self.step_size*self.comp_acc()

    def step_verlet(self):
        first_acc = self.comp_acc()
        self.pos = self.pos + self.step_size*self.vel + 0.5*(first_acc * self.step_size**2)
        second_acc = self.comp_acc()
        self.vel = self.vel + 0.5*(first_acc + second_acc)*self.step_size

    def __iter__(self):
        if self.method == Methods.EULER:
            call = self.step_euler
        elif self.method == Methods.VERLET:
            call = self.step_verlet
        else:
            raise ValueError("The method must either be from Euler or Verlet")

        for idx in range(self.num_steps):
            call()
            if self.record:
                self.all_pos[idx, :] = self.pos
                self.all_vel[idx, :] = self.vel
            yield self.pos, self.vel

class Data:
    def __init__(self, train_perc, lookback, file_path='TBP_dataset.csv'):
        self.train_perc = train_perc
        self.lookback = lookback
        self.path = file_path
        self.data = self.read_data()

        self.train_data, self.test_data = self.train_test_split()
        self.train_X, self.train_Y = self.timeseries_conversion(self.train_data)
        self.test_X, self.test_Y = self.timeseries_conversion(self.test_data)

    def read_data(self):
        df = pd.read_csv(self.path)
        data = df[['r1_x', 'r1_y', 'v1_x', 'v1_y', 'r2_x', 'r2_y', 'v2_x', 'v2_y', 'r3_x', 'r3_y', 'v3_x',
                   'v3_y']].values.tolist()
        return data

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



if __name__ == "__main__":
    gen = DataGenerator(
        pos=np.asarray([[0, 1], [0, -1], [0, 0]]),
        vel=np.asarray([[0.3471, 0.5327], [0.3471, 0.5327], [-2 * 0.3471, -2 * 0.5327]]),
        masses=np.asarray([1, 1, 1]),
        step_size=0.001,
        G=1,
        num_steps=1000,
        method=Methods.EULER,
        record=True
    )

    for _ in gen:
        pass

    pass

