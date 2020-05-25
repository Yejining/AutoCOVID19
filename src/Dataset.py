from datetime import datetime, timedelta

from keras.utils.io_utils import HDF5Matrix
import numpy as np
import h5py


class Dataset:
    def __init__(self, X_set, y_set, route_info, model_info):
        self.X_set = X_set
        self.y_set = y_set
        self.route_info = route_info
        self.model_info = model_info

    def save_dataset(self, path, first_day):
        print("save_dataset")
        day = [first_day.year, first_day.month, first_day.day]
        print(type(self.X_set), type(self.y_set), type(day))

        with h5py.File(path, 'w') as f:
            set_X = f.create_dataset('X_set', data=self.X_set)
            set_y = f.create_dataset('y_set', data=self.y_set)
            set_day = f.create_dataset('start_day', data=day)

    def load_data_from_file(self, path):
        sample_num = self.route_info.duration - (2 * self.model_info.n_step)
        split_num = self.model_info.set_split_num(sample_num)
        X_train, y_train, X_test, y_test, start_day1, start_day2 = self.load_data(path, split_num)

        return X_train, y_train, X_test, y_test, start_day1, start_day2

    def load_data(self, path, n_train):
        X_train = HDF5Matrix(path, 'X_set', start=0, end=n_train)
        y_train = HDF5Matrix(path, 'y_set', start=0, end=n_train)
        X_test = HDF5Matrix(path, 'X_set', start=n_train)
        y_test = HDF5Matrix(path, 'y_set', start=n_train)

        start_day = np.array(HDF5Matrix(path, 'start_day'))
        start_day = "%d-%.2d-%.2d" % (start_day[0], start_day[1], start_day[2])
        start_day1 = datetime.strptime(start_day, "%Y-%m-%d")
        start_day2 = start_day1 + timedelta(days=n_train)

        return X_train, y_train, X_test, y_test, start_day1, start_day2

    def figure_load_data(self, path):
        X_test = HDF5Matrix(path, 'X_set')
        y_test = HDF5Matrix(path, 'y_set')
        start_day = np.array(HDF5Matrix(path, 'start_day'))
        start_day = "%d-%.2d-%.2d" % (start_day[0], start_day[1], start_day[2])
        start_day = datetime.strptime(start_day, "%Y-%m-%d")

        return X_test, y_test, start_day