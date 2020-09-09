from datetime import datetime, timedelta

from keras.utils.io_utils import HDF5Matrix
import numpy as np
import h5py
from pathlib import Path


class Dataset:
    def __init__(self,route_info, model_info):
        self.route_info = route_info
        self.model_info = model_info

    def save_dataset(self, path, X_set, y_set, first_day):
        print("save_dataset")
        day = [first_day.year, first_day.month, first_day.day]
        print(type(X_set), type(y_set), type(day))

        with h5py.File(path, 'w') as f:
            set_X = f.create_dataset('X_set', data=X_set)
            set_y = f.create_dataset('y_set', data=y_set)
            set_day = f.create_dataset('start_day', data=day)

    def load_data_from_file(self, path):
        X_train, y_train, X_eval, y_eval, X_test, y_test, start_day1, start_day2, start_day3 = self.load_data(path)

        return X_train, y_train, X_eval, y_eval, X_test, y_test, start_day1, start_day2, start_day3

    def load_data(self, path):
        print("data load path: ", path)
        X_train = HDF5Matrix(path, 'X_set', start=0, end=53)
        y_train = HDF5Matrix(path, 'y_set', start=0, end=53)
        X_eval = HDF5Matrix(path, 'X_set', start=53, end=53+20)
        y_eval = HDF5Matrix(path, 'y_set', start=53, end=53+20)
        X_test = HDF5Matrix(path, 'X_set', start=53+20)
        y_test = HDF5Matrix(path, 'y_set', start=53+20)

        start_day = np.array(HDF5Matrix(path, 'start_day'))
        start_day = "%d-%.2d-%.2d" % (start_day[0], start_day[1], start_day[2])
        start_day1 = datetime.strptime(start_day, "%Y-%m-%d")
        start_day2 = start_day1 + timedelta(days=53)
        start_day3 = start_day1 + timedelta(days=53+20)

        return X_train, y_train, X_eval, y_eval, X_test, y_test, start_day1, start_day2, start_day3

    def figure_load_data(self, path):
        X_test = HDF5Matrix(path, 'X_set')
        y_test = HDF5Matrix(path, 'y_set')
        start_day = np.array(HDF5Matrix(path, 'start_day'))
        start_day = "%d-%.2d-%.2d" % (start_day[0], start_day[1], start_day[2])
        start_day = datetime.strptime(start_day, "%Y-%m-%d")

        return X_test, y_test, start_day