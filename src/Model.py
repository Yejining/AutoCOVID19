from datetime import datetime, timedelta
from pathlib import Path

from keras.layers import AveragePooling3D
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv3D, Conv1D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import numpy as np
from numpy import minimum


class Model:
    def __init__(self, model_info, path_info, route_info, feature_info):
        self.model_info = model_info
        self.path_info = path_info
        self.route_info = route_info
        self.feature_info = feature_info

    def get_model(self, size):
        channel = self.model_info.channel
        n_step = self.model_info.n_step
        activation = self.model_info.activation
        optimizer = self.model_info.optimizer
        loss = self.model_info.loss

        with K.tf_ops.device('/GPU:0'):
            seq = Sequential()
            seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), data_format='channels_first',
                               input_shape=(n_step, channel, size, size),
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), data_format='channels_first',
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), data_format='channels_first',
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), data_format='channels_first',
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                           activation=activation,
                           padding='same', data_format='channels_first'))

            seq.compile(optimizer=optimizer, loss=loss)

            seq.summary()

        return seq

    def train(self, X_train, y_train, size):
        epochs = self.model_info.epochs
        batch_size = self.model_info.batch_size

        seq = self.get_model(size)
        seq.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle='batch')
        seq.save(self.path_info.get_model_path())

    def get_accuracy(self, pred, y_test):
        diff1 = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]))
        diff2 = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]))
        diff3 = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]))
        diff4 = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]))
        rmse = np.zeros((pred.shape[0], pred.shape[2]))
        mape = np.zeros((pred.shape[0], pred.shape[2]))
        test_max_value = np.zeros((pred.shape[0], pred.shape[2]))
        pred_max_value = np.zeros((pred.shape[0], pred.shape[2]))

        for i in range(pred.shape[0]):  # days
            for j in range(pred.shape[2]):  # features
                pred_diff = pred[i][0][j]
                y_test_diff = y_test[i][0][j]

                diff1[i, j, :, :] = 255 - np.abs(y_test_diff - pred_diff)
                diff2[i, j, :, :] = 255 - (y_test_diff - pred_diff)
                diff3[i, j, :, :] = 255 - (pred_diff - y_test_diff)
                diff4[i, j, :, :] = minimum(pred_diff, y_test_diff)
                diff2[diff2 > 255] = 255
                diff3[diff3 > 255] = 255

                pred_diff[pred_diff >= 0] += 1
                y_test_diff[y_test_diff >= 0] += 1

                rmse[i][j] = sqrt(mean_squared_error(y_test_diff, pred_diff))
                test_max_value[i][j] = np.amax(y_test[i][0][j]) + 1
                pred_max_value[i][j] = np.amax(pred[i][0][j]) + 1
                mape[i][j] = np.mean(np.abs((y_test_diff - pred_diff) / y_test_diff))

        return diff1, diff2, diff3, diff4, rmse, mape, test_max_value, pred_max_value

