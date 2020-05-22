from datetime import datetime, timedelta
from pathlib import Path

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


class Model:
    def __init__(self, model_info, path_info, route_info):
        self.model_info = model_info
        self.path_info = path_info
        self.route_info = route_info

    def get_model(self, size):
        channel = self.model_info.channel
        n_step = self.model_info.n_step
        activation = self.model_info.activation
        optimizer = self.model_info.optimizer
        loss = self.model_info.loss

        with K.tf_ops.device('/GPU:0'):
            seq = Sequential()
            seq.add(ConvLSTM2D(filters=channel, kernel_size=(3, 3), data_format='channels_first',
                               input_shape=(n_step, channel, size, size),
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=channel, kernel_size=(3, 3), data_format='channels_first',
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=channel, kernel_size=(3, 3), data_format='channels_first',
                               padding='same', return_sequences=True))
            seq.add(BatchNormalization())

            seq.add(ConvLSTM2D(filters=channel, kernel_size=(3, 3), data_format='channels_first',
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
        seq.save(self.path_info.get_model_path)
