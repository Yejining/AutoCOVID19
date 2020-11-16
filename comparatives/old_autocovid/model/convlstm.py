import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from os import listdir
from PIL import Image
from math import sqrt
from pathlib import Path
from os.path import join, isdir
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D

from comparatives.old_autocovid.tools.loader import Dataset
from comparatives.old_autocovid.tools.image import ImageGenerator


class COVIDConvLSTM:
    def __init__(self, args):
        self.args = args

    def train_model_and_save_prediction(self, dataset):
        test_set = dataset['test']
        trained_model = self.train(dataset)

        print('predict')
        prediction = trained_model.predict(test_set['x_set'])
        self.save_prediction_in_h5(test_set['y_set'], prediction)

        print('get accuracy')
        accuracy = self.get_accuracy(test_set['y_set'], prediction)
        self.save_accuracy(accuracy)

        print('saving arguments')
        args_path = self._get_results_path()

        args_dict = dict()
        for arg in vars(self.args):
            args_dict.update({arg: str(getattr(self.args, arg))})

        with open(join(args_path, 'args.json'), 'w') as f:
            json.dump(args_dict, f)

    def get_model(self):
        size = self.args.size
        channel = self.args.channel
        n_step = self.args.n_step
        activation = self.args.activation
        optimizer = self.args.optimizer
        loss = self.args.loss
        convlstm_kernel = self.args.convlstm_kernel
        conv_kernel = self.args.conv_kernel

        with tf.device('/GPU:0'):
            model = Sequential()
            model.add(ConvLSTM2D(filters=1, kernel_size=(convlstm_kernel, convlstm_kernel), data_format='channels_first',
                                 input_shape=(n_step, channel, size, size),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            model.add(ConvLSTM2D(filters=1, kernel_size=(convlstm_kernel, convlstm_kernel), data_format='channels_first',
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            model.add(ConvLSTM2D(filters=1, kernel_size=(convlstm_kernel, convlstm_kernel), data_format='channels_first',
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            model.add(ConvLSTM2D(filters=1, kernel_size=(convlstm_kernel, convlstm_kernel), data_format='channels_first',
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            model.add(Conv3D(filters=1, kernel_size=(conv_kernel, conv_kernel, conv_kernel),
                             activation=activation,
                             padding='same', data_format='channels_first'))

            model.compile(optimizer=optimizer, loss=loss)

        return model

    def train(self, dataset):
        train_set = dataset['train']
        val_set = dataset['val']

        x_train = train_set['x_set']
        y_train = train_set['y_set']

        x_val = val_set['x_set']
        y_val = val_set['y_set']

        epochs = self.args.epochs
        batch_size = self.args.batch_size

        model = self.get_model()
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=epochs, batch_size=batch_size, shuffle='batch')

        model.summary()
        model.save(str(join(self._get_results_path(), 'model.h5')))

        return model

    def get_accuracy(self, true, pred):
        print('get tools accuracy')
        mape_list = []
        rmse_list = []

        for sample in range(pred.shape[0]): # sample, 1, 1, size, size
            sample_pred = pred[sample, 0, 0, ] # size, size
            sample_true = true[sample, 0, 0, ]

            mape_elements = []
            for row in range(sample_pred.shape[0]):
                for col in range(sample_pred.shape[1]):
                    if sample_true[row][col] == 0: continue
                    sample_diff = sample_true[row][col] - sample_pred[row][col]
                    sample_divided = np.abs(sample_diff / sample_true[row][col])
                    mape_elements.append(sample_divided)

            if len(mape_elements) != 0:
                mape = np.mean(mape_elements)
                mape_list.append(mape)

            mse = mean_squared_error(sample_true, sample_pred)
            rmse = sqrt(mse)
            rmse_list.append(rmse)

        mape = np.mean(mape_list)
        rmse = np.mean(rmse_list)

        prev_mape_list = []
        for sample in range(pred.shape[0]):
            sample_pred = pred[sample, 0, 0, ]
            sample_true = true[sample, 0, 0, ]

            sample_pred[sample_pred >= 0] += 1
            sample_true[sample_true >= 0] += 1

            temp_mape = np.mean(np.abs((sample_pred - sample_true) / sample_true))
            prev_mape_list.append(temp_mape)

        print('mape: %.3lf, rmse: %.3lf' % (mape, rmse))

        accuracy = {'mape': mape, 'prev_mape': np.mean(prev_mape_list), 'rmse': rmse}
        return accuracy

    def save_prediction_in_h5(self, true, prediction):
        sample_num = prediction.shape[0]
        size = self.args.size

        prediction = prediction.reshape((sample_num, size, size))
        true = np.asarray(true)
        true = true.reshape((sample_num, size, size))
        start_date = datetime.strptime(self.args.test_start, '%Y-%m-%d')
        date = [start_date.year, start_date.month, start_date.day]

        h5_path = join(self._get_results_path(), 'prediction.h5')
        print('save predicted image and true image to %s' % h5_path)
        with h5py.File(h5_path, 'w') as f:
            prediction = f.create_dataset('prediction', data=prediction)
            true = f.create_dataset('true', data=true)
            start_date = f.create_dataset('start_date', data=date)

    def save_accuracy(self, accuracy):
        path = join(self._get_results_path(), 'accuracy.json')
        print('save tools accuracy to %s' % path)
        with open(path, 'w') as f:
            json.dump(accuracy, f)

    def _get_results_path(self):
        feature_level = 'feature_level_%d' % self.args.feature_depth
        model_path = join(self.args.root, 'comparatives', 'old_autocovid', 'results',
                          feature_level, self.args.feature_type, self.args.name)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    # ========== Path ========== #
    args.root = Path(os.getcwd())

    # ========== Test ========== #
    args.model_type = 'convlstm'

    # ========== Dataset ========== #
    args.train_start = '2020-01-22'
    args.train_end = '2020-03-18'
    args.val_start = '2020-03-19'
    args.val_end = '2020-04-07'
    args.test_start = '2020-04-08'
    args.test_end = '2020-04-27'
    args.accumulating_days = 3

    # ========== Feature ========== #
    args.feature_depth = 1
    args.general_depth = 1

    # ========== Image ========== #
    args.size = 256
    args.kde_kernel_size = 9
    args.weight = 1

    # ========== Model ========== #
    args.n_step = 3
    args.epochs = 300
    args.batch_size = 16
    args.activation = 'relu'
    args.optimizer = 'rmsprop'
    args.loss = 'mean_squared_error'
    args.convlstm_kernel = 3
    args.conv_kernel = 3

    # ========== Code ========== #9
    setattr(args, 'feature_type', 'all')
    setattr(args, 'feature_name', 'all')
    loader = Dataset(args)
    age_features = loader.age_features
    gender_features = loader.gender_features
    reason_features = loader.reason_features
    type_features = loader.type_features

    feature_type_list = ['age', 'gender', 'reason', 'type']
    names = [age_features, gender_features, reason_features, type_features]

    for i, feature_type in enumerate(feature_type_list):
        setattr(args, 'feature_type', feature_type)
        for name in names[i]:
            setattr(args, 'feature_name', name)
            setattr(args, 'name', name)

            # generate_dataset
            loader = Dataset(args)
            route_dict = loader.load_dataset()

            image_generator = ImageGenerator(args)
            dataset_list = image_generator.generate_and_save_dataset(route_dict)
            dataset = {'train': dataset_list[0], 'val': dataset_list[1], 'test': dataset_list[2]}

            # learning
            trainer = COVIDConvLSTM(args)
            trainer.train_model_and_save_prediction(dataset)

            # reset
            delattr(args, 'age_features')
            delattr(args, 'gender_features')
            delattr(args, 'type_features')
            delattr(args, 'reason_features')
            delattr(args, 'x_max')
            delattr(args, 'y_max')
            delattr(args, 'channel')
