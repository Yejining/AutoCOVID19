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

from autocovid.tools.loader import Dataset
from autocovid.tools.image import ImageGenerator


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
        feature_level = 'feature_level_%d' % self.args.feature_depth
        path = join(self.args.root, 'results', self.args.model_type,
                    feature_level, self.args.feature_type, self.args.name)
        Path(path).mkdir(parents=True, exist_ok=True)

        args_dict = dict()
        for arg in vars(args):
            args_dict.update({arg: str(getattr(args, arg))})

        with open(join(path, 'args.json'), 'w') as f:
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

        feature_level = 'feature_level_%d' % self.args.feature_depth
        path = join(self.args.root, 'results', self.args.model_type, feature_level,
                    self.args.feature_type, self.args.name)
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save(str(join(path, 'model.h5')))

        return model

    def get_accuracy(self, true, pred):
        print('get tools accuracy')
        mape_list = []
        rmse_list = []

        for sample in range(pred.shape[0]): # sample, 1, 1, size, size
            sample_pred = pred[sample, 0, 0, ] # size, size
            sample_true = true[sample, 0, 0, ]

            sample_pred[sample_pred >= 0] += 1
            sample_true[sample_true >= 0] += 1

            sample_diff = np.abs(sample_true - sample_pred)
            mape = np.mean(sample_diff / sample_true)
            mape_list.append(mape)

            mse = mean_squared_error(sample_true, sample_pred)
            rmse = sqrt(mse)
            rmse_list.append(rmse)

        mape = np.mean(mape_list)
        rmse = np.mean(rmse_list)

        print('mape: %.3lf, rmse: %.3lf' % (mape, rmse))

        accuracy = {'mape': mape, 'rmse': rmse}
        return accuracy

    def save_prediction_in_h5(self, true, prediction):
        sample_num = prediction.shape[0]
        size = self.args.size

        prediction = prediction.reshape((sample_num, size, size))
        true = np.asarray(true)
        true = true.reshape((sample_num, size, size))
        start_date = datetime.strptime(self.args.test_start, '%Y-%m-%d')
        date = [start_date.year, start_date.month, start_date.day]

        feature_level = 'feature_level_%d' % self.args.feature_depth
        saving_path = join(self.args.root, 'results', self.args.model_type, feature_level,
                           self.args.feature_type, self.args.name)
        Path(saving_path).mkdir(parents=True, exist_ok=True)

        h5_path = join(saving_path, 'prediction.h5')
        print('save predicted image and true image to %s' % h5_path)
        with h5py.File(h5_path, 'w') as f:
            prediction = f.create_dataset('prediction', data=prediction)
            true = f.create_dataset('true', data=true)
            start_date = f.create_dataset('start_date', data=date)

    def save_accuracy(self, accuracy):
        feature_level = 'feature_level_%d' % self.args.feature_depth
        path = join(self.args.root, 'results', self.args.model_type, feature_level,
                    self.args.feature_type, self.args.name)
        Path(path).mkdir(parents=True, exist_ok=True)

        path = join(path, 'accuracy.json')
        print('save tools accuracy to %s' % path)
        with open(path, 'w') as f:
            json.dump(accuracy, f)


def generate_and_save_dataset(args, feature_type, feature_list):
    print('generate and save dataset of %s' % feature_type)
    setattr(args, 'feature_type', feature_type)

    for feature_name in feature_list:
        setattr(args, 'name', feature_name)
        setattr(args, 'feature_name', feature_name)
        print('generate dataset of %s' % feature_name)

        loader = Dataset(args, feature_type, feature_name)
        dataset = loader.load_dataset()

        image_generator = ImageGenerator(args)
        image_generator.generate_and_save_dataset(dataset)


def load_dataset_and_predict(args, feature_type, feature_list):
    setattr(args, 'feature_type', feature_type)
    for feature_name in feature_list:
        setattr(args, 'name', feature_name)
        setattr(args, 'feature_name', feature_name)
        print('convolution process on %s' % args.name)

        loader = Dataset(args, feature_type, feature_name)
        channel = loader.get_channel_length()
        print('channel: %d' % channel)
        setattr(args, 'channel', channel)

        print('load dataset')
        image_generator = ImageGenerator(args)
        train_set, val_set, test_set = image_generator.load_image_dataset()
        dataset = {'train': train_set, 'val': val_set, 'test': test_set}

        trainer = COVIDConvLSTM(args)
        trainer.train_model_and_save_prediction(dataset)


def save_all_accuracy(args, feature_types):
    result_df = pd.DataFrame()

    for feature in feature_types:
        feature_level = 'feature_level_%d' % args.feature_depth
        dir_path = join(args.root, 'results', args.model_type, feature_level, feature)
        sub_dir_list = [sub_dir for sub_dir in os.listdir(dir_path)]
        for sub_dir in sub_dir_list:
            accuracy_path = join(dir_path, sub_dir, 'accuracy.json')
            accuracy = pd.read_json(accuracy_path, orient='index')
            mape = accuracy.loc['mape'].iloc[0]
            rmse = accuracy.loc['rmse'].iloc[0]
            accuracy_dict = {'name': '%s_%s' % (feature, sub_dir), 'mape': mape, 'rmse': rmse}
            result_df = result_df.append(accuracy_dict, ignore_index=True)

    result_df = result_df.set_index(['name'])

    feature_level = 'feature_level_%d' % args.feature_depth
    result_path = join(args.root, 'results', args.model_type, feature_level, 'accuracy.csv')
    result_df.to_csv(result_path, encoding='utf-8-sig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    # ========== Path ========== #
    args.root = Path(os.getcwd())

    # ========== Test ========== #
    args.model_type = 'convlstm'
    args.is_logged = False

    # ========== Dataset ========== #
    args.train_start = '2020-01-22'
    args.train_end = '2020-05-28'
    args.val_start = '2020-05-29'
    args.val_end = '2020-08-12'
    args.test_start = '2020-08-13'
    args.test_end = '2020-09-02'
    args.accumulating_days = 3

    # ========== Feature ========== #
    # args.feature_depth = 1
    args.general_depth = 1

    # ========== Image ========== #
    args.size = 256
    args.kde_kernel_size = 17
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

    # ========== Code ========== #

