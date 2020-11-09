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

        for sample in range(pred.shape[0]):
            sample_pred = pred[sample, 0, 0, ]
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
        if feature_name != 'group': continue
        setattr(args, 'name', feature_name)
        setattr(args, 'feature_name', feature_name)
        print('generate dataset of %s' % feature_name)

        loader = Dataset(args, feature_type, feature_name)
        dataset = loader.load_dataset()
        train_df = dataset['train']
        val_df = dataset['val']
        test_df = dataset['test']

        image_generator = ImageGenerator(args)

        train_set = image_generator.get_image_dataset(train_df)
        val_set = image_generator.get_image_dataset(val_df)
        test_set = image_generator.get_image_dataset(test_df)

        dataset_list = image_generator.normalize([train_set, val_set, test_set])
        train_set = dataset_list[0]
        val_set = dataset_list[1]
        test_set = dataset_list[2]

        image_generator.save_image_dataset(train_set, 'train')
        image_generator.save_image_dataset(val_set, 'val')
        image_generator.save_image_dataset(test_set, 'test')

        # image_generator.save_images(train_set, 'train')
        # image_generator.save_images(val_set, 'val')
        # image_generator.save_images(test_set, 'test')

        if args.is_logged:
            feature_level = 'feature_level_%d' % args.feature_depth
            path = join(args.root, 'dataset', feature_level, args.feature_type, args.name)
            Path(path).mkdir(parents=True, exist_ok=True)

            args_dict = dict()
            for arg in vars(args):
                args_dict.update({arg: str(getattr(args, arg))})

            with open(join(path, 'args.json'), 'w') as f:
                json.dump(args_dict, f)


def train_and_predict(args, feature_type, feature_list, city_depth=None, type_depth=None, reason_depth=None):
    print('train and predict dataset of %s' % feature_type)
    setattr(args, 'feature_type', feature_type)

    for feature_name in feature_list:
        setattr(args, 'name', feature_name)
        setattr(args, 'feature_name', feature_name)
        print('convolution process on %s' % feature_name)

        loader = Dataset(args, feature_type, feature_name, city_depth, type_depth, reason_depth)
        channel = loader.get_channel_length()
        print('channel: %d' % channel)
        setattr(args, 'channel', channel)

        print('load dataset')
        image_generator = ImageGenerator(args)

        train_set, val_set, test_set = image_generator.load_image_dataset()
        dataset = {'train': train_set, 'val': val_set}

        trainer = COVIDConvLSTM(args)
        print('train model')
        trained_model = trainer.train(dataset)

        print('predict')
        prediction = trained_model.predict(test_set['x_set'])
        trainer.save_prediction_in_h5(test_set['y_set'], prediction)

        print('get accuracy')
        accuracy = trainer.get_accuracy(test_set['y_set'], prediction)
        trainer.save_accuracy(accuracy)

        print('saving arguments')
        feature_level = 'feature_level_%d' % args.feature_depth
        path = join(args.root, 'results', args.model_type, feature_level, feature_type, args.name)
        Path(path).mkdir(parents=True, exist_ok=True)

        args_dict = dict()
        for arg in vars(args):
            args_dict.update({arg: str(getattr(args, arg))})

        with open(join(path, 'args.json'), 'w') as f:
            json.dump(args_dict, f)


def abc(args):
    print('convolution process on %s' % args.name)

    loader = Dataset(args)
    channel = loader.get_channel_length()
    print('channel: %d' % channel)
    setattr(args, 'channel', channel)

    print('load dataset')
    image_generator = ImageGenerator(args)

    train_set, val_set, test_set = image_generator.load_image_dataset()
    dataset = {'train': train_set, 'val': val_set}

    trainer = COVIDConvLSTM(args)
    print('train model')
    trained_model = trainer.train(dataset)

    print('predict')
    prediction = trained_model.predict(test_set['x_set'])
    trainer.save_prediction_in_h5(test_set['y_set'], prediction)

    print('get accuracy')
    accuracy = trainer.get_accuracy(test_set['y_set'], prediction)
    trainer.save_accuracy(accuracy)

    print('saving arguments')
    feature_level = 'feature_level_%d' % args.feature_depth
    path = join(args.root, 'results', args.model_type, feature_level, args.feature_type, args.name)
    Path(path).mkdir(parents=True, exist_ok=True)

    args_dict = dict()
    for arg in vars(args):
        args_dict.update({arg: str(getattr(args, arg))})

    with open(join(path, 'args.json'), 'w') as f:
        json.dump(args_dict, f)


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
    # generate dataset
    # feature_depth_list = [3, 4, 5, 6]
    # for depth in feature_depth_list:
    #     print(depth)
    #     setattr(args, 'type_depth', depth)
    #     setattr(args, 'feature_depth', depth)
    #
    #     loader = Dataset(args, 'all', 'all')
    #     type_features = loader.type_features
    #     print('type_features: %s' % type_features)
    #     generate_and_save_dataset(args, 'type', type_features)
    #     print()

    # feature_depth_list = [3, 4, 5, 6, 7]
    # for depth in feature_depth_list:
    #     print(depth)
    #     setattr(args, 'reason_depth', depth)
    #     setattr(args, 'feature_depth', depth)
    #
    #     loader = Dataset(args, 'all', 'all')
    #     reason_features = loader.reason_features
    #     print('reason_features: %s' % reason_features)
    #     generate_and_save_dataset(args, 'reason', reason_features)
    #     print()

    feature_depth_list = [3, 4, 5, 6]
    for depth in feature_depth_list:
        print(depth)
        setattr(args, 'type_depth', depth)
        setattr(args, 'feature_depth', depth)

        result_df = pd.DataFrame()
        dir_path = join(args.root, 'dataset', 'feature_level_%d' % depth, 'type')
        sub_dirs = [sub for sub in listdir(dir_path) if isdir(join(dir_path, sub))]
        print(sub_dirs)
        train_and_predict(args, 'type', sub_dirs, type_depth=depth)
        # for sub_dir in sub_dirs:
        #     accuracy_path = join(dir_path, sub_dir, 'accuracy.json')
        #     accuracy = pd.read_json(accuracy_path, orient='index')
        #     mape = accuracy.loc['mape'].iloc[0]
        #     rmse = accuracy.loc['rmse'].iloc[0]
        #     accuracy_dict = {'name': 'city_%s' % sub_dir, 'mape': mape, 'rmse': rmse}
        #     result_df = result_df.append(accuracy_dict, ignore_index=True)
        #
        # result_df = result_df.set_index(['name'])
        #
        # feature_level = 'feature_level_%d' % args.feature_depth
        # result_path = join(args.root, 'results', 'convlstm', feature_level, 'accuracy.csv')
        # result_df.to_csv(result_path, encoding='utf-8-sig')

    delattr(args, 'type_depth')

    feature_depth_list = [3, 4, 5, 6, 7]
    for depth in feature_depth_list:
        print(depth)
        setattr(args, 'reason_depth', depth)
        setattr(args, 'feature_depth', depth)

        result_df = pd.DataFrame()
        dir_path = join(args.root, 'dataset', 'feature_level_%d' % depth, 'reason')
        sub_dirs = [sub for sub in listdir(dir_path) if isdir(join(dir_path, sub))]
        print(sub_dirs)
        train_and_predict(args, 'reason', sub_dirs, reason_depth=depth)
