from os import listdir
from os.path import join
from pathlib import Path
from datetime import datetime, timedelta

from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from autocovid.tools.loader import Dataset
from autocovid.tools.vector import VectorGenerator


class COVIDLSTM:
    def __init__(self, args):
        self.args = args

    def train_model_and_save_results(self, dataset):
        print('train model and predict, than save model and predicted results')
        test_set = dataset['test']

        trained_model = self.train(dataset)
        true, prediction = self.predict(trained_model, test_set)
        self.save_prediction_in_h5(true, prediction, test_set['start_date'])
        accuracy = self.get_accuracy(test_set['y_set'], prediction)
        self.save_accuracy(accuracy)

    def train(self, dataset):
        train_set = dataset['train']
        val_set = dataset['val']

        root = self.args.root
        feature_depth = self.args.feature_depth
        feature_type = self.args.feature_type
        name = self.args.name

        vector_size = self.args.vector_size
        optimizer = self.args.optimizer
        frame_in = self.args.frame_in
        frame_out = self.args.frame_out
        channel = self.args.channel
        activation = self.args.activation
        loss = self.args.loss
        batch_size = self.args.batch_size
        epochs = self.args.epochs

        with tf.device('/GPU:0'):
            model = Sequential()

            model.add(LSTM(vector_size, input_shape=(frame_in, channel)))
            model.add(RepeatVector(frame_out))
            model.add(LSTM(vector_size, return_sequences=True))
            model.add(TimeDistributed(Dense(vector_size, activation=activation)))
            model.add(TimeDistributed(Dense(1)))
            model.compile(loss=loss, optimizer=optimizer)

            model.summary()

            model_path = join(root, 'results', 'lstm', 'feature_level_%d' % feature_depth,
                              feature_type, name)
            Path(join(model_path, 'models')).mkdir(parents=True, exist_ok=True)
            model_saving_path = join(model_path, 'model.h5')

            history = model.fit(train_set['x_set'], train_set['y_set'], batch_size=batch_size, epochs=epochs,
                                validation_data=(val_set['x_set'], val_set['y_set']), shuffle='batch')
            model.save(model_saving_path)

            return model

    def predict(self, model, test_set):
        prediction = model.predict(test_set['x_set'])
        true = test_set['y_set']
        y_max = test_set['y_set']

        inversed_true = true * y_max
        inversed_prediction = prediction * y_max

        return inversed_true, inversed_prediction

    def save_prediction_in_h5(self, true, prediction, start_date):
        date = [start_date.year, start_date.month, start_date.day]

        saving_path = join(self.args.root, 'results', self.args.model_type,
                           'feature_level_%d' % self.args.feature_depth,
                           self.args.feature_type, self.args.name)
        Path(saving_path).mkdir(parents=True, exist_ok=True)

        h5_path = join(saving_path, 'prediction.h5')
        print('save predcted vector and true vector to %s' % h5_path)
        with h5py.File(h5_path, 'w') as f:
            prediction = f.create_dataset('prediction', data=prediction)
            true = f.create_dataset('true', data=true)
            start_date = f.create_dataset('start_date', data=date)

    def get_accuracy(self, true, prediction):
        zero_indices = np.where(true == 0)
        non_zero_indices = np.where(true != 0)
        y_data_adjusted = np.delete(true, zero_indices)
        y_pred_adjusted = np.delete(prediction, zero_indices)

        value_list = []
        for sample in range(prediction.shape[0]):
            sample_pred = prediction[sample, 0, 0]
            sample_true = true[sample, 0, 0]

            sample_diff = np.abs(sample_true - sample_pred)
            divided_value = sample_diff / sample_true
            value_list.append(divided_value)
        mape = np.mean(np.asarray(value_list))

        rmse = np.sqrt(mean_squared_error(y_data_adjusted, y_pred_adjusted))
        max_min = np.max(y_data_adjusted) - np.min(y_data_adjusted)
        nrmse = rmse / max_min

        pred_zero_indices = np.where(prediction == 0)
        pred_non_zero_indices = np.where(prediction != 0)
        true_positive = np.intersect1d(zero_indices, pred_zero_indices).size
        true_negative = np.intersect1d(non_zero_indices, pred_non_zero_indices).size
        false_positive = np.intersect1d(non_zero_indices, pred_zero_indices).size
        false_negative = np.intersect1d(zero_indices, pred_non_zero_indices).size

        print('in test datsaet, zeros: %d, non_zeros: %d' % (zero_indices[0].size, non_zero_indices[0].size))
        print('in prediction, zeros: %d, non_zeros: %d' % (pred_zero_indices[0].size, pred_non_zero_indices[0].size))

        print('true_positive: %d, true_negative: %d, false_positive: %d, false_negative: %d' %
              (true_positive, true_negative, false_positive, false_negative))

        all = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / all if all != 0 else -1
        true_predics = true_positive + false_positive
        precision = true_positive / true_predics if true_predics != 0 else -1
        true_cases = true_positive + false_negative
        recall = true_positive / true_cases if true_cases != 0 else -1
        if precision != -1 and recall != -1:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = -1

        print('nrmse: %lf, accuracy: %lf, f1_score: %lf' % (nrmse, accuracy, f1_score))

        result = dict()
        result.update({'mape': mape, 'rmse': rmse, 'nrmse': nrmse, 'accuracy': accuracy, 'f1_score': f1_score})
        print(result)
        return result

    def save_accuracy(self, accuracy):
        path = join(self.args.root, 'results', self.args.model_type,
                    'feature_level_%d' % self.args.feature_depth,
                    self.args.feature_type, self.args.name)
        Path(path).mkdir(parents=True, exist_ok=True)

        path = join(path, 'accuracy.json')
        print('save lstm model accuracy to %s' % path)
        with open(path, 'w') as f:
            json.dump(accuracy, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    # ========== Path ========== #
    args.root = Path(os.getcwd())

    # ========== Test ========== #
    args.model_type = 'lstm'
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

    # ========== Model ========== #
    args.vector_size = 256
    args.frame_in = 3
    args.frame_out = 1
    args.epochs = 200
    args.activation = 'relu'
    args.optimizer = 'rmsprop'
    args.loss = 'mse'
    args.batch_size = 64

    # ========== Feature Depth 1 ========== # all
    args.feature_depth = 2

    # loader = Dataset(args, 'all', 'all')
    # city_features = loader.city_features
    # type_features = loader.type_features
    # reason_features = loader.reason_features

    feature_types = ['city', 'type', 'reason']

    # ========== Feature Depth 1 ========== # generate dataset
    # feature_names = []
    # for feature_type in feature_types:
    #     setattr(args, 'feature_type', feature_type)
    #
    #     if feature_type == 'all': feature_names = ['all']
    #     if feature_type == 'city': feature_names = city_features
    #     if feature_type == 'type': feature_names = type_features
    #     if feature_type == 'reason': feature_names = reason_features
    #
    #     for feature_name in feature_names:
    #         setattr(args, 'feature_name', feature_name)
    #         setattr(args, 'name', feature_name)
    #
    #         loader = Dataset(args, feature_type, feature_name)
    #         dataset = loader.load_dataset()
    #
    #         vector_generator = VectorGenerator(args)
    #         vector_generator.generate_and_save_dataset(dataset)

    # ========== Feature Depth 1 ========== # train than save results
    feature_names = []
    for feature_type in feature_types:
        setattr(args, 'feature_type', feature_type)

        dir_path = join(args.root, 'dataset', args.model_type,
                        'feature_level_%d' % args.feature_depth, feature_type)
        result_path = join(args.root, 'results', args.model_type,
                           'feature_level_%d' % args.feature_depth, feature_type)

        feature_names = [sub_dir for sub_dir in listdir(dir_path)]
        for feature_name in feature_names:
            setattr(args, 'feature_name', feature_name)
            setattr(args, 'name', feature_name)

            loader = Dataset(args, feature_type, feature_name)
            channel_len = loader.get_channel_length()
            setattr(args, 'channel', channel_len)

            vector_generator = VectorGenerator(args)
            train_set, val_set, test_set = vector_generator.load_vector_dataset()
            dataset = {'train': train_set, 'val': val_set, 'test': test_set}

            trainer = COVIDLSTM(args)
            trainer.train_model_and_save_results(dataset)

            saving_path = join(result_path, feature_name)
            Path(saving_path).mkdir(parents=True, exist_ok=True)

            args_dict = dict()
            for arg in vars(args):
                args_dict.update({arg: str(getattr(args, arg))})

            with open(join(saving_path, 'args.json'), 'w') as f:
                json.dump(args_dict, f)
