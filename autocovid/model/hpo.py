import os
import json
import logging
import argparse

from os.path import join
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns

from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D

from image import ImageGenerator
from loader import Dataset

logging.basicConfig(level=logging.DEBUG)


class COVIDWorker(Worker):
    def __init__(self,  *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

    def set_dataset(self, args, dataset):
        print('set dataset')
        self.args = args
        self.dataset = dataset

    def compute(self, config, budget, **kwargs):
        print("compute")

        train_set = self.dataset['train']
        val_set = self.dataset['val']
        test_set = self.dataset['test']

        num_filters = config['num_convlstm_layers']

        filter1 = 1 if num_filters == 1 else config['num_filters_1']
        filter2 = 1 if num_filters <= 2 else config['num_filters_2']
        filter3 = 1 if num_filters <= 3 else config['num_filters_3']

        with tf.device('/GPU:0'):
            model = Sequential()

            model.add(ConvLSTM2D(filters=filter1,
                                 kernel_size=(config['convlstm_k_size_1'], config['convlstm_k_size_1']),
                                 data_format='channels_first',
                                 input_shape=(self.args.n_step, self.args.channel, self.args.size, self.args.size),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 1:
                model.add(ConvLSTM2D(filters=filter2,
                                     kernel_size=(config['convlstm_k_size_2'], config['convlstm_k_size_2']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 2:
                model.add(ConvLSTM2D(filters=filter3,
                                     kernel_size=(config['convlstm_k_size_3'], config['convlstm_k_size_3']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 3:
                model.add(ConvLSTM2D(filters=1,
                                     kernel_size=(config['convlstm_k_size_4'], config['convlstm_k_size_4']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            model.add(Conv3D(filters=1,
                             kernel_size=(config['conv3d_k_size'], config['conv3d_k_size'], config['conv3d_k_size']),
                             activation=self.args.activation, padding='same', data_format='channels_first'))

            model.compile(loss=self.args.loss,
                          optimizer=keras.optimizers.RMSprop(lr=self.args.learning_rate),
                          metrics=['accuracy'])

            model.summary()

            model.fit(train_set['x_set'], train_set['y_set'],
                      batch_size=config['batch_size'],
                      epochs=int(budget),
                      shuffle='batch',
                      validation_data=(val_set['x_set'], val_set['y_set']))

            train_score = model.evaluate(train_set['x_set'], train_set['y_set'], verbose=0)
            validation_score = model.evaluate(val_set['x_set'], val_set['y_set'], verbose=0)
            test_score = model.evaluate(test_set['x_set'], test_set['y_set'], verbose=0)
            count_params = model.count_params()

        result = {
            'loss': 1 - validation_score[1],
            'info': {
                'test accuracy': test_score[1],
                'train accuracy': train_score[1],
                'validation accuracy': validation_score[1],
                'number of parameters': count_params,
                'epochs': str(budget)
            }
        }

        return result

    def get_configspace(self):
        print('get configuration space')
        configuration_space = CS.ConfigurationSpace()

        # training configurations
        batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64])
        configuration_space.add_hyperparameter(batch_size)

        # hyper-parameters which can effect channel selection
        num_convlstm_layers = CSH.UniformIntegerHyperparameter('num_convlstm_layers',
                                                               lower=1, upper=4, default_value=4, log=True)

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1',
                                                         lower=1, upper=self.args.channel, default_value=1, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2',
                                                         lower=1, upper=self.args.channel, default_value=1, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3',
                                                         lower=1, upper=self.args.channel, default_value=1, log=True)

        convlstm_k_size_1 = CSH.CategoricalHyperparameter('convlstm_k_size_1', [1, 3, 5, 7, 9])
        convlstm_k_size_2 = CSH.CategoricalHyperparameter('convlstm_k_size_2', [1, 3, 5, 7, 9])
        convlstm_k_size_3 = CSH.CategoricalHyperparameter('convlstm_k_size_3', [1, 3, 5, 7, 9])
        convlstm_k_size_4 = CSH.CategoricalHyperparameter('convlstm_k_size_4', [1, 3, 5, 7, 9])
        conv3d_k_size = CSH.CategoricalHyperparameter('conv3d_k_size', [1, 3, 5, 7, 9])

        configuration_space.add_hyperparameters([num_convlstm_layers,
                                                 num_filters_1, num_filters_2, num_filters_3,
                                                 convlstm_k_size_1, convlstm_k_size_2,
                                                 convlstm_k_size_3, convlstm_k_size_4])

        configuration_space.add_hyperparameter(conv3d_k_size)

        # adding conditions
        childs = [num_filters_1, num_filters_2, num_filters_3,
                  convlstm_k_size_2, convlstm_k_size_3, convlstm_k_size_4]
        numbers = [1, 2, 3, 1, 2, 3]

        for i in range(len(childs)):
            condition = CS.GreaterThanCondition(childs[i], num_convlstm_layers, numbers[i])
            configuration_space.add_condition(condition)

        print("configuration space ended")
        print(configuration_space)

        return configuration_space


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    # ========== Path ========== #
    args.root = Path(os.getcwd())

    # ========== Test ========== #
    args.model_type = 'automl'
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
    args.feature_depth = 1

    # ========== Image ========== #
    args.size = 256
    args.kde_kernel_size = 17
    args.weight = 1

    # ========== Model ========== #
    args.nameserver = '127.0.0.1'
    args.min_budget = 30
    args.max_budget = 300
    args.n_iterations = 10
    args.n_step = 3

    # ========== Code ========== #
    setattr(args, 'feature_type', 'all')
    setattr(args, 'feature_name', 'all')
    loader = Dataset(args)

    feature_types = ['all']
    for feature in feature_types:
        setattr(args, 'feature_type', feature)
        setattr(args, 'feature_hame', feature)
        setattr(args, 'name', feature)

        channel = 1 if feature == 'one' else loader.get_channel_length()
        setattr(args, 'channel', channel)

        image_generator = ImageGenerator(args)
        train_set, val_set, test_set = image_generator.load_image_dataset()
        dataset = {'train': train_set, 'val': val_set, 'test': test_set}

        name_server = hpns.NameServer(run_id=args.name, host=args.nameserver, port=None)
        name_server.start()

        worker = COVIDWorker(sleep_interval=0, nameserver=args.nameserver, run_id=args.name)
        worker.set_dataset(args, dataset)
        worker.run(background=True)

        bohb = BOHB(configspace=worker.get_configspace(),
                    run_id=args.name, nameserver=args.nameserver,
                    min_budget=args.min_budget, max_budget=args.max_budget)

        result = bohb.run(n_iterations=args.n_iterations)

        bohb.shutdown(shutdown_workers=True)
        name_server.shutdown()

        id2config = result.get_id2config_mapping()
        incumbent = result.get_incumbent_id()

        print('Best found configurations:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where excuted.' % len(result.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in result.get_all_runs()]) / args.max_budget))

        args_path = join(args.root, 'results', args.model_type, args.feature_type, args.name)
        Path(args_path).mkdir(parents=True, exist_ok=True)

        args_dict = dict()
        for arg in vars(args):
            args_dict.update({arg: str(getattr(args, arg))})
        args_dict['best_hyper_parameter'] = id2config[incumbent]['config']

        with open(join(args_path, 'args.json'), 'w') as file:
            json.dump(args_dict, file)
