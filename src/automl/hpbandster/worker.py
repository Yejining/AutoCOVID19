import json
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

import logging
logging.basicConfig(level=logging.DEBUG)


class COVIDWorker2(Worker):
    def __init__(self,  *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

    def init(self, dataset, information):
        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_validation = dataset[2]
        self.y_validation = dataset[3]
        self.x_test = dataset[4]
        self.y_test = dataset[5]

        self.n_step = information[0]
        self.channel = information[1]
        self.size = information[2]
        self.path = information[3]

        print(self.x_train.shape, self.y_train.shape, self.x_validation.shape, self.y_validation.shape)

    def compute(self, config, budget, **kwargs):
        print("compute")
        n_step = self.n_step
        channel = self.channel
        size = self.size

        num_filters = config['num_convlstm_layers']
        filter1 = 1
        filter2 = 1
        filter3 = 1

        if num_filters == 1:
            filter1 = 1
        elif num_filters == 2:
            filter1 = config['num_filters_1']
            filter2 = 1
        elif num_filters == 3:
            filter1 = config['num_filters_1']
            filter2 = config['num_filters_2']
            filter3 = 1
        else:
            filter1 = config['num_filters_1']
            filter2 = config['num_filters_2']
            filter3 = config['num_filters_3']

        with K.tf_ops.device('/GPU:0'):
            model = Sequential()

            print("num_convlstm_layers:", num_filters)
            print("filters:", filter1)
            print("kernel size:", config['kernel_size_1'])
            print("input shape:", n_step, channel, size)

            model.add(ConvLSTM2D(filters=filter1,
                                 kernel_size=(config['kernel_size_1'], config['kernel_size_1']),
                                 data_format='channels_first',
                                 input_shape=(n_step, channel, size, size),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 1:
                model.add(ConvLSTM2D(filters=filter2,
                                     kernel_size=(config['kernel_size_2'], config['kernel_size_2']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 2:
                model.add(ConvLSTM2D(filters=filter3,
                                     kernel_size=(config['kernel_size_3'], config['kernel_size_3']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if config['num_convlstm_layers'] > 3:
                model.add(ConvLSTM2D(filters=1,
                                     kernel_size=(config['kernel_size_4'], config['kernel_size_4']),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            model.add(Conv3D(filters=1,
                             kernel_size=(config['kernel_size_5'], config['kernel_size_5'], config['kernel_size_5']),
                             activation='relu', padding='same', data_format='channels_first'))

            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.RMSprop(lr=config['learning_rate']),
                          metrics=['accuracy'])

            model.summary()

            model.fit(self.x_train, self.y_train,
                      batch_size=config['batch_size'],
                      epochs=int(budget),
                      shuffle='batch',
                      validation_data=(self.x_test, self.y_test))

            train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
            validation_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
            test_score = model.evaluate(self.x_test, self.y_test, verbose=0)
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

        file1 = json.dumps(config)
        file2 = json.dumps(result)
        appendResult(self.path, [file1, file2, "\n"])

        return result

    @staticmethod
    def get_configspace(num_channels):
        configuration_space = CS.ConfigurationSpace()

        # training configurations
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1,
                                                       default_value='1e-2', log=True)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [1, 2, 4, 8])

        configuration_space.add_hyperparameter(learning_rate)
        configuration_space.add_hyperparameter(batch_size)

        # hyperparameters which can effect channel selection
        num_convlstm_layers = CSH.UniformIntegerHyperparameter('num_convlstm_layers',
                                                               lower=1, upper=4, default_value=3, log=True)
        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1',
                                                         lower=1, upper=num_channels, default_value=1, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2',
                                                         lower=1, upper=num_channels, default_value=1, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3',
                                                         lower=1, upper=num_channels, default_value=1, log=True)
        kernel_size_1 = CSH.CategoricalHyperparameter('kernel_size_1', [1, 3, 5, 7, 9])
        kernel_size_2 = CSH.CategoricalHyperparameter('kernel_size_2', [1, 3, 5, 7, 9])
        kernel_size_3 = CSH.CategoricalHyperparameter('kernel_size_3', [1, 3, 5, 7, 9])
        kernel_size_4 = CSH.CategoricalHyperparameter('kernel_size_4', [1, 3, 5, 7, 9])
        kernel_size_5 = CSH.CategoricalHyperparameter('kernel_size_5', [1, 3, 5, 7, 9])

        configuration_space.add_hyperparameters([num_convlstm_layers,
                                                 num_filters_1, num_filters_2, num_filters_3,
                                                 kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4])
        configuration_space.add_hyperparameter(kernel_size_5)

        # adding conditions
        childs = [num_filters_1, num_filters_2, num_filters_3,
                  kernel_size_2, kernel_size_3, kernel_size_4]
        numbers = [1, 2, 3, 1, 2, 3]
        parent = num_convlstm_layers

        for i in range(len(childs)):
            condition = CS.GreaterThanCondition(childs[i], parent, numbers[i])
            configuration_space.add_condition(condition)

        print("configuration space ended")
        print(configuration_space)

        return configuration_space


def appendResult(path, contents):
    if not os.path.exists(path):
        os.mknod(path)

    with open(path, 'a+') as file:
        for content in contents:
            file.write(content)


if __name__ == "__main__":
    dataset = []
    information = []
    channels = 1

    # worker = COVIDWorker2(run_id='all cases', dataset=dataset, information=information)
    # config_space = worker.get_configspace(channels)
    # config = config_space.sample_configuration().get_dictionary()
    # print(config)
    # result = worker.compute(config=config)
    # print(result)
