import pandas

from src.Cases import Index, IndexAge, IndexGender, IndexCause, IndexVisitType
from src.constant import AGE_MODE, GENDER_MODE, CAUSE_MODE, VISIT_MODE, N_STEP, SIZE
from src.process import set_gpu, Process

import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

CASES = [Index(), IndexAge(AGE_MODE.DEVELOPMENTAL),
         IndexAge(AGE_MODE.REMOVE), IndexGender(GENDER_MODE.REMOVE),
         IndexCause(CAUSE_MODE.MERGE_THREE), IndexCause(CAUSE_MODE.REMOVE),
         IndexVisitType(VISIT_MODE.HOSPITAL), IndexVisitType(VISIT_MODE.TRANSPORTATION),
         IndexVisitType(VISIT_MODE.FNB), IndexVisitType(VISIT_MODE.REMOVE)]


class Worker:
    def __init__(self, dataset, information, configurations):
        self.dataset = dataset
        self.information = information
        self.configurations = configurations

        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_validation = dataset[2]
        self.y_validation = dataset[3]
        self.x_test = dataset[4]
        self.y_test = dataset[5]

        self.n_step = information[0]
        self.n_channel = information[1]
        self.size = information[2]
        self.path = information[3]

    def train(self, path):
        num_layers = int(self.configurations['num_convlstm_layers'].values[0])
        filter1 = 1
        filter2 = 1
        filter3 = 1

        if num_layers == 1:
            filter1 = 1
        elif num_layers == 2:
            filter1 = int(self.configurations['num_filters_1'].values[0])
            filter2 = 1
        elif num_layers == 3:
            filter1 = int(self.configurations['num_filters_1'].values[0])
            filter2 = int(self.configurations['num_filters_2'].values[0])
            filter3 = 1
        else:
            filter1 = int(self.configurations['num_filters_1'].values[0])
            filter2 = int(self.configurations['num_filters_2'].values[0])
            filter3 = int(self.configurations['num_filters_3'].values[0])

        with K.tf_ops.device('/GPU:0'):
            model = Sequential()

            kernel_size_1 = int(self.configurations['kernel_size_1'].values[0])
            model.add(ConvLSTM2D(filters=filter1,
                                 kernel_size=(kernel_size_1, kernel_size_1),
                                 data_format='channels_first',
                                 input_shape=(self.n_step, self.n_channel, self.size, self.size),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

            if int(self.configurations['num_convlstm_layers'].values[0]) > 1:
                kernel_size_2 = int(self.configurations['kernel_size_2'].values[0])
                model.add(ConvLSTM2D(filters=filter2,
                                     kernel_size=(kernel_size_2, kernel_size_2),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if int(self.configurations['num_convlstm_layers'].values[0]) > 2:
                kernel_size_3 = int(self.configurations['kernel_size_3'].values[0])
                model.add(ConvLSTM2D(filters=filter3,
                                     kernel_size=(kernel_size_3, kernel_size_3),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            if int(self.configurations['num_convlstm_layers'].values[0]) > 3:
                kernel_size_4 = int(self.configurations['kernel_size_4'].values[0])
                model.add(ConvLSTM2D(filters=1,
                                     kernel_size=(kernel_size_4, kernel_size_4),
                                     padding='same', return_sequences=True,
                                     data_format='channels_first'))
                model.add(BatchNormalization())

            kernel_size_5 = int(self.configurations['kernel_size_5'].values[0])
            model.add(Conv3D(filters=1,
                             kernel_size=(kernel_size_5, kernel_size_5, kernel_size_5),
                             activation='relu', padding='same', data_format='channels_first'))

            learning_rate = int(self.configurations['learning_rate'].values[0])
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                          metrics=['accuracy'])

            model.summary()

            batch_size = int(self.configurations['batch_size'].values[0])
            model.fit(self.x_train, self.y_train,
                      batch_size=batch_size,
                      epochs=300,
                      shuffle='batch',
                      validation_data=(self.x_test, self.y_test))
            model.save(path + "model.h5")
            score = model.evaluate(self.x_validation, self.y_validation)

            return score

    def predict(self, path):
        model = load_model(path + "model.h5")
        pred = model.predict(self.x_test)
        return pred


def train_than_predict(hyperparameter, index):
    name = hyperparameter['id'].values[0]
    print(name)

    process = Process('hpbandster_' + str(index + 1) + 'th_' + name, CASES[index])
    process.load_dataset()

    dataset = []
    dataset.append(process.X_train)
    dataset.append(process.y_train)
    dataset.append(process.X_eval)
    dataset.append(process.y_eval)
    dataset.append(process.X_test)
    dataset.append(process.y_test)

    information = []
    information.append(N_STEP)  # n_step
    information.append(process.feature_info.get_all_counts())  # channels
    information.append(SIZE)  # image size
    information.append(process.path_info.get_result_path())

    path = process.path_info.name

    print("training")
    worker = Worker(dataset, information, hyperparameter)
    worker.train(path)

    print("predicting")
    pred = worker.predict(path)
    process.predict_from_outside(pred)

    print("saving")
    process.save_prediction()
    process.save_readme()
    process.statistic_raw_data()


if __name__ == "__main__":
    hyperparameters = pandas.read_csv('hpo.csv')

    for i in range(0, 1):
        train_than_predict(hyperparameters[i:i + 1], i)
