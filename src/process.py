import csv

from keras.models import load_model
import tensorflow as tf
from pathlib import Path
import numpy as np

from src.Arguments import PathInfo, RouteInfo, ImageInfo, ModelInfo, GeneralInfo
from src.Dataset import Dataset
from src.Model import Model
from src.RouteConverter import RouteToIndexConverter


class Process:
    def __init__(self, name, feature_info):
        self.initial_setting(name, feature_info)

    def initial_setting(self, name, feature_info):
        self.path_info = PathInfo(name)
        self.route_info = RouteInfo()
        self.image_info = ImageInfo()
        self.feature_info = feature_info
        self.model_info = ModelInfo(self.feature_info.get_all_counts())
        self.general_info = GeneralInfo()
        self.converter = RouteToIndexConverter(
            self.path_info, self.route_info, self.image_info, self.feature_info, self.model_info)
        self.loader = Dataset(self.route_info, self.model_info)
        self.trainer = Model(self.model_info, self.path_info, self.route_info, self.feature_info)

        self.dataset = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.start_day1 = None
        self.start_day2 = None
        self.model = None
        self.pred = None

    # save raw csv file into images
    def save_raw_route(self):
        print("saving original route")
        self.converter.convert_original_route()
        print()

        print("saving accumulated route")
        self.converter.convert_accumulated_route()
        print()

        print("saving complete route")
        self.converter.convert_complete_route()
        print()

    # save image in h5 format
    def save_route_in_h5(self):
        self.X_set, self.y_set = self.converter.get_dataset()
        self.loader.save_dataset(self.path_info.get_dataset_path(), self.X_set, self.y_set, self.route_info.first_day)

    def correlate(self, sequence=True):
        visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                        'beauty_salon', 'school', 'church', 'bank', 'cafe',
                        'bar', 'post_office', 'real_estate_agency', 'lodging',
                        'public_transportation', 'restaurant', 'etc', 'store',
                        'hospital', 'pharmacy', 'airport']
        correlation_matrix = np.zeros((len(visit_types), len(visit_types)))
        correlation_matrix = self.converter.correlate(visit_types, correlation_matrix, sequence)
        self.converter.create_correlation_file(visit_types, correlation_matrix)


    # get dataset(images) from file
    def load_dataset(self):
        path = self.path_info.get_dataset_path()
        self.X_train, self.y_train, self.X_test, self.y_test, self.start_day1, self.start_day2\
            = self.loader.load_data_from_file(path)

    def train(self):
        self.trainer.train(self.X_train, self.y_train, self.image_info.size)

    def predict(self):
        self.model = load_model(self.path_info.get_model_path())
        self.pred = self.model.predict(self.X_test)
        self.diff1, self.diff2, self.diff3, self.diff4, self.rmse, self.mape,\
        self.test_max_value, self.pred_max_value, self.rmse_1_added, self.mape_1_added=\
            self.trainer.get_accuracy(self.pred, self.y_test)

    def save_accuracy(self):
        self.converter.save_accuracy(self.start_day2, self.diff1,
                                     self.rmse, self.mape,
                                     self.test_max_value, self.pred_max_value,
                                     self.rmse_1_added, self.mape_1_added)

    def save_prediction(self):
        print(self.X_test.shape, self.y_test.shape, self.pred.shape)
        self.converter.save_prediction_image(self.X_test, self.y_test, self.pred, self.start_day2,
                                             self.diff1, self.diff2, self.diff3, self.diff4, self.rmse, self.mape,
                                             self.test_max_value, self.pred_max_value,
                                             self.rmse_1_added, self.mape_1_added)

    def save_readme(self):
        self.general_info.save_readme(
            self.route_info, self.image_info, self.feature_info, self.model_info, self.path_info)

    def statistic_raw_data(self):
        raw_routes_combined = self.converter.statistic_by_day()
        accumulated_routes_combined = self.converter.statistic_by_day(raw=False)

        Path(self.path_info.get_statistics_path()).mkdir(parents=True, exist_ok=True)
        with open(self.path_info.get_statistics_path() + "raw_route.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(raw_routes_combined)

        with open(self.path_info.get_statistics_path() + "accumulated_route.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(accumulated_routes_combined)

    def one_cycle(self):
        self.save_raw_route()
        self.save_route_in_h5()
        self.load_dataset()
        self.train()
        self.predict()
        self.save_prediction()
        self.save_readme()
        self.statistic_raw_data()

    def train_then_predict(self):
        Path(self.path_info.name).mkdir(parents=True, exist_ok=True)
        self.save_route_in_h5()
        self.load_dataset()
        self.train()
        self.predict()
        self.save_prediction()
        self.save_readme()
        self.statistic_raw_data()

    def load_then_predict(self):
        self.load_dataset()
        self.predict()
        self.save_prediction()
        # self.save_readme()
        # self.statistic_raw_data()

    def load_then_save_accuracy(self):
        self.load_dataset()
        self.predict()
        self.save_accuracy()


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
