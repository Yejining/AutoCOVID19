import csv

from keras.models import load_model
import tensorflow as tf

from src.Arguments import PathInfo, RouteInfo, ImageInfo, FeatureInfo, ModelInfo, GeneralInfo
from src.Dataset import Dataset
from src.Model import Model
from src.RouteConverter import RouteToIndexConverter


class Process:
    def __init__(self, index=10):
        self.initial_setting(index)

    def initial_setting(self, index=10):
        self.path_info = PathInfo(index)
        self.route_info = RouteInfo()
        self.image_info = ImageInfo()
        self.feature_info = FeatureInfo()
        self.model_info = ModelInfo(self.feature_info.get_all_counts)
        self.general_info = GeneralInfo()
        self.converter = RouteToIndexConverter(
            self.path_info, self.route_info, self.image_info, self.feature_info, self.model_info)
        self.loader = None
        self.trainer = None

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
        self.converter.convert_original_route()
        self.converter.convert_accumulated_route()
        self.converter.convert_complete_route()

    # save image in h5 format
    def save_route_in_h5(self):
        self.dataset = self.converter.get_dataset()
        self.loader = Dataset(self.dataset, self.route_info, self.model_info)
        self.loader.save_dataset(self.path_info.get_dataset_path(), self.route_info.first_day)

    # get dataset(images) from file
    def load_dataset(self):
        path = self.path_info.get_dataset_path()
        self.X_train, self.y_train, self.X_test, self.y_test, self.start_day1, self.start_day2\
            = self.loader.load_data_from_file(path)

    def train(self):
        self.trainer = Model(self.model_info, self.path_info, self.route_info)
        self.trainer.train(self.X_train, self.y_train, self.image_info.size)

    def predict(self):
        self.model = load_model(self.path_info.get_model_path())
        self.pred = self.model.predict(self.X_test)

    def save_prediction(self):
        self.converter.save_prediction_image(self.X_test, self.y_test, self.pred, self.start_day2)

    def save_readme(self):
        self.general_info.save_readme(
            self.route_info, self.image_info, self.feature_info, self.model_info, self.path_info)

    def statistic_raw_data(self):
        raw_routes_combined = self.converter.statistic_by_day()
        accumulated_routes_combined = self.converter.statistic_by_day(raw=False)

        with open(self.path_info.get_statistics_path() + "raw_route.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(raw_routes_combined)

        with open(self.path_info.get_statistics_path() + "accumulated_route.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(accumulated_routes_combined)


def main(index):
    set_gpu()
    process = Process(index=index)
    process.save_raw_route()
    process.save_route_in_h5()
    process.load_dataset()
    process.train()
    process.predict()
    process.save_prediction()
    process.save_readme()


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


if __name__ == "__main__":
    main(index=10)
