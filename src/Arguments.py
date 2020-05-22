import pandas as pd
from datetime import datetime, timedelta


class PathInfo:
    patient_daily_path = 'patient_daily/'
    patient_daily_3days_accumulated_path = 'patient_daily_3days_accumulated/'
    all_daily_path = 'all_daily/'
    figure_path = 'figure/'
    dataset_path = 'merged.h5'
    dataset_figure_path = 'figure.h5'
    model_path = 'model.h5'
    pred_path = 'pred/'
    x_test_path = 'x_test/'
    y_test_path = 'y_test/'
    y_pred_path = 'pred/'
    readme_path = 'README.md'
    scaled_path = 'scaled/'

    def __init__(self, index=0, appendix=None):
        self.name = '../covid_datatset/{}th'.format(str(index))
        if appendix is not None: self.name += '_{}/'.format(appendix)

    def get_route_saving_path(self):
        return self.name + self.patient_daily_path

    def get_accumulated_route_saving_path(self):
        return self.name + self.patient_daily_3days_accumulated_path

    def get_complete_route_saving_path(self):
        return self.name + self.all_daily_path

    def get_dataset_path(self):
        return self.name + self.dataset_path

    def get_model_path(self):
        return self.name + self.model_path

    def get_pred_path(self):
        return self.name + self.pred_path

    def get_x_test_path(self):
        return self.name + self.pred_path + self.x_test_path

    def get_y_test_path(self):
        return self.name + self.pred_path + self.y_test_path

    def get_y_pred_path(self):
        return self.name + self.pred_path + self.y_pred_path

    def get_y_scaled_path(self):
        return self.name + self.pred_path + self.scaled_path


class RouteInfo:
    def __init__(self, route_path=None):
        if route_path is not None: self.routes = pd.read_csv(route_path)
        else: self.routes = pd.read_csv('../covid_dataset/raw/MergedRoute.csv')

        self.dates = self.unique_value(self.routes, 'date')
        self.patients = self.unique_value(self.routes, 'patient_id')

        self.day_info()

    def unique_value(self, array, property):
        array = array[property]
        array = array.drop_duplicates(keep='last')
        array = array.tolist()
        array.sort()
        return array

    def day_info(self):
        self.first_day = datetime.strptime(self.dates[0], "%Y-%m-%d")
        self.last_day = datetime.strptime(self.dates[-1], "%Y-%m-%d") + timedelta(days=3)
        delta = self.last_day - self.first_day
        self.duration = delta.days + 1

    def get_places(self, patient):
        return self.routes[self.routes['patient_id'] == patient]

    def get_patient_day_info(self, patient):
        places = self.get_places(patient)
        dates = self.unique_value(places, 'date')
        first_day = datetime.strptime(dates[0], "%Y-%m-%d")
        last_day = datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=3)
        delta = last_day - first_day
        duration = delta.days + 1
        return places, dates, first_day, last_day, duration


class ImageInfo:
    def __init__(self, size=255, kernel_size=60, weight=5):
        self.size = size
        self.kernel_size = kernel_size
        self.weight = weight


class FeatureInfo:
    def __init__(self, names=None, counts=None, visit_types=None, causes=None):
        if names is not None: self.names = names
        else: self.names = ['age', 'sex', 'infection_case', 'type', 'date']

        if counts is not None: self.counts = counts
        else: self.counts = [11, 2, 4, 21, 7]

        if visit_types is not None: self.visit_type = visit_types
        else: self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']

        if causes is not None: self.cause = causes
        else: self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def get_all_counts(self):
        return sum(count for count in self.counts)


class ModelInfo:
    def __init__(self, channel, split_num=0, n_step=3,
                 epochs=200, batch_size=1,
                 optimizer='rmsprop', loss='mean_squared_error', activation='relu'):
        self.channel = channel
        self.split_num = split_num
        self.n_step = n_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer # 'adadelta'
        self.loss = loss # 'binary_crossentropy'
        self.activation = activation # 'relu', 'sigmoid

    def set_split_num(self, sample_num):
        if self.split_num == 0: self.split_num = int(sample_num * 0.7)
        return self.split_num


class GeneralInfo:
    def save_readme(self, route, image, feature, model, path):
        read = ['- dataset info',
                '  - start_date: ',
                '  - end_date: ',
                '  - duration: ',
                '- setting data preprocessing',
                '  - image_size: ',
                '  - kde_bandwidth: ',
                '  - weight_per_person: ',
                '- setting feature',
                '  - feature_names: ',
                '  - feature_counts: ',
                '  - visit_types: ',
                '  - causes: ',
                '- setting model',
                '  - epochs: ',
                '  - batch_size: ',
                '  - optimizer: ',
                '  - loss: ',
                '  - activation: ',
                '  - #train, #test: ',
                'extended 3 days of figure set']

        read[1] += datetime.strftime(route.first_day, "%Y-%m-%d")
        read[2] += datetime.strftime(route.last_day, "%Y-%m-%d")
        read[3] += str(route.duration)

        read[5] += str(image.size)
        read[6] += str(image.kernel_size)
        read[7] += str(image.weight)

        counts = map(str, feature.counts)
        read[8] += ', '.join(feature.names)
        read[10] += ', '.join(counts)
        read[11] += ', '.join(feature.visit_types)
        read[12] += ', '.join(feature.causes)

        sample_num = route.duration - (2 * model.n_step)
        train_num = int(sample_num * 0.7)
        test_num = sample_num - train_num
        read[14] += str(model.epochs)
        read[15] += str(model.batch_size)
        read[16] += model.optimizer
        read[17] += model.loss
        read[18] += model.activation
        read[18] += str(train_num) + ", " + str(test_num)

        read = '\n'.join(read)

        with open(path.name + path.readme_path, 'w') as f:
            f.write(read)
