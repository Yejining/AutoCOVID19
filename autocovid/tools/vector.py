import json
import h5py
import numpy as np
import pandas as pd

from pathlib import Path
from os.path import join
from datetime import datetime, timedelta


class VectorGenerator:
    def __init__(self, args):
        self.args = args

        status_path = join(self.args.root, 'data', 'extracted', 'Korea_Covid_Patient.csv')
        self.status_df = pd.read_csv(status_path)
        self.status_df['data_date'] = pd.to_datetime(self.status_df['data_date'])

    def generate_and_save_dataset(self, route_dict):
        vector_dict = self.get_vector_dataset(route_dict)
        normalized_dict = self.normalize(vector_dict)
        self.save_vector_dataset(normalized_dict)

        saving_path = join(self.args.root, 'dataset', self.args.model_type,
                           'feature_level_%d' % self.args.feature_depth,
                           self.args.feature_type, self.args.name)
        Path(saving_path).mkdir(parents=True, exist_ok=True)

        args_dict = dict()
        for arg in vars(self.args):
            args_dict.update({arg: str(getattr(self.args, arg))})

        with open(join(saving_path, 'args.json'), 'w') as f:
            json.dump(args_dict, f)

        return normalized_dict

    def get_vector_dataset(self, route_dict):
        print('get vector dataset with given dataset')

        route_df_list = []
        route_df_list.append(route_dict['train'])
        route_df_list.append(route_dict['val'])
        route_df_list.append(route_dict['test'])

        result_list = []
        for route_df in route_df_list:
            vector_dict = self._generate_route_vectors(route_df)
            dataset = self._convert_route_vectors_to_dataset(vector_dict)

            start_date = vector_dict['start_date']
            end_date = vector_dict['end_date']
            x_set = dataset['x_set']
            y_set = dataset['y_set']

            result = {'x_set': x_set, 'y_set': y_set,
                      'start_date': start_date, 'end_date': end_date}
            result_list.append(result)

        dataset_dict = {'train': result_list[0],
                        'val': result_list[1],
                        'test': result_list[2]}

        return dataset_dict

    def normalize(self, dataset_dict):
        x_max = 0
        y_max = 0

        dataset_list = []
        dataset_list.append(dataset_dict['train'])
        dataset_list.append(dataset_dict['val'])
        dataset_list.append(dataset_dict['test'])

        for dataset in dataset_list:
            x_set = dataset['x_set']
            y_set = dataset['y_set']

            x_max = np.max(x_set) if np.max(x_set) > x_max else x_max
            y_max = np.max(y_set) if np.max(y_set) > y_max else y_max

        setattr(self.args, 'x_max', x_max)
        setattr(self.args, 'y_max', y_max)

        result_list = []
        for dataset in dataset_list:
            normalized_x = self._normalize_x(dataset['x_set'])
            normalized_y = self._normalize_y(dataset['y_set'])
            result = {'x_set': normalized_x, 'y_set': normalized_y,
                      'start_date': dataset['start_date'], 'end_date': dataset['end_date']}
            result_list.append(result)

        result_dict = {'train': result_list[0],
                       'val': result_list[1],
                       'test': result_list[2]}

        return result_dict

    def save_vector_dataset(self, dataset_dict):
        names = ['train', 'val', 'test']

        for name in names:
            print('save vector %s' % name)

            start_date = dataset_dict[name]['start_date']
            start = [start_date.year, start_date.month, start_date.day]

            end_date = dataset_dict[name]['end_date']
            end = [end_date.year, end_date.month, end_date.day]

            x_set = dataset_dict[name]['x_set']
            y_set = dataset_dict[name]['y_set']

            feature_level = 'feature_level_%d' % self.args.feature_depth
            path = join(self.args.root, 'dataset', self.args.model_type, feature_level,
                        self.args.feature_type, self.args.name, 'dataset')
            Path(path).mkdir(parents=True, exist_ok=True)

            h5_path = join(path, '%s.h5' % name)
            with h5py.File(h5_path, 'w') as f:
                x_set = f.create_dataset('x_%s' % name, data=x_set)
                y_set = f.create_dataset('y_%s' % name, data=y_set)
                start_date = f.create_dataset('start_date', data=start)
                end_date = f.create_dataset('end_date', data=end)
                x_max = f.create_dataset('x_max', data=self.args.x_max)
                y_max = f.create_dataset('y_max', data=self.args.y_max)

    def load_vector_dataset(self):
        print('load vector dataset')
        train_set = self._load_single_dataset('train')
        val_set = self._load_single_dataset('val')
        test_set = self._load_single_dataset('test')

        return train_set, val_set, test_set

    def _generate_route_vectors(self, route_df):
        print('generate vectors of visited places')

        start_date = route_df['date'].min()
        end_date = route_df['date'].max()
        delta = end_date - start_date

        x_day_list = []
        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            print('generate vector of date %s' % day.strftime('%Y-%m-%d'))
            x_day_list.append(self._generate_day_vector(route_df, day))
        x_vectors = np.stack(x_day_list)

        status_path = join(self.args.root, 'data', 'extracted', 'Korea_Covid_Patient.csv')
        status_df = pd.read_csv(status_path)
        y_vectors = np.zeros(((delta.days + 1), 1, 1))

        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            str_date = day.strftime('%Y%m%d')

            new_df = status_df.loc[status_df['data_date'] == int(str_date)].iloc[0]
            y_vectors[i, 0, 0] = new_df['new_pat']

        result = {'x_set': x_vectors, 'y_set': y_vectors,
                  'start_date': start_date, 'end_date': end_date}
        return result

    def _generate_day_vector(self, route_df, day):
        city_features = self.args.city_features
        type_features = self.args.type_features
        reason_features = self.args.reason_features

        channel = len(city_features) + len(type_features) + len(reason_features)

        day_df = route_df.loc[route_df['date'] == day]
        x_vectors = np.zeros((channel, 1))

        for index, row in day_df.iterrows():
            if row['city'] in city_features:
                city_index = city_features.index(row['city'])
                x_vectors[city_index, 0] += 1

            if row['type'] in type_features:
                type_index = type_features.index(row['type'])
                x_vectors[type_index, 0] += 1

            if row['infection_case'] in reason_features:
                reason_index = reason_features.index(row['infection_case'])
                x_vectors[reason_index, 0] += 1

        return x_vectors

    def _convert_route_vectors_to_dataset(self, vector_dict):
        print('convert generated vectors to dataset')

        x_vectors = vector_dict['x_set']
        y_vectors = vector_dict['y_set']

        x_list = []
        for day in range(x_vectors.shape[0] - self.args.frame_in):
            first_day = x_vectors[day, ]
            second_day = x_vectors[day + 1, ]
            third_day = x_vectors[day + 2, ]

            days_vector = np.concatenate([first_day, second_day, third_day], axis=1)
            transposed = days_vector.transpose()

            x_list.append(transposed)
        x_set = np.stack(x_list)

        y_images = y_vectors[self.args.frame_in: ]
        y_set = np.stack(y_images)

        x_set = np.asarray(x_set)
        y_set = np.asarray(y_set)

        dataset = {'x_set': x_set, 'y_set': y_set}
        return dataset

    def _normalize(self, array, data_type):
        max_value = self.args.x_max if data_type == 'x' else self.args.y_max
        divided_by_max = array / max_value
        normalized = divided_by_max * 1
        return normalized

    def _normalize_x(self, x_set): # sample, step, channel, 3
        for sample in range(x_set.shape[0]):
            for step in range(x_set.shape[1]):
                for channel in range(x_set.shape[2]):
                    array = x_set[sample, step, channel,]
                    x_set[sample, step, channel,] = self._normalize(array, 'x')

        return x_set

    def _normalize_y(self, y_set): # sample, 1, 1
        for sample in range(y_set.shape[0]):
            array = y_set[sample, 0, ]
            y_set[sample, 0, ] = self._normalize(array, 'y')

        return y_set

    def _load_single_dataset(self, name):
        print('load %s dataset' % name)
        feature_level = 'feature_level_%d' % self.args.feature_depth
        path = join(self.args.root, 'dataset', self.args.model_type, feature_level,
                    self.args.feature_type, self.args.name, 'dataset')

        f = h5py.File(join(path, '%s.h5' % name), 'r')
        x_set = f.get('x_%s' % name).value
        y_set = f.get('y_%s' % name).value
        start = f['start_date']
        end = f['end_date']
        x_max = f['x_max'][()]
        y_max = f['y_max'][()]

        start_date = datetime(start[0], start[1], start[2])
        end_date = datetime(end[0], end[1], end[2])

        dataset = {'x_set': x_set, 'y_set': y_set,
                   'start_date': start_date, 'end_date': end_date,
                   'x_max': x_max, 'y_max': y_max}
        return dataset
