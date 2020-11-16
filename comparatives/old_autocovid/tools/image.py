from datetime import timedelta, datetime
from os.path import join
from pathlib import Path
from PIL import Image

import json
import h5py
import math
import numpy as np


class ImageGenerator:
    def __init__(self, args):
        self.args = args

    def generate_and_save_dataset(self, route_dict):
        train_set = self.get_image_dataset(route_dict['train'])
        val_set = self.get_image_dataset(route_dict['val'])
        test_set = self.get_image_dataset(route_dict['test'])

        dataset_list = self.normalize([train_set, val_set, test_set])
        self.save_image_dataset(dataset_list[0], 'train')
        self.save_image_dataset(dataset_list[1], 'val')
        self.save_image_dataset(dataset_list[2], 'test')

        args_dict = dict()
        for arg in vars(self.args):
            args_dict.update({arg: str(getattr(self.args, arg))})

        with open(join(join(self._get_path()), 'args.json'), 'w') as f:
            json.dump(args_dict, f)

        return dataset_list

    def get_image_dataset(self, route_df):
        print('get image dataset with given dataframe')
        image_dict = self._generate_route_images(route_df)
        dataset = self._convert_route_images_to_dataset(image_dict)

        start_date = image_dict['start_date']
        end_date = image_dict['end_date']
        x_set = dataset['x_set']
        y_set = dataset['y_set']

        result = {'x_set': x_set, 'y_set': y_set,
                  'start_date': start_date, 'end_date': end_date}

        return result

    def save_images(self, dataset, name):
        print('save %s image to local' % name)
        path = join(self._get_image_path(), name)

        x_set = dataset['x_set']
        y_set = dataset['y_set']
        start_date = dataset['start_date']

        print('save x_%s set' % name)
        for sample in range(x_set.shape[0]):
            sample_path = join(path, 'x_set', 'sample%04d' % sample)
            date = start_date + timedelta(days=sample)

            for step in range(x_set.shape[1]):
                step_date = date + timedelta(days=step)
                str_date = step_date.strftime('%Y-%m-%d')
                step_path = join(sample_path, str_date)
                Path(step_path).mkdir(parents=True, exist_ok=True)

                for channel in range(x_set.shape[2]):
                    image_path = join(step_path, '%d.png' % channel)
                    array = x_set[sample, step, channel, ]
                    self._save_image(array, image_path)

        print('save y_%s set' % name)
        image_path = join(path, 'y_set')
        Path(image_path).mkdir(parents=True, exist_ok=True)
        for sample in range(y_set.shape[0]):
            date = start_date + timedelta(days=sample)
            str_date = date.strftime('%Y-%m-%d')
            sample_path = join(image_path, '%s.png' % str_date)

            array = y_set[sample, 0, 0]
            self._save_image(array, sample_path)

    def _save_image(self, array, path):
        print('saving %s' % path)
        array = array.astype('uint8')
        image = Image.fromarray(array, mode='L')
        image.save(path)

    def save_image_dataset(self, dataset, name):
        print('save image %s dataset' % name)

        start_date = dataset['start_date']
        start = [start_date.year, start_date.month, start_date.day]

        end_date = dataset['end_date']
        end = [end_date.year, end_date.month, end_date.day]

        x_set = dataset['x_set']
        y_set = dataset['y_set']

        h5_path = join(self._get_dataset_path(), '%s.h5' % name)
        with h5py.File(h5_path, 'w') as f:
            x_set = f.create_dataset('x_%s' % name, data=x_set)
            y_set = f.create_dataset('y_%s' % name, data=y_set)
            start_date = f.create_dataset('start_date', data=start)
            end_date = f.create_dataset('end_date', data=end)

    def load_image_dataset(self):
        print('load image dataset')
        train_set = self._load_single_dataset('train')
        val_set = self._load_single_dataset('val')
        test_set = self._load_single_dataset('test')

        return train_set, val_set, test_set

    def normalize(self, dataset_list):
        x_max = 0
        y_max = 0

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

        return result_list

    def _normalize(self, array, data_type):
        max_value = self.args.x_max if data_type == 'x' else self.args.y_max
        divided_by_max = array / max_value
        normalized = divided_by_max * 255
        return normalized

    def _normalize_x(self, x_set):
        for sample in range(x_set.shape[0]):
            for step in range(x_set.shape[1]):
                for channel in range(x_set.shape[2]):
                    array = x_set[sample, step, channel, ]
                    x_set[sample, step, channel, ] = self._normalize(array, 'x')

        return x_set

    def _normalize_y(self, y_set):
        for sample in range(y_set.shape[0]):
            array = y_set[sample, 0, 0, ]
            y_set[sample, 0, 0] = self._normalize(array, 'y')

        return y_set

    def _load_single_dataset(self, name):
        print('load %s dataset' % name)
        h5_path = join(self._get_dataset_path(), '%s.h5' % name)
        f = h5py.File(h5_path, 'r')
        x_set = f['x_%s' % name]
        y_set = f['y_%s' % name]
        start = f['start_date']
        end = f['end_date']

        start_date = datetime(start[0], start[1], start[2])
        end_date = datetime(end[0], end[1], end[2])

        dataset = {'x_set': x_set, 'y_set': y_set,
                   'start_date': start_date, 'end_date': end_date}
        return dataset

    def _generate_route_images(self, route_df):
        print('generate images of visited places')

        start_date = route_df['date'].min()
        end_date = route_df['date'].max()
        delta = end_date - start_date

        x_day_list = []
        y_day_list = []
        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            print('generate image of date %s' % day.strftime('%Y-%m-%d'))

            day_images = self._generate_day_image(route_df, day)
            x_day_list.append(day_images['x_set'])
            y_day_list.append(day_images['y_set'])

        print('apply triangular kernel on generated images')
        x_images = np.stack(x_day_list) # day, channel, size, size
        y_images = np.stack(y_day_list) # day, size, size

        for day in range(x_images.shape[0]):
            for channel in range(x_images.shape[1]):
                image = x_images[day, channel, ]
                x_images[day, channel, ] = self._apply_triangular_kernel(image)

        for day in range(y_images.shape[0]):
            image = y_images[day, 0, 0, ]
            y_images[day, 0, 0, ] = self._apply_triangular_kernel(image)

        result = {'x_set': x_images, 'y_set': y_images,
                  'start_date': start_date, 'end_date': end_date}
        return result

    def _convert_route_images_to_dataset(self, image_dict):
        print('convert generated images to dataset')

        x_images = image_dict['x_set'] # day, channel, size, size
        y_images = image_dict['y_set']

        x_list = []
        for day in range(x_images.shape[0] - self.args.n_step):
            day_images = []
            for step in range(self.args.n_step):
                day_images.append(x_images[day + step, ])
            x_list.append(np.stack(day_images)) # n_step, channel, size, size
        x_set = np.stack(x_list) # sample, n_step, channel, size, size

        y_images = y_images[self.args.n_step: ]
        y_set = np.stack(y_images)

        x_set = np.asarray(x_set)
        y_set = np.asarray(y_set)

        dataset = {'x_set': x_set, 'y_set': y_set}
        return dataset

    def _generate_day_image(self, route_df, day):
        age_features = self.args.age_features
        gender_features = self.args.gender_features
        type_features = self.args.type_features
        reason_features = self.args.reason_features

        channel = len(age_features) + len(gender_features) + \
                  len(type_features) + len(reason_features)
        if self.args.feature_type == 'one': channel = 1

        day_df = route_df.loc[route_df['date'] == day]
        x_images = np.zeros((channel, self.args.size, self.args.size))
        y_images = np.zeros((1, 1, self.args.size, self.args.size))

        for index, row in day_df.iterrows():
            row_value = int(row['row'])
            col_value = int(row['col'])

            y_images[0, 0, row_value, col_value] += self.args.weight

            if self.args.feature_type == 'one':
                x_images[0, row_value, col_value] += self.args.weight
                continue

            if row['age'] in age_features:
                age_index = age_features.index(row['age'])
                x_images[age_index, row_value, col_value] += self.args.weight

            if row['sex'] in gender_features:
                gender_index = gender_features.index(row['sex'])
                gender_index += len(age_features)
                x_images[gender_index, row_value, col_value] += self.args.weight

            if row['type'] in type_features:
                type_index = type_features.index(row['type'])
                type_index += len(age_features) + len(gender_features)
                x_images[type_index, row_value, col_value] += self.args.weight

            if row['infection_case'] in reason_features:
                reason_index = reason_features.index(row['infection_case'])
                reason_index += len(age_features) + len(gender_features) + len(type_features)
                x_images[reason_index, row_value, col_value] += self.args.weight

        images = {'x_set': x_images, 'y_set': y_images}
        return images

    def _apply_triangular_kernel(self, array):
        new_image = np.zeros((array.shape))

        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                if array[row][col] == 0: continue

                value = array[row][col]
                kde_applied = self._triangular_kernel(array.shape, row, col, value)
                new_image += kde_applied

        return new_image

    def _triangular_kernel(self, shape, row, col, value):
        array = np.zeros((shape))
        stride = int((self.args.kde_kernel_size - 1) / 2)
        ratio = 1 / (stride + 1)

        for i in range(row - stride, row + stride + 1):
            if i < 0 or i >= array.shape[0]: continue
            for j in range(col - stride, col + stride + 1):
                if j < 0 or j >= array.shape[1]: continue
                distance = math.sqrt((row - i) ** 2 + (col - j) ** 2)
                new_value = value * (1 - (distance * ratio))
                if new_value < 0: new_value = 0
                array[i][j] = new_value

        array[row][col] = value
        return array

    def _get_path(self):
        feature_level = 'feature_level_%d' % self.args.feature_depth
        dataset_path = join(self.args.root, 'comparatives', 'old_autocovid', 'dataset',
                            feature_level, self.args.feature_type, self.args.name)
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        return dataset_path

    def _get_image_path(self):
        image_path = join(self._get_path(), 'images')
        Path(image_path).mkdir(parents=True, exist_ok=True)
        return image_path

    def _get_dataset_path(self):
        dataset_path = join(self._get_path(), 'dataset')
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        return dataset_path
