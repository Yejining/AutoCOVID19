import pandas as pd

from os.path import join
from pathlib import Path
from datetime import datetime, timedelta


class Dataset:
    def __init__(self, args):
        self.args = args

        print('initialize dataset class')
        extracted_path = join(self.args.root, 'data', 'extracted')
        self.route_df = pd.read_csv(join(extracted_path, 'merged_route_final.csv'))
        self.route_df['date'] = pd.to_datetime(self.route_df['date'])

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self._set_columns()

    def _initialize_dataset(self):
        self._set_columns()
        self._tune_feature_depth()
        self._split_dataset()

        self.train_df = self._accumulate_dataset(self.train_df, 'train')
        self.val_df = self._accumulate_dataset(self.val_df, 'val')
        self.test_df = self._accumulate_dataset(self.test_df, 'test')

    def _set_columns(self):
        print('set columns')
        column_path = join(self.args.root, 'data', 'checklist')
        self.city_column_df = pd.read_csv(join(column_path, 'city_column.csv'))
        self.type_column_df = pd.read_csv(join(column_path, 'type_column.csv'))
        self.reason_column_df = pd.read_csv(join(column_path, 'infection_case_column.csv'))

        print('set feature lists by type')
        self.city_column = self.city_column_df.columns[self.args.feature_depth]
        self.type_column = self.type_column_df.columns[self.args.feature_depth]
        self.reason_column = self.reason_column_df.columns[self.args.feature_depth]

        unique_city = self.get_unique_values(self.city_column_df, self.city_column)
        unique_type = self.get_unique_values(self.type_column_df, self.type_column)
        unique_reason = self.get_unique_values(self.reason_column_df, self.reason_column)

        self.city_features = unique_city[self.city_column].to_list()
        self.type_features = unique_type[self.type_column].to_list()
        self.reason_features = unique_reason[self.reason_column].to_list()

        print('feature type: %s' % self.args.feature_type)
        if self.args.feature_depth == 1:
            if self.args.feature_type == 'city':
                self.city_features = [self.args.feature_name]
            elif self.args.feature_type == 'type':
                self.type_features = [self.args.feature_name]
            elif self.args.feature_type == 'reason':
                self.reason_features = [self.args.feature_name]
        elif self.args.feature_depth == 2:
            if self.args.feature_type == 'city':
                self.city_features = []
            elif self.args.feature_type == 'type':
                self.type_features = []
            elif self.args.feature_type == 'reason':
                self.reason_features = []

        setattr(self.args, 'city_features', self.city_features)
        setattr(self.args, 'type_features', self.type_features)
        setattr(self.args, 'reason_features', self.reason_features)

    def _tune_feature_depth(self):
        print('tune feature depth')
        for index, row in self.route_df.iterrows():
            city_row = self.city_column_df[self.city_column_df['city'] == row['city']].iloc[0]
            self.route_df.loc[index, 'city'] = city_row[self.city_column]

            type_row = self.type_column_df[self.type_column_df['type'] == row['type']].iloc[0]
            self.route_df.loc[index, 'type'] = type_row[self.type_column]

            reason_row = self.reason_column_df[self.reason_column_df['infection_case'] == row['infection_case']].iloc[0]
            self.route_df.loc[index, 'infection_case'] = reason_row[self.reason_column]

        if self.args.is_logged:
            path = join(self.args.root, 'dataset', self.args.feature_type, self.args.name)
            Path(path).mkdir(parents=True, exist_ok=True)

            print('saving feature depth tuned dataset')
            self.route_df.to_csv(join(path, 'feature_tuned_route.csv'), encoding='utf-8-sig', index=False)

    def _split_dataset(self):
        print('split dataset')
        train_start = datetime.strptime(self.args.train_start, '%Y-%m-%d')
        train_end = datetime.strptime(self.args.train_end, '%Y-%m-%d')
        val_start = datetime.strptime(self.args.val_start, '%Y-%m-%d')
        val_end = datetime.strptime(self.args.val_end, '%Y-%m-%d')
        test_start = datetime.strptime(self.args.test_start, '%Y-%m-%d')
        test_end = datetime.strptime(self.args.test_end, '%Y-%m-%d')

        train_mask = (self.route_df.date >= train_start) & (self.route_df.date <= train_end)
        val_mask = (self.route_df.date >= val_start) & (self.route_df.date <= val_end)
        test_mask = (self.route_df.date >= test_start) & (self.route_df.date <= test_end)

        self.train_df = self.route_df.loc[train_mask]
        self.val_df = self.route_df.loc[val_mask]
        self.test_df = self.route_df.loc[test_mask]

        if self.args.is_logged:
            feature_level = 'feature_level_%d' % self.args.feature_depth
            path = join(self.args.root, 'dataset', feature_level, self.args.feature_type, self.args.name)
            Path(path).mkdir(parents=True, exist_ok=True)

            print('saving train_df, val_df, test_df')
            self.train_df.to_csv(join(path, 'train.csv'), encoding='utf-8-sig', index=False)
            self.val_df.to_csv(join(path, 'val.csv'), encoding='utf-8-sig', index=False)
            self.test_df.to_csv(join(path, 'test.csv'), encoding='utf-8-sig', index=False)

    def _accumulate_dataset(self, dataset_df, name=''):
        new_rows = []

        print('accumulate visits of %s set' % name)

        for index, row in dataset_df.iterrows():
            new_row = row
            new_date = row['date']

            for day in range(1, self.args.accumulating_days):
                new_row['date'] = new_date + timedelta(days=day)
                new_rows.append(new_row.copy())

        for new_row in new_rows:
            dataset_df = dataset_df.append(new_row, ignore_index=True)

        dataset_df = dataset_df.sort_values(by='date')
        dataset_df = dataset_df.reset_index(drop=True)

        if self.args.is_logged:
            feature_level = 'feature_level_%d' % self.args.feature_depth
            path = join(self.args.root, 'dataset', feature_level, self.args.feature_type, self.args.name)
            Path(path).mkdir(parents=True, exist_ok=True)

            print('saving accumulated %s set' % name)
            filepath = join(path, 'accumulated_%s.csv' % name)
            dataset_df.to_csv(filepath, encoding='utf-8-sig', index=False)

        return dataset_df

    def load_dataset(self):
        self._initialize_dataset()
        dataset = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}
        return dataset

    def get_channel_length(self):
        if self.args.feature_type == 'one': return 1

        channel_len = len(self.city_features) + len(self.type_features) + len(self.reason_features)
        return channel_len

    def get_unique_values(self, df, column):
        series = df[column].drop_duplicates()
        series = series.sort_values(ascending=True)
        series = series.reset_index(drop=True)
        return series.to_frame(name=column)
