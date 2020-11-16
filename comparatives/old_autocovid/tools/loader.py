import pandas as pd

from os.path import join
from datetime import datetime


class Dataset:
    def __init__(self, args):
        self.args = args

        print('initialize dataset class')
        self.route_df = pd.read_csv(join(self._get_data_path(), 'AccumulatedMergedRoute.csv'))
        self.route_df['date'] = pd.to_datetime(self.route_df['date'])

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self._set_columns()

    def _initialize_dataset(self):
        self._set_columns()
        self._tune_feature_depth()
        self._split_dataset()
        setattr(self.args, 'channel', self.get_channel_length())

    def _set_columns(self):
        print('set columns')
        column_path = self._get_data_path()
        self.age_column_df = pd.read_csv(join(column_path, 'age_column.csv'))
        self.gender_column_df = pd.read_csv(join(column_path, 'gender_column.csv'))
        self.reason_column_df = pd.read_csv(join(column_path, 'reason_column.csv'))
        self.type_column_df = pd.read_csv(join(column_path, 'type_column.csv'))

        print('set feature lists by type')
        age_depth = self.args.general_depth if not hasattr(self.args, 'age_depth') else self.args.age_depth
        self.age_column = self.age_column_df.columns[age_depth - 1]
        gender_depth = self.args.general_depth if not hasattr(self.args, 'gender_depth') is None else self.args.gender_depth
        self.gender_column = self.gender_column_df.columns[gender_depth - 1]
        type_depth = self.args.general_depth if not hasattr(self.args, 'type_depth') is None else self.args.type_depth
        self.type_column = self.type_column_df.columns[type_depth - 1]
        reason_depth = self.args.general_depth if not hasattr(self.args, 'reason_depth') is None else self.args.reason_depth
        self.reason_column = self.reason_column_df.columns[reason_depth - 1]

        unique_age = self.get_unique_values(self.age_column_df, self.age_column)
        unique_gender = self.get_unique_values(self.gender_column_df, self.gender_column)
        unique_type = self.get_unique_values(self.type_column_df, self.type_column)
        unique_reason = self.get_unique_values(self.reason_column_df, self.reason_column)

        self.age_features = unique_age[self.age_column].to_list()
        self.gender_features = unique_gender[self.gender_column].to_list()
        self.type_features = unique_type[self.type_column].to_list()
        self.reason_features = unique_reason[self.reason_column].to_list()

        print('feature type: %s' % self.args.feature_type)
        # select one feature of selected feature type
        if self.args.feature_depth == 1 and self.args.feature_name != 'all':
            if self.args.feature_type == 'age':
                self.age_features = [self.args.feature_name]
            elif self.args.feature_type == 'gender':
                self.gender_features = [self.args.feature_name]
            elif self.args.feature_type == 'type':
                self.type_features = [self.args.feature_name]
            elif self.args.feature_type == 'reason':
                self.reason_features = [self.args.feature_name]
        # remove all features of selected feature type
        elif self.args.feature_depth == 2:
            if self.args.feature_type == 'age':
                self.age_features = []
            elif self.args.feature_type == 'gender':
                self.gender_features = []
            elif self.args.feature_type == 'type':
                self.type_features = []
            elif self.args.feature_type == 'reason':
                self.reason_features = []

        setattr(self.args, 'age_features', self.age_features)
        setattr(self.args, 'gender_features', self.gender_features)
        setattr(self.args, 'type_features', self.type_features)
        setattr(self.args, 'reason_features', self.reason_features)

        print('age_features: %s' % self.age_features)
        print('gender_features: %s' % self.gender_features)
        print('type_features: %s' % self.type_features)
        print('reason_features: %s' % self.reason_features)

    def _tune_feature_depth(self):
        print('tune feature depth')
        for index, row in self.route_df.iterrows():
            age_row = self.age_column_df[self.age_column_df['age'] == row['age']].iloc[0]
            self.route_df.loc[index, 'age'] = age_row[self.age_column]

            gender_row = self.gender_column_df[self.gender_column_df['sex'] == row['sex']].iloc[0]
            self.route_df.loc[index, 'sex'] = gender_row[self.gender_column]

            type_row = self.type_column_df[self.type_column_df['type'] == row['type']].iloc[0]
            self.route_df.loc[index, 'type'] = type_row[self.type_column]

            reason_row = self.reason_column_df[self.reason_column_df['infection_case'] == row['infection_case']].iloc[0]
            self.route_df.loc[index, 'infection_case'] = reason_row[self.reason_column]

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

    def load_dataset(self):
        self._initialize_dataset()

        dataset = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}
        return dataset

    def get_channel_length(self):
        if self.args.feature_type == 'one': return 1

        channel_len = len(self.age_features) + len(self.gender_features) + \
                      len(self.type_features) + len(self.reason_features)
        return channel_len

    def get_unique_values(self, df, column):
        series = df[column].drop_duplicates()
        series = series.sort_values(ascending=True)
        series = series.reset_index(drop=True)
        return series.to_frame(name=column)

    def _get_data_path(self):
        data_path = join(self.args.root, 'comparatives', 'old_autocovid', 'data')
        return data_path
