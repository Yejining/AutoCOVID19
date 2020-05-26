import csv
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from csv import writer
import numpy as np
from PIL import Image


class RouteToIndexConverter:
    def __init__(self, path_info, route_info, image_info, feature_info, model_info):
        self.path_info = path_info
        self.route_info = route_info
        self.image_info = image_info
        self.feature_info = feature_info
        self.model_info = model_info

    def convert_original_route(self):
        path = self.path_info.get_route_saving_path()
        for patient in self.route_info.patients:
            if patient <= 1000000034: continue
            self.save_raw_patient_route(path, patient)

    def initialize_combined_route(self):
        today = self.route_info.first_day
        combined_route = []
        while True:
            features = [0] * (self.feature_info.get_all_counts() + 1)
            features[0] = datetime.strftime(today, "%Y-%m-%d")
            combined_route.append(features)

            if today == self.route_info.last_day: break
            today += timedelta(days=1)

        return combined_route

    def statistic_by_day(self, raw=True):
        routes_combined = self.initialize_combined_route()

        for patient in self.route_info.patients:
            if raw: routes = self.get_patient_route(patient)
            else: routes = self.accumulate_patient(patient)

            for each_route in routes:
                route_day = datetime.strptime(each_route[0], "%Y-%m-%d")
                route_places = each_route[1]
                index = (route_day - self.route_info.first_day).days
                if len(route_places) == 0: continue
                for each_place in route_places:
                    age = each_place[0]
                    if age != -1: routes_combined[index][age + 1] += 1
                    sex = each_place[1]
                    if sex != -1: routes_combined[index][sex + 1] += 1
                    infection = each_place[2]
                    if infection != -1: routes_combined[index][infection + 1] += 1
                    purpose = each_place[3]
                    if purpose != -1:  routes_combined[index][purpose + 1] += 1
                    day = each_place[4]
                    if day != -1: routes_combined[index][day + 1] += 1

        return routes_combined

    def convert_accumulated_route(self):
        path = self.path_info.get_accumulated_route_saving_path()
        for patient in self.route_info.patients:
            accumulated_routes = self.accumulate_patient(patient)
            self.save_accumulated_patient_route(path, patient, accumulated_routes)

    def convert_complete_route(self):
        path = self.path_info.get_complete_route_saving_path()
        complete_routes = self.get_complete_routes()
        for days in complete_routes:
            date_path = path + days[0] + '/'
            Path(date_path).mkdir(parents=True, exist_ok=True)
            self.indices_save_image(date_path, days[1])

    def save_raw_patient_route(self, path, patient):
        print("saving raw patient route: ", patient)
        patient_places = self.route_info.get_places(patient)

        patient_path = path + str(patient)
        Path(patient_path).mkdir(parents=True, exist_ok=True)

        today = self.route_info.first_day
        while True:
            print(today, end=' ')
            today_str = datetime.strftime(today, "%Y-%m-%d")
            patient_day_places = patient_places[patient_places['date'] == today_str]
            places_indices = self.combine_places(patient_day_places)
            patient_date_path = patient_path + "/" + today_str + "/"
            print(patient_date_path)
            Path(patient_date_path).mkdir(parents=True, exist_ok=True)
            self.indices_save_image(patient_date_path, places_indices)
            if today == self.route_info.last_day: break
            today += timedelta(days=1)

    def save_accumulated_patient_route(self, path, patient, patient_routes):
        path += str(patient) + '/'
        Path(path).mkdir(parents=True, exist_ok=True)
        for routes in patient_routes:
            patient_date_path = path + routes[0] + '/'
            Path(patient_date_path).mkdir(parents=True, exist_ok=True)
            self.indices_save_image(patient_date_path, routes[1])

    def combine_places(self, places):
        indices = []
        for i in range(len(places)):
            one_visit = places.iloc[i]
            indices.append(self.df_to_grid_index(one_visit))
        return indices

    def indices_save_image(self, path, place_indices):
        all_counts = self.feature_info.get_all_counts()
        size = self.image_info.size
        weight = self.image_info.weight

        visit_grid = np.zeros((all_counts, size, size))

        for index in place_indices:
            row = index[5]
            col = index[6]
            for feature in range(5):
                if feature == -1: continue
                visit_grid[index[feature]][row][col] += weight

        for channel in range(visit_grid.shape[0]):
            self.save_grid(path + str(channel) + ".png", visit_grid[channel])

    def df_to_grid_index(self, one_visit):
        index = 0
        p_age = self.feature_info.age_category(one_visit['age'])
        index += self.feature_info.counts[0]

        p_sex = self.feature_info.sex_category(one_visit['sex'])
        if p_sex != -1: p_sex += index
        index += self.feature_info.counts[1]

        p_infection_case = self.feature_info.infection_case_category(one_visit['infection_case'])
        if p_infection_case != -1: p_infection_case += index
        index += self.feature_info.counts[2]

        p_type = self.feature_info.type_category(one_visit['type'], self.feature_info.visit_types)
        if p_type != -1: p_type += index
        index += self.feature_info.counts[3]

        p_date = self.feature_info.day_category(one_visit['date'])
        if p_date != -1: p_date += index

        row = one_visit['row']
        col = one_visit['col']

        return [p_age, p_sex, p_infection_case, p_type, p_date, row, col]

    def save_grid(self, path, grid, kernel=True):
        if kernel: grid = self.overlay_kernel(grid)
        img = Image.fromarray(grid.astype('uint8'), 'L')
        img.save(path)

    def put_triangular_kernel(self, array, row, col, value):
        stride = int((self.image_info.kernel_size - 1) / 2)
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

    def overlay_kernel(self, array):
        new_image = np.zeros((array.shape[0], array.shape[1]))
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                if array[row][col] == 0: continue
                new_image += self.put_triangular_kernel(np.zeros((array.shape[0], array.shape[1])), row, col,
                                                   array[row][col])

        return new_image

    def accumulate_two_days(self, day1, day2):
        day2[1].extend(day1[1])
        return day2

    def accumulate_patient(self, patient):
        patient_route = self.get_patient_route(patient)
        patient_days = len(patient_route)

        second = patient_days - 1
        first = second - 1
        for i in range(2 * patient_days - 3):
            patient_route[second] = self.accumulate_two_days(patient_route[first], patient_route[second])
            if second - first == 2:
                second -= 1
            else:
                first -= 1

        return patient_route

    def get_patient_route(self, patient):
        patient_places = self.route_info.get_places(patient)

        patient_routes = []
        today = self.route_info.first_day
        while True:
            today_str = datetime.strftime(today, "%Y-%m-%d")
            patient_day_places = patient_places[patient_places['date'] == today_str]
            places_indices = self.combine_places(patient_day_places)
            patient_routes.append([today_str, places_indices])

            if today == self.route_info.last_day: break
            today += timedelta(days=1)

        return patient_routes

    def get_complete_routes(self):
        print("get complete routes")
        today = self.route_info.first_day
        complete_routes = []
        while True:
            today_str = datetime.strftime(today, "%Y-%m-%d")
            places = []
            complete_routes.append([today_str, places])

            if today == self.route_info.last_day: break
            today += timedelta(days=1)

        for patient in self.route_info.patients:
            accumulated_routes = self.accumulate_patient(patient)
            for each_route in accumulated_routes:
                route_day = datetime.strptime(each_route[0], "%Y-%m-%d")
                route_places = each_route[1]
                index = (route_day - self.route_info.first_day).days
                complete_routes[index][1].extend(route_places)
        return complete_routes

    def get_dataset(self):
        print("get_dataset")
        complete_routes = self.get_complete_routes()
        feature_size = self.feature_info.get_all_counts()
        size = self.image_info.size
        dataset = np.zeros((len(complete_routes), feature_size, size, size))

        for i, days in enumerate(complete_routes):
            print(days)
            self.get_array_image(days, dataset[i, :, :, :])

        print("split X_set and y_set_temp")
        n = dataset.shape[0]
        X_set = []
        y_set_temp = []
        for i in range(self.model_info.n_step, n):
            X_set.append(dataset[i - self.model_info.n_step:i, :, :, :])
            y_set_temp.append(dataset[i:i + 1, :, :, :])
        X_set = np.asarray(X_set)
        y_set_temp = np.asarray(y_set_temp)
        print(X_set.shape, y_set_temp.shape)

        print("combine features in y_set_temp")
        index_range = self.feature_info.get_y_set_index()
        print("going to combine %d features" % index_range)
        y_set = np.zeros((y_set_temp.shape[0], 1, 1, size, size))
        for i in range(len(y_set_temp)):
            for j in range(index_range):
                y_set[i][0][0] += y_set_temp[i, 0, j, :, :]

        print(X_set.shape, y_set.shape)

        return X_set, y_set

    def get_array_image(self, place_indices, data_array):
        for index in place_indices[1]:
            row = index[5]
            col = index[6]
            for feature in range(5):
                if feature == -1: continue
                data_array[index[feature]][row][col] += self.image_info.weight

        for channel in range(data_array.shape[0]):
            data_array[channel] = self.overlay_kernel(data_array[channel])

        return data_array

    def save_prediction_image(self, X_test, y_test, pred, start_day, diff, rmse, mape, test_max_value, pred_max_value):
        # X_test
        first_day = start_day
        for sample in range(X_test.shape[0]):
            sample_path = self.path_info.get_x_test_path() + 'sample%d/' % sample
            first_day2 = first_day
            for day in range(X_test.shape[1]):
                day_path = sample_path + datetime.strftime(first_day2, "%Y-%m-%d") + '/'
                Path(day_path).mkdir(parents=True, exist_ok=True)
                self.array_save_image(day_path, X_test[sample][day])
                first_day2 += timedelta(days=1)
            first_day += timedelta(days=1)

        # y_test
        first_day = start_day + timedelta(days=3)
        for l_sample in range(y_test.shape[0]):
            sample_path = self.path_info.get_y_test_path() + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, y_test[l_sample][0])
            first_day += timedelta(days=1)

        # pred
        first_day = start_day + timedelta(days=3)
        self.create_diff_file(self.path_info.get_diff_path(), self.path_info.diff_name)
        for l_sample in range(pred.shape[0]):
            sample_path = self.path_info.get_y_pred_path() + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, pred[l_sample][0])

            sample_path = self.path_info.get_y_scaled_path() + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, pred[l_sample][0], scaled=True)
            first_day += timedelta(days=1)

        # diff
        l_first_day = start_day + timedelta(days=3)
        for l_sample in range(diff.shape[0]):
            sample_path = self.path_info.get_diff_path() + datetime.strftime(l_first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, diff[l_sample])

            for l_feature in range(diff.shape[1]):
                self.append_list_as_row(self.path_info.get_diff_name(),
                                   [datetime.strftime(l_first_day, "%Y-%m-%d"), l_feature,
                                    mape[l_sample][l_feature], rmse[l_sample][l_feature],
                                    test_max_value[l_sample][l_feature], pred_max_value[l_sample][l_feature]])
            l_first_day += timedelta(days=1)

    def create_diff_file(self, file_path, file_name):
        Path(file_path).mkdir(parents=True, exist_ok=True)
        with open(file_path + file_name, 'w') as csvfile:
            headers = ['date', 'feature', 'mape', 'rmse', 'max', 'max_pred']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            writer.writeheader()

    def append_list_as_row(self, file_name, list_of_elem):
        print(list_of_elem[4])
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)

    def array_save_image(self, path, array, scaled=False):
        for channel in range(array.shape[0]):
            if scaled:
                new_array = array[channel]
                new_array[new_array >= 0] *= 255
                self.save_grid(path + str(channel) + ".png", new_array, kernel=False)
            else:
                self.save_grid(path + str(channel) + ".png", array[channel], kernel=False)
