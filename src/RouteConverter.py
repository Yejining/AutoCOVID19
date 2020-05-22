import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from PIL import Image


class RouteToIndexConverter:
    def __init__(self, path_info, route_info, image_info, feature_info, model_info):
        self.path_info = path_info
        self.route_info = route_info
        self.image_info = image_info
        self.feature_info = feature_info
        self.model_info = model_info

    def age_category(self, age):
        age = int(age[:-1])
        if age == 0:
            return 0
        elif age == 100:
            return 10
        return age // 10

    def sex_category(self, sex):
        if sex == 'male': return 0
        return 1

    def infection_case_category(self, infection_case, causes):
        return causes.index(infection_case)

    def type_category(self, visit_type, move_types):
        return move_types.index(visit_type)

    def day_category(self, day):
        day = datetime.strptime(day, "%Y-%m-%d")
        return day.weekday()

    def convert_original_route(self):
        path = self.path_info.get_route_saving_path()
        for patient in self.route_info.patients:
            self.save_raw_patient_route(path, patient)

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
        patient_places = self.route_info.get_places(patient)

        patient_path = path + str(patient)
        Path(patient_path).mkdir(parents=True, exist_ok=True)

        today = self.route_info.first_day
        while True:
            today_str = datetime.strftime(today, "%Y-%m-%d")
            patient_day_places = patient_places[patient_places['date'] == today_str]
            places_indices = self.combine_places(patient_day_places)
            patient_date_path = patient_path + "/" + today_str + '/'
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
        all_counts = self.feature_info.get_all_counts
        size = self.image_info.size
        weight = self.image_info.weight

        visit_grid = np.zeros((all_counts, size, size))

        for index in place_indices:
            row = index[5]
            col = index[6]
            for feature in range(5):
                visit_grid[index[feature]][row][col] += weight

        for channel in range(visit_grid.shape[0]):
            self.save_grid(path + str(channel) + ".png", visit_grid[channel])

    def df_to_grid_index(self, one_visit):
        index = 0
        p_age = self.age_category(one_visit['age'])
        index += self.feature_info.counts[0]
        p_sex = self.sex_category(one_visit['sex']) + index
        index += self.feature_info.counts[1]
        p_infection_case = self.infection_case_category(one_visit['infection_case'], self.feature_info.causes) + index
        index += self.feature_info.counts[2]
        p_type = self.type_category(one_visit['type'], self.feature_info.visit_types) + index
        index += self.feature_info.counts[3]
        p_date = self.day_category(one_visit['date']) + index
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
        complete_routes = self.get_complete_routes()
        kernel_size = self.image_info.kernel_size
        size = self.image_info.size
        dataset = np.zeros((len(complete_routes), kernel_size, size, size))

        for i, days in enumerate(complete_routes):
            self.get_array_image(days, dataset[i, :, :, :])

        return dataset

    def get_array_image(self, place_indices, data_array):
        for index in place_indices[1]:
            row = index[5]
            col = index[6]
            for feature in range(5):
                data_array[index[feature]][row][col] += self.image_info.weight

        for channel in range(data_array.shape[0]):
            data_array[channel] = self.overlay_kernel(data_array[channel])

        return data_array

    def save_prediction_image(self, X_test, y_test, pred, start_day):
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
        first_day = first_day + timedelta(days=3)
        for l_sample in range(y_test.shape[0]):
            sample_path = self.path_info.get_y_test_path() + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, y_test[l_sample][0])
            first_day += timedelta(days=1)

        # pred
        first_day = first_day + timedelta(days=3)
        for l_sample in range(pred.shape[0]):
            sample_path = self.path_info.get_y_pred_path + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            Path(sample_path).mkdir(parents=True, exist_ok=True)
            self.array_save_image(sample_path, pred[l_sample][0])

            sample_path = self.path_info.get_y_scaled_path() + datetime.strftime(first_day, "%Y-%m-%d") + '/'
            self.array_save_image(sample_path, pred[l_sample][0], scaled=True)
            first_day += timedelta(days=1)

    def array_save_image(self, path, array, scaled=False):
        for channel in range(array.shape[0]):
            if scaled:
                new_array = array[channel]
                new_array[new_array >= 0] *= 255
                self.save_grid(path + str(channel) + ".png", new_array, kernel=False)
            else:
                self.save_grid(path + str(channel) + ".png", array, kernel=False)
