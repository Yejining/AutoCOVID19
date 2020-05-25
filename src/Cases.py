from datetime import datetime


class Index:
    def __init__(self):
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def get_all_counts(self):
        return sum(count for count in self.counts)

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

    def infection_case_category(self, infection_case):
        return self.causes.index(infection_case)

    def type_category(self, visit_type, move_types):
        return move_types.index(visit_type)

    def day_category(self, day):
        day = datetime.strptime(day, "%Y-%m-%d")
        return day.weekday()

    def get_y_set_index(self):
        for count in self.counts:
            if count != 0:
                return count


class Index2(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [5, 2, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                                'beauty_salon', 'school', 'church', 'bank', 'cafe',
                                'bar', 'post_office', 'real_estate_agency', 'lodging',
                                'public_transportation', 'restaurant', 'etc', 'store',
                                'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def age_category(self, age):
        age = int(age[:-1])
        if age == 0 or age == 10: return 0
        elif age == 20 or age == 30: return 1
        elif age == 40 or age == 50: return 2
        elif age == 60 or age == 70: return 3
        else: return 4


class Index3(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [5, 2, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def age_category(self, age):
        age = int(age[:-1])
        if age == 0: return 0
        elif age == 10: return 1
        elif age == 20: return 2
        elif age == 30 or age == 40 or age == 50 or age == 60: return 3
        else: return 4


class Index4(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [0, 2, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def age_category(self, age):
        return -1


class Index5(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 0, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def sex_category(self, sex):
        return -1


class Index6(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 3, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return -1


class Index7(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 3, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 1


class Index8(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 2, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 1


class Index9(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 3, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 0


class Index10(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 3, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 2


class Index11(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 2, 7]
        self.visit_types = ['hospital', 'pharmacy']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1


class Index12(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 5, 7]
        self.visit_types = ['karaoke', 'gym', 'pc_cafe', 'school', 'church']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1


class Index13(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 2, 7]
        self.visit_types = ['public_transportation', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1


class Index14(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 0, 7]
        self.visit_types = []
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def type_category(self, visit_type, move_types):
        return -1


class Index15(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 21, 4]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def day_category(self, day):
        day = datetime.strptime(day, "%Y-%m-%d")
        weekday = day.weekday()
        if weekday == 0 or weekday == 1: return 0
        elif weekday == 2 or weekday == 3: return 1
        elif weekday == 4: return 2
        else: return 3


class Index16(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 21, 2]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def day_category(self, day):
        day = datetime.strptime(day, "%Y-%m-%d")
        weekday = day.weekday()
        if weekday == 5 or weekday == 6: return 1
        else: return 0


class Index17(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 21, 0]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def day_category(self, day):
        return -1