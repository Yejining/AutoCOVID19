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


class IndexCause(Index):
    def __init__(self):
        super().__init__()
        self.counts = [11, 2, 1, 21, 7]
        self.causes = []

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return -1


class IndexVisitType(Index):
    def __init__(self):
        super().__init__()
        self.counts = [11, 2, 4, 1, 7]
        self.visit_types = []

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1


class IndexAge(Index):
    def __init__(self,):
        super().__init__()
        self.counts = [1, 2, 4, 21, 7]
        self.age = ""

    def age_category(self, age):
        if age == self.age: return 0
        else: return -1


class IndexSex(Index):
    def __init__(self):
        super().__init__()
        self.counts = [11, 1, 4, 21, 7]
        self.sex = ""

    def sex_category(self, sex):
        if sex == self.sex: return 0
        else: return -1


class IndexDay(Index):
    def __init__(self):
        super().__init__()
        self.counts = [11, 2, 4, 21, 1]
        self.day = 0

    def day_category(self, day):
        if day == self.day: return 0
        else: return -1


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


class Index18(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 0, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                       'beauty_salon', 'school', 'church', 'bank', 'cafe',
                       'bar', 'post_office', 'real_estate_agency', 'lodging',
                       'public_transportation', 'restaurant', 'etc', 'store',
                       'hospital', 'pharmacy', 'airport']
        self.causes = []

    def infection_case_category(self, infection_case):
        return -1


class Index19(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 4, 1, 7]
        self.visit_types = ['church']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1


class Index20(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 3, 2, 0]
        self.visit_types = ['hospital', 'pharmacy']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 2

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1

    def day_category(self, day):
        return -1


class Index21(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [5, 0, 3, 2, 0]
        self.visit_types = ['hospital', 'pharmacy']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def age_category(self, age):
        age = int(age[:-1])
        if age == 0: return 0
        elif age == 10: return 1
        elif age == 20: return 2
        elif age == 30 or age == 40 or age == 50 or age == 60: return 3
        else: return 4

    def sex_category(self, sex):
        return -1

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return self.causes.index(infection_case)
        else:
            return 2

    def type_category(self, visit_type, move_types):
        if move_types in self.causes:
            return move_types.index(visit_type)
        else:
            return -1

    def day_category(self, day):
        return -1


class Index22(IndexCause):
    def __init__(self):
        super().__init__()
        self.causes = ['community infection']


class Index23(IndexCause):
    def __init__(self):
        super().__init__()
        self.causes = ['etc']


class Index24(IndexCause):
    def __init__(self):
        super().__init__()
        self.causes = ['contact with patient']


class Index25(IndexCause):
    def __init__(self):
        super().__init__()
        self.causes = ['overseas inflow']


class Index26(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['karaoke']


class Index27(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['gas_station']


class Index28(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['gym']


class Index29(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['bakery']


class Index30(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['pc_cafe']


class Index31(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['beauty_salon']


class Index32(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['school']


class Index33(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['church']


class Index34(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['bank']


class Index35(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['cafe']


class Index36(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['bar']


class Index37(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['post_office']


class Index38(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['real_estate_agency']


class Index39(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['lodging']


class Index40(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['public_transportation']


class Index41(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['restaurant']


class Index42(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['etc']


class Index43(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['store']


class Index44(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['hospital']


class Index45(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['pharmacy']


class Index46(IndexVisitType):
    def __init__(self):
        super().__init__()
        self.visit_types = ['airport']


class Index47(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "0s"


class Index48(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "10s"


class Index49(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "20s"


class Index50(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "30s"


class Index51(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "40s"


class Index52(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "50s"


class Index53(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "60s"


class Index54(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "70s"


class Index55(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "80s"


class Index56(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "90s"


class Index57(IndexAge):
    def __init__(self):
        super().__init__()
        self.age = "100s"


class Index58(IndexSex):
    def __init__(self):
        super().__init__()
        self.sex = "male"


class Index59(IndexSex):
    def __init__(self):
        super().__init__()
        self.sex = "female"


class Index60(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 0


class Index61(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 1


class Index62(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 2


class Index63(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 3


class Index64(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 4


class Index65(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 5


class Index66(IndexDay):
    def __init__(self):
        super().__init__()
        self.day = 6


class Index67(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 1, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = []

    def infection_case_category(self, infection_case):
        return 0


class Index68(Index):
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
            return -1


class Index69(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [11, 2, 1, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'contact with patient', 'overseas inflow']

    def infection_case_category(self, infection_case):
        if infection_case in self.causes:
            return 0
        else:
            return -1


class Index70(Index):
    def __init__(self):
        super().__init__()
        self.names = ['age', 'sex', 'infection_case', 'type', 'date']
        self.counts = [1, 2, 4, 21, 7]
        self.visit_types = ['karaoke', 'gas_station', 'gym', 'bakery', 'pc_cafe',
                            'beauty_salon', 'school', 'church', 'bank', 'cafe',
                            'bar', 'post_office', 'real_estate_agency', 'lodging',
                            'public_transportation', 'restaurant', 'etc', 'store',
                            'hospital', 'pharmacy', 'airport']
        self.causes = ['community infection', 'etc', 'contact with patient', 'overseas inflow']

    def age_category(self, age):
        return 0