from src.constant import AGE_MODE, GENDER_MODE, CAUSE_MODE, VISIT_MODE


class GeneralCase:
    def __init__(self):
        self.names = ['age', 'sex', 'infection_case', 'type']
        self.counts = [11, 2, 4, 21]
        self.sex = ['male', 'female']
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
        return self.sex.index(sex)

    def infection_case_category(self, infection_case):
        return self.causes.index(infection_case)

    def type_category(self, move_types):
        return self.visit_types.index(move_types)


class AgeFeature(GeneralCase):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == AGE_MODE.ALL:
            self.counts = [11, 2, 4, 21]
        elif mode == AGE_MODE.EACH:
            self.counts = [1, 2, 4, 21]
        elif mode == AGE_MODE.DEVELOPMENTAL:
            self.counts = [5, 2, 4, 21]
        elif mode == AGE_MODE.REMOVE:
            self.counts = [0, 2, 4, 21]

        self.age = ""

    def age_category(self, age):
        if self.mode == AGE_MODE.ALL:
            age = int(age[:-1])
            if age == 0:
                return 0
            elif age == 100:
                return 10
            return age // 10
        elif self.mode == AGE_MODE.EACH:
            if age == self.age:
                return 0
        elif self.mode == AGE_MODE.DEVELOPMENTAL:
            age = int(age[:-1])
            if age == 0 or age == 10:
                return 0
            elif age == 20 or age == 30:
                return 1
            elif age == 40 or age == 50:
                return 2
            elif age == 60 or age == 70:
                return 3
            else:
                return 4
        elif self.mode == AGE_MODE.REMOVE:
            return -1


class GenderFeature(GeneralCase):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == GENDER_MODE.ALL:
            self.counts = [11, 2, 4, 21]
        elif mode == GENDER_MODE.EACH:
            self.counts = [11, 1, 4, 21]
        elif mode == GENDER_MODE.REMOVE:
            self.counts = [11, 0, 4, 21]

        self.sex = ""

    def sex_category(self, sex):
        if self.mode == GENDER_MODE.ALL:
            if sex in self.sex:
                return self.sex.index(sex)
            else:
                return -1
        elif self.mode == GENDER_MODE.EACH:
            if sex == self.sex:
                return 0
            else:
                return -1
        elif self.mode == GENDER_MODE.REMOVE:
            return -1


class VisitFeature(GeneralCase):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == VISIT_MODE.ALL:
            self.counts = [11, 2, 4, 21]
        elif mode == VISIT_MODE.EACH:
            self.counts = [11, 2, 4, 1]
        elif mode == VISIT_MODE.HOSPITAL:
            self.counts = [11, 2, 4, 1]
            self.visit_types = ['hospital', 'pharmacy']
        elif mode == VISIT_MODE.TRANSPORTATION:
            self.counts = [11, 2, 4, 1]
            self.visit_types = ['public_transportation', 'airport']
        elif mode == VISIT_MODE.FNB:
            self.counts = [11, 2, 4, 1]
            self.visit_types = ['bakery', 'cafe', 'bar', 'restaurant']
        elif mode == VISIT_MODE.REMOVE:
            self.counts = [11, 2, 4, 0]

    def type_category(self, move_types):
        if self.mode == VISIT_MODE.ALL:
            if move_types in self.visit_types:
                return move_types.index(move_types)
            else:
                return -1
        elif self.mode == VISIT_MODE.EACH:
            if move_types == self.visit_types:
                return 0
            else:
                return -1
        elif self.mode == VISIT_MODE.HOSPITAL or \
             self.mode == VISIT_MODE.TRANSPORTATION or \
             self.mode == VISIT_MODE.FNB:
            if move_types in self.visit_types:
                return 0
            else:
                return -1
        elif self.mode == VISIT_MODE.REMOVE:
            return -1


class CauseFeature(GeneralCase):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == CAUSE_MODE.ALL:
            self.counts = [11, 2, 4, 21]
        elif mode == CAUSE_MODE.EACH:
            self.counts = [11, 2, 1, 21]
        elif mode == CAUSE_MODE.MERGE_THREE:
            self.counts = [11, 2, 1, 21]
            self.causes = ['community infection', 'contact with patient', 'overseas inflow']
        elif mode == CAUSE_MODE.REMOVE:
            self.counts = [11, 2, 0, 21]

    def infection_case_category(self, infection_case):
        if self.mode == CAUSE_MODE.ALL:
            if infection_case in self.causes:
                return self.causes.index(infection_case)
            else:
                return -1
        elif self.mode == CAUSE_MODE.EACH:
            if infection_case == self.causes:
                return 0
            else:
                return -1
        elif self.mode == CAUSE_MODE.MERGE_THREE:
            if infection_case in self.causes:
                return 0
            else:
                return -1
        elif self.mode == CAUSE_MODE.REMOVE:
            return -1
