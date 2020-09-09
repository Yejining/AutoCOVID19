class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


AGE_MODE = Enum(['ALL', 'EACH', 'DEVELOPMENTAL', 'REMOVE'])
GENDER_MODE = Enum(['ALL', 'EACH', 'REMOVE'])
CAUSE_MODE = Enum(['ALL', 'EACH', 'MERGE_THREE', 'REMOVE'])
VISIT_MODE = Enum(['ALL', 'EACH', 'HOSPITAL', 'TRANSPORTATION', 'FNB', 'REMOVE'])

N_STEP = 3
SIZE = 255
