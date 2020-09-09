import sys

from src.Cases import Index, Index2, Index3, Index4, Index5, Index6, Index7, Index8, Index9, Index10, Index11, Index12, \
    Index13, Index14, Index15, Index16, Index17, Index18, Index19, Index20, Index21, Index22, Index23, Index24, Index25, \
    Index26, Index27, Index28, Index29, Index30, Index31, Index32, Index33, Index34, Index35, Index36, Index37, Index38, \
    Index39, Index40, Index41, Index42, Index43, Index44, Index45, Index46, Index47, Index48, Index49, Index50, Index51, \
    Index52, Index53, Index54, Index55, Index56, Index57, Index58, Index59, Index60, Index61, Index62, Index63, Index64, \
    Index65, Index66, Index67, Index68, Index69, Index70, Index71, Index72, Index73, Index74, Index75, Index76, Index77, \
    Index78, Index79, Index80, Index81, Index82, Index83, Index84, Index85, Index86, Index87
from src.process import set_gpu, Process


NEW_NAMES = ["22th_1th", "22th_3th", "22th_4th", "22th_5th", "22th_69th", "22th_18th", "22th_11th", "22th_13th", "22th_83th"]
FILE_NAMES = ["13th_1th", "15th_3th", "14th_4th", "16th_5th", "17th_69th", "21th_18th", "18th_11th", "19th_13th", "20th_83th"]
PROCESS = [Index(), Index3(), Index4(), Index5(), Index69(), Index18(), Index11(), Index13(), Index83()]
DETAILS = ["all cases", "developmental_stage", "remove_age", "remove_gender", "merge_three_infection_cases",
           "remove_infection_cases", "hospital_facilities", "transportation_facilities", "bnf"]

PARAMETERS = [9, 9, 100, 1,
              9, 2, 300, 1,
              9, 7, 100, 1,
              3, 3, 200, 16,
              5, 9, 200, 1,
              3, 3, 200, 16,
              2, 3, 300, 16,
              3, 2, 200, 1,
              2, 2, 300, 16]


def main(index):
    set_gpu()

    index = [Index(), Index(), Index2(), Index3(), Index4(), Index5(), Index6(), Index7(), Index8(), Index9(), Index10(), Index11(), Index12(), \
    Index13(), Index14(), Index15(), Index16(), Index17(), Index18(), Index19(), Index20(), Index21(), Index22(), Index23(), Index24(), Index25(), \
    Index26(), Index27(), Index28(), Index29(), Index30(), Index31(), Index32(), Index33(), Index34(), Index35(), Index36(), Index37(), Index38(), \
    Index39(), Index40(), Index41(), Index42(), Index43(), Index44(), Index45(), Index46(), Index47(), Index48(), Index49(), Index50(), Index51(), \
    Index52(), Index53(), Index54(), Index55(), Index56(), Index57(), Index58(), Index59(), Index60(), Index61(), Index62(), Index63(), Index64(), \
    Index65(), Index66(), Index67(), Index68(), Index69(), Index70(), Index71(), Index72(), Index73(), Index74(), Index75(), Index76(), Index77(), \
    Index78(), Index79(), Index80(), Index81(), Index82(), Index83(), Index84(), Index85(), Index86(), Index87()]

    # sys.stdout = open("test_filters_stdout.txt", "w")

    process = Process("test_filters", Index())
    process.train_then_predict()

    # sys.stdout.close()

    # for i in range(len(NEW_NAMES)):
    #     sys.stdout = open(NEW_NAMES[i] + ".txt", "w")
    #
    #     process = Process(FILE_NAMES[i], PROCESS[i])
    #     process.load_then_predict(DETAILS[i] + ".h5")
    #
    #     sys.stdout.close()



if __name__ == "__main__":
    main(index=10)

