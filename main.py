from src.Cases import Index, Index2, Index3, Index4, Index5, Index6, Index7, Index8, Index9, Index10, Index11, Index12, \
    Index13, Index14, Index15, Index16, Index17, Index18
from src.process import set_gpu, Process


def main(index):
    set_gpu()

    # process1 = Process("12th_1st", Index())
    # process1.load_then_predict()

    process2 = Process("12th_2nd", Index2())
    process2.load_then_predict()

    process3 = Process("12th_3rd", Index3())
    process3.load_then_predict()

    process4 = Process("12th_4th", Index4())
    process4.load_then_predict()

    process5 = Process("12th_5th", Index5())
    process5.load_then_predict()

    process6 = Process("12th_6th", Index6())
    process6.load_then_predict()

    process7 = Process("12th_7th", Index7())
    process7.load_then_predict()

    process8 = Process("12th_8th", Index8())
    process8.load_then_predict()

    process9 = Process("12th_9th", Index9())
    process9.load_then_predict()

    process10 = Process("12th_10th", Index10())
    process10.load_then_predict()

    process11 = Process("12th_11th", Index11())
    process11.load_then_predict()

    process12 = Process("12th_12th", Index12())
    process12.load_then_predict()

    process13 = Process("12th_13th", Index13())
    process13.load_then_predict()

    process14 = Process("12th_14th", Index14())
    process14.load_then_predict()

    process15 = Process("12th_15th", Index15())
    process15.load_then_predict()

    process16 = Process("12th_16th", Index16())
    process16.load_then_predict()

    process17 = Process("12th_17th", Index17())
    process17.load_then_predict()

    process18 = Process("12th_18th", Index18())
    process18.load_then_predict()


if __name__ == "__main__":
    main(index=10)
