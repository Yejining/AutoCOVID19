from src.Cases import Index, Index2, Index3, Index4, Index5, Index6, Index7, Index8, Index9, Index10, Index11, Index12, \
    Index13, Index14, Index15, Index16, Index17, Index18
from src.process import set_gpu, Process


def main(index):
    set_gpu()

    process18 = Process("12th_18th", Index18())
    process18.train_then_predict()


if __name__ == "__main__":
    main(index=10)
