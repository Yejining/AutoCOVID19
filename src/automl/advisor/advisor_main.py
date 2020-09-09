import sys

from src.Cases import Index, Index4, Index3, Index5, Index69, Index11, Index13, Index83, Index18
from src.automl.advisor.advisor_util import FILE_NAME, TEST_NAME, main
from src.process import set_gpu, Process


# MergeAllInfectionCases 17th_69th
# HospitalFacilities 18th_11th
# TransportFacilities 19th_13th
# BeverageAndFood 20th_83th


def train(conv_kernel_size, convlstm_kernel_size, epoch, batch_size):
    process = Process(FILE_NAME, Index18())
    process.load_dataset()
    process.conv_kernel_size = conv_kernel_size
    process.convlstm_kernel_size = convlstm_kernel_size
    process.epoch = epoch
    process.batch_size = batch_size
    score = process.train()
    print("test loss and accuracy", score)

    return score


if __name__ == "__main__":
    for i in range(4):
        sys.stdout = open(TEST_NAME + ".txt", "w")
        set_gpu()
        main(train)
        sys.stdout.close()
