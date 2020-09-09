from src.Cases import Index, Index2, Index3, Index4, Index5, Index6, Index7, Index8, Index9, Index10, Index11, Index12, \
    Index13, Index14, Index15, Index16, Index17, Index18, Index19, Index20, Index21, Index22, Index23, Index24, Index25, \
    Index26, Index27, Index28, Index29, Index30, Index31, Index32, Index33, Index34, Index35, Index36, Index37, Index38, \
    Index39, Index40, Index41, Index42, Index43, Index44, Index45, Index46, Index47, Index48, Index49, Index50, Index51, \
    Index52, Index53, Index54, Index55, Index56, Index57, Index58, Index59, Index60, Index61, Index62, Index63, Index64, \
    Index65, Index66, Index67, Index68, Index69, Index70
from src.process import set_gpu, Process

def main(index):
    set_gpu()

    index = [Index(), Index(), Index2(), Index3(), Index4(), Index5(), Index6(), Index7(), Index8(), Index9(), Index10(), Index11(), Index12(), \
    Index13(), Index14(), Index15(), Index16(), Index17(), Index18(), Index19(), Index20(), Index21(), Index22(), Index23(), Index24(), Index25(), \
    Index26(), Index27(), Index28(), Index29(), Index30(), Index31(), Index32(), Index33(), Index34(), Index35(), Index36(), Index37(), Index38(), \
    Index39(), Index40(), Index41(), Index42(), Index43(), Index44(), Index45(), Index46(), Index47(), Index48(), Index49(), Index50(), Index51(), \
    Index52(), Index53(), Index54(), Index55(), Index56(), Index57(), Index58(), Index59(), Index60(), Index61(), Index62(), Index63(), Index64(), \
    Index65(), Index66(), Index67(), Index68(), Index69(), Index70()]

    name = "12th_"

    process = Process(name + "100th", Index())
    process.correlate(sequence=False)

    # load then save accuracy
    # for i in range(1, 46 + 1):
    #     if i == 1: new_name = name + "1st"
    #     elif i == 2: new_name = name + "2nd"
    #     elif i == 3: new_name = name + "3rd"
    #     else: new_name = name + str(i) + "th"
    #
    #     process = Process(new_name, index[i])
    #     process.load_then_save_accuracy()

    # train then predict
    # for i in range(48, 57 + 1):
    #     process = Process(name + str(i) + "th", index[i])
    #     process.train_then_predict()
    # for i in range(67, 70 + 1):
    #     process = Process(name + str(i) + "th", index[i])
    #     process.train_then_predict()


if __name__ == "__main__":
    main(index=10)
