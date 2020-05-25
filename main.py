from src.Cases import Index, Index2
from src.process import set_gpu, Process


def main(index):
    # set_gpu()
    process = Process("11th_saving_raw_routes", Index2())
    # process.save_raw_route()
    process.save_route_in_h5()
    # process.load_dataset()
    # process.train()
    # process.predict()
    # process.save_prediction()
    # process.save_readme()
    # process.statistic_raw_data()


if __name__ == "__main__":
    main(index=10)
