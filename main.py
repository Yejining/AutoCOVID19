from src.process import set_gpu, Process


def main(index):
    set_gpu()
    process = Process(index=index)
    process.save_raw_route()
    process.save_route_in_h5()
    process.load_dataset()
    process.train()
    process.predict()
    process.save_prediction()
    process.save_readme()


if __name__ == "__main__":
    main(index=10)
