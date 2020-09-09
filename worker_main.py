from src.Cases import Index, IndexAge, IndexGender, IndexCause, IndexVisitType
from src.automl.hpbandster.worker import COVIDWorker2, appendResult
from src.constant import N_STEP, SIZE
from src.process import set_gpu, Process

import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from src.constant import *

from hpbandster.optimizers import BOHB as BOHB


# NAME = "hpbandster_1th_all_cases"
# RUN_ID = "all_cases"
# INDEX = Index()
N_ITERATIONS = 1
HOST = '127.0.0.1'

NAMES = ["hpbandster2_1th_all_cases", "hpbandster2_2th_developmental_stage",
         "hpbandster2_3th_remove_age", "hpbandster2_4th_remove_gender",
         "hpbandster2_5th_merge_three_infection_cases", "hpbandster2_6th_remove_infection_cases",
         "hpbandster2_7th_hospital_facilities", "hpbandster2_8th_transportation_facilities",
         "hpbandster2_9th_food_and_beverage", "hpbandster2_10th_remove_visit_type"]
ID_ARRAY = ["all_cases2", "developmental_stage2",
            "remove_age2", "remove_gender2",
            "merge_three_infection_cases2", "remove_infection_cases2",
            "hospital_facilities2", "transportation_facilities2",
            "food_and_beverage2", "remove_visit_type2"]
CASES = [Index(), IndexAge(AGE_MODE.DEVELOPMENTAL),
         IndexAge(AGE_MODE.REMOVE), IndexGender(GENDER_MODE.REMOVE),
         IndexCause(CAUSE_MODE.MERGE_THREE), IndexCause(CAUSE_MODE.REMOVE),
         IndexVisitType(VISIT_MODE.HOSPITAL), IndexVisitType(VISIT_MODE.TRANSPORTATION),
         IndexVisitType(VISIT_MODE.FNB), IndexVisitType(VISIT_MODE.REMOVE)]

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=10)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=400)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
args=parser.parse_args()


def hpo(index):
    NS = hpns.NameServer(run_id=ID_ARRAY[index], host=HOST, port=None)
    NS.start()
    print("ns started")

    process = Process(NAMES[i], CASES[i])
    # process.save_route_in_h5()
    process.load_dataset()

    dataset = []
    dataset.append(process.X_train)
    dataset.append(process.y_train)
    dataset.append(process.X_eval)
    dataset.append(process.y_eval)
    dataset.append(process.X_test)
    dataset.append(process.y_test)

    information = []
    information.append(N_STEP)   # n_step
    information.append(process.feature_info.get_all_counts())   # channels
    information.append(SIZE)    # image size
    information.append(process.path_info.get_result_path())

    channels = process.feature_info.get_all_counts()

    worker = COVIDWorker2(sleep_interval=0, nameserver=HOST, run_id=ID_ARRAY[i])
    worker.init(dataset, information)
    worker.run(background=True)
    print("worker run")

    bohb = BOHB(configspace=worker.get_configspace(channels),
                run_id=ID_ARRAY[i], nameserver=HOST,
                min_budget=args.min_budget, max_budget=args.max_budget)
    print("hobh initialized")
    res = bohb.run(n_iterations=N_ITERATIONS)
    print("bohb run")

    bohb.shutdown(shutdown_workers=True)
    print("bohb shutdown")
    NS.shutdown()
    print("ns shutdown")

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    files = []
    files.append('Best found configuration:' + str(id2config[incumbent]))
    files.append('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    files.append('A total of %i runs where executed.' % len(res.get_all_runs()))
    files.append('\n')

    for file in files:
        print(file)

    appendResult(process.path_info.get_result_path(), files)


if __name__ == "__main__":
    set_gpu()
    index = 0

    for i in range(0, 1):
        hpo(i)

