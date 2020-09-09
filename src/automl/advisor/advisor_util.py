import json

from advisor_client.client import AdvisorClient

from src.Cases import Index69, Index11, Index13, Index83, Index18

# FILE_NAMES = ["17th_69th", "18th_11th", "19th_13th", "20th_83th"]
# TEST_NAMES = ["MergeAllInfectionCases", "HospitalFacilities",
#               "TransportFacilities", "BeverageAndFood"]
# PROCESS = [Index69(), Index11(), Index13(), Index83()]
# INDEX = 3
FILE_NAME = "21th_18th"
TEST_NAME = "RemoveInfectionCauses"
PROCESS = Index18()


def get_architecture_parameter(parameters):
    architecture_parameters = []

    for i in range(4):
        if i == 0:
            key_string = "conv_kernel_size"
        elif i == 1:
            key_string = "convlstm_kernel_size"
        elif i == 2:
            key_string = "epoch"
        elif i == 3:
            key_string = "batch_size"

        print(key_string, parameters.get(key_string))

        architecture_parameters.append(parameters.get(key_string))

    return architecture_parameters


def get_parameters(architecture_parameters):
    conv_kernel_size = architecture_parameters[0]
    convlstm_kernel_size = architecture_parameters[1]
    epoch = architecture_parameters[2]
    batch_size = architecture_parameters[3]

    print("parameters:", conv_kernel_size, convlstm_kernel_size, epoch, batch_size)

    return int(conv_kernel_size), int(convlstm_kernel_size), int(epoch), int(batch_size)


def main(train_function):
    client = AdvisorClient()

    name = FILE_NAME
    study_configuration = {
        "goal": "MINIMIZE",
        "randomInitTrials": 10,
        "params": [
            {
                "parameterName": "convlstm_kernel_size",
                "type": "DISCRETE",
                "minValue": 0,
                "maxValue": 0,
                "feasiblePoints": "2, 3, 5, 7, 9",
                "scalingType": "LINEAR"
            },
            {
                "parameterName": "conv_kernel_size",
                "type": "DISCRETE",
                "minValue": 0,
                "maxValue": 0,
                "feasiblePoints": "2, 3, 5, 7, 9",
                "scalingType": "LINEAR"
            },
            {
                "parameterName": "epoch",
                "type": "DISCRETE",
                "minValue": 0,
                "maxValue": 0,
                "feasiblePoints": "50, 100, 200, 300, 400",
                "scalingType": "LINEAR"
            },
            {
                "parameterName": "batch_size",
                "type": "DISCRETE",
                "minValue": 0,
                "maxValue": 0,
                "feasiblePoints": "1, 16, 32, 64, 128",
                "scalingType": "LINEAR"
            }
        ]
    }

    study = client.create_study(name, study_configuration, "BayesianOptimization")

    for num_trial in range(20):
        trials = client.get_suggestions(study.name, 10)

        print(len(trials))

        parameter_value_dicts = []
        for trial in trials:
            parameter_value_dict = json.loads(trial.parameter_values)
            print("The suggested parameters: {}".format(parameter_value_dict))
            parameter_value_dicts.append(parameter_value_dict)

        metrics = []
        for i in range(len(trials)):
            architecture_parameter = get_architecture_parameter(parameter_value_dicts[i])
            conv_kernel_size, convlstm_kernel_size, epoch, batch_size = get_parameters(architecture_parameter)
            metric = train_function(conv_kernel_size, convlstm_kernel_size, epoch, batch_size)
            metrics.append(metric)

        for i in range(len(trials)):
            trial = trials[i]
            client.complete_trial_with_one_metric(trial, metrics[i])
        is_done = client.is_study_done(study.name)
        best_trial = client.get_best_trial(study.name)
        print("The Study: {}. best trial: {}".format(study, best_trial))
