import json
import os
import numpy as np


def config_datasets(datasets_dir, create_json=True):

    id = 0
    dataset_list = []

    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "mnist"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1


    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "cifar"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1

    # Dataset based on Gaussian data + MLP-1
    for num_features in [10, 10000]:
        for num_data in [1e3, 1e4, 1e5]:
            for trial in range(3):
                config = {}
                config["dataset_id"] = id
                config["dataset_name"] = "Gauss_MLP-1"
                config["num_data"] = num_data
                config["num_features"] = num_features
                config["trial"] = trial
                config["validation_size"] = 1000
                config["testing_size"] = 1000
                config["num_classes"] = 2
                config["name_file"] = str(num_features) + '_' + str(num_data) + '_' + str(trial) + '.pickle'
                if create_json:
                    with open(datasets_dir + 'configs_datasets/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                dataset_list.append(config.copy())
                id += 1

    print(id)
    # UCI Dataset curated by Kim
    dataset_names = []
    with open(datasets_dir + 'datasets/UCI/datasets.csv', 'r') as file:
        for line in file:
            dataset_names.append(line[:-1])

    for dataset_id in range(46):
        config = {}
        config["dataset_id"] = id
        config["dataset_name"] = "UCI"
        config["name_file"] = dataset_names[dataset_id]
        tmpY = np.genfromtxt(datasets_dir + 'datasets/UCI/' + config["name_file"] + "_train" + "Y.csv", delimiter=',')
        config["num_classes"] = int(np.max(tmpY))

        if create_json:
            with open(datasets_dir + 'configs_datasets/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        dataset_list.append(config.copy())
        id += 1


    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "fashion_mnist"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1

    config = {}
    config["dataset_id"] = id
    config["dataset_name"] = "cifar"
    config["validation_size"] = 1000
    config["testing_size"] = 1000
    config["num_classes"] = 10
    if create_json:
        with open(datasets_dir + 'configs_datasets/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    dataset_list.append(config.copy())
    id += 1

    print(str(id) + " dataset config files created")
    return dataset_list


