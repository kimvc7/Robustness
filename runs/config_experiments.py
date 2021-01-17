import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    # Models with robustness
    for net in ["MLP", "CNN"]:
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            for epsilon in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
                config = base_config.copy()
                config["model_name"] = str(id)
                config["backbone"] = net
                config["initial_learning_rate"] = lr
                config["epsilon"] = epsilon
                config["robust_training"] = True
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    # Models without robustness
    for net in ["MLP", "CNN"]:
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                config = base_config.copy()
                config["model_name"] = str(id)
                config["backbone"] = net
                config["initial_learning_rate"] = lr
                config["epsilon"] = epsilon
                config["robust_training"] = False

                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    # CNNs with more extreme epsilons
    for net in ["CNN"]:
        for epsilon in [1e-3, 1e-4, 1e-5, 1e2, 1e3]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                config = base_config.copy()
                config["model_name"] = str(id)
                config["backbone"] = net
                config["initial_learning_rate"] = lr
                config["epsilon"] = epsilon
                config["robust_training"] = True

                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    config = base_config.copy()
    config["model_name"] = str(id)
    config["Madry"] = True
    config["backbone"] = "CNN"
    config["robust_training"] = True
    if create_json:
        with open(results_dir + 'configs/' + str(id) + '.json', 'w') as json_file:
            json.dump(config, json_file)
    experiment_list.append(config.copy())
    id += 1

    print(str(id) + " config files created")
    return experiment_list


def check_uncompleted(results_dir, experiments_list):

    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/training.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check train completed")

    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check test completed")
