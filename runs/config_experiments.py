import json
import os


def config_experiments(results_dir):

    with open(results_dir + 'configs/base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        for epsilon in [0.1, 0.3, 0.01]:
            config = base_config.copy()
            config["model_name"] = str(id)
            config["initial_learning_rate"] = lr
            config["epsilon"] = epsilon
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
            id += 1

    print(str(id) + " config files created")