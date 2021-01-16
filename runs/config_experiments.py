import json
import os


def config_experiments(results_dir):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    for net in ["MLP", "CNN"]:
        for robust_training in [True, False]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [0.1, 0.3, 0.01]:
                    config = base_config.copy()
                    config["model_name"] = str(id)
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["robust_training"] = robust_training
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                    id += 1

    print(str(id) + " config files created")