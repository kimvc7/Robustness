import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    net = "ThreeLayer"
    batch_size = 32
    normalization = "01"
    standarize = False
    multiplier = 1.0
    upper = 1.0
    lower = 0.0
    lr = 1e-6
    epsilon = 1
    epsilon_pgd_training = 1

    id = 0
    experiment_list = []

    for dataset in [0, 66, 67]:

        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        config["restart"] = restart
        config["backbone"] = net
        config["training_batch_size"] = batch_size
        config["initial_learning_rate"] = lr
        config["robust_training"] = False
        config["pgd_training"] = False
        config["max_num_training_steps"] = 10000
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1

        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        config["restart"] = restart
        config["training_batch_size"] = batch_size
        config["backbone"] = net
        config["initial_learning_rate"] = lr
        config["epsilon"] = epsilon
        config["max_num_training_steps"] = 10000
        config["robust_training"] = True
        config["pgd_training"] = False
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1


        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        config["training_batch_size"] = batch_size
        config["restart"] = restart
        config["backbone"] = net
        config["initial_learning_rate"] = lr
        config["epsilon"] = epsilon
        config["max_num_training_steps"] = 10000
        config["robust_training"] = True
        config["type_robust"] = "l1"
        config["pgd_training"] = False
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1


        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        config["training_batch_size"] = batch_size
        config["restart"] = restart
        config["backbone"] = net
        config["initial_learning_rate"] = lr
        config["epsilon"] = epsilon
        config["max_num_training_steps"] = 10000
        config["robust_training"] = True
        config["type_robust"] = "grad"
        config["pgd_training"] = False
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1


        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        config["training_batch_size"] = batch_size
        config["restart"] = restart
        config["backbone"] = net + "+pgd"
        config["initial_learning_rate"] = lr
        config["max_num_training_steps"] = 10000
        config["epsilon"] = epsilon
        config["robust_training"] = False
        config["pgd_training"] = True
        config["epsilon_pgd_training"] = epsilon_pgd_training
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1


        config = base_config.copy()
        config["data_set"] = dataset
        config["model_name"] = str(id)
        if (batch_size == 256) & (dataset==67):
            config["training_batch_size"] = 128
        else:
            config["training_batch_size"] = batch_size
        config["backbone"] = net
        config["restart"] = restart
        config["initial_learning_rate"] = lr
        config["epsilon"] = epsilon
        config["max_num_training_steps"] = 10000
        config["robust_training"] = True
        config["type_robust"] = "certificate"
        config["pgd_training"] = False
        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
        config["bound_lower"] = lower
        config["bound_upper"] = upper
        config["standarize"] = standarize
        config["standarize_multiplier"] = multiplier

        if create_json:
            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                json.dump(config, json_file)
        experiment_list.append(config.copy())
        id += 1

    print(str(id) + " config files created")
    return experiment_list

