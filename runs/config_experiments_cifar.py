import json
import os
import numpy as np

def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    for dataset in [67] + [0, 66]:
        restart = False
        if dataset == 67: #CIFAR
            standarize = True
            multiplier = 255.0
        elif dataset == 0: #MNIST
            standarize = True
            multiplier = 255.0
        elif dataset == 66: #fashion MNIST
            standarize = True
            multiplier = 255.0

        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                config = base_config.copy()
                config["data_set"] = dataset
                config["model_name"] = str(id)
                config["backbone"] = net
                config["restart"] = restart
                config["training_batch_size"] = 256
                config["initial_learning_rate"] = lr
                config["robust_training"] = False
                config["pgd_training"] = False
                config["max_num_training_steps"] = 5000
                config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                config["bound_lower"] = -1e10
                config["bound_upper"] = 1e10
                config["standarize"] = standarize
                config["standarize_multiplier"] = multiplier

                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["restart"] = restart
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

        for net in ["CNN+pgd"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon_pgd_training in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["restart"] = restart
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["max_num_training_steps"] = 5000
                    config["epsilon"] = epsilon
                    config["robust_training"] = False
                    config["pgd_training"] = True
                    config["epsilon_pgd_training"] = epsilon_pgd_training
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["l1_robustness"] = True
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1
        #additional
        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["l1_robustness"] = True
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

    #additional
    for dataset in [67] + [0, 66]:

        restart = False
        if dataset == 67: #CIFAR
            standarize = True
            multiplier = 255.0
        elif dataset == 0: #MNIST
            standarize = True
            multiplier = 255.0
        elif dataset == 66: #fashion MNIST
            standarize = True
            multiplier = 255.0

        for net in ["CNN"]: #linf
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                for epsilon in [1e1, 1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["restart"] = restart
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

        for net in ["CNN+pgd"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                for epsilon_pgd_training in [1e1, 1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["restart"] = restart
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["max_num_training_steps"] = 5000
                    config["epsilon"] = epsilon
                    config["robust_training"] = False
                    config["pgd_training"] = True
                    config["epsilon_pgd_training"] = epsilon_pgd_training
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

        for net in ["CNN"]:
            for lr in [1e-6, 1e-7]:
                for epsilon in [1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["type_robust"] = "l1"
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

        #additional
    for dataset in [67] + [0, 66]:

        restart = False
        if dataset == 67: #CIFAR
            standarize = True
            multiplier = 255.0
            num_features = 32*32*3
        elif dataset == 0: #MNIST
            standarize = True
            multiplier = 255.0
            num_features = 28*28
        elif dataset == 66: #fashion MNIST
            standarize = True
            multiplier = 255.0
            num_features = 28*28


        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = np.sqrt(num_features)*epsilon
                    config["max_num_training_steps"] = 5000
                    config["type_robust"] = "l1"
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1
    for dataset in [67] + [0, 66]:

        restart = False
        if dataset == 67: #CIFAR
            standarize = True
            multiplier = 255.0
            num_features = 32*32*3
        elif dataset == 0: #MNIST
            standarize = True
            multiplier = 255.0
            num_features = 28*28
        elif dataset == 66: #fashion MNIST
            standarize = True
            multiplier = 255.0
            num_features = 28*28

        for net in ["CNN"]:
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["type_robust"] = "l1+inf"
                    config["robust_training"] = True
                    config["pgd_training"] = False
                    config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                    config["bound_lower"] = -1e10
                    config["bound_upper"] = 1e10
                    config["standarize"] = standarize
                    config["standarize_multiplier"] = multiplier

                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
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
