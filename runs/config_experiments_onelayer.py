import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    for dataset in [67] + [0, 66] + list(range(20, 66)): #165 experiments per datasets

        if dataset == 67: #CIFAR
            standarize = True
            multiplier = 255.0
        elif dataset == 0: #MNIST
            standarize = True
            multiplier = 128.0 * 255.0
        elif dataset == 66: #fashion MNIST
            standarize = True
            multiplier = 255.0
        else: #UCI
            standarize = False
            multiplier = 255.0

        for net in ["OneLayer"]: #Vanilla
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                config = base_config.copy()
                config["data_set"] = dataset
                config["model_name"] = str(id)
                config["backbone"] = net
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

        for net in ["OneLayer"]: #Linf approx
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
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

        for net in ["OneLayer"]: #L1 approx
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["l1_robustness"] = True
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

        for net in ["OneLayer+pgd"]: #Madry
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon_pgd_training in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
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

        for net in ["OneLayer"]: #Certificate
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["l1_robustness"] = True
                    config["certificate"] = True
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
        for net in ["OneLayer"]: #L1 approx
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["l1_robustness"] = True
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


        for net in ["OneLayer"]: #Certificate
            for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                for epsilon in [1e2, 1e3, 1e4]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = 256
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["l1_robustness"] = True
                    config["certificate"] = True
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
