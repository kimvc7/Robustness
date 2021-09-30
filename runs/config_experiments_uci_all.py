import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    for net in ["ThreeLayer", "OneLayer"]:

        for dataset in range(20, 66): #315 experiments per dataset!

            restart = False
            standarize = False
            multiplier = 255.0
            batch_size = 256

            #Vanilla
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                config = base_config.copy()
                config["data_set"] = dataset
                config["model_name"] = str(id)
                config["restart"] = restart
                config["backbone"] = net
                config["training_batch_size"] = batch_size
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

            #Madry
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for epsilon_pgd_training in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = batch_size
                    config["restart"] = restart
                    config["backbone"] = net + "+pgd"
                    config["initial_learning_rate"] = lr
                    config["max_num_training_steps"] = 5000
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

            #Certificate
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = batch_size
                    config["backbone"] = net
                    config["restart"] = restart
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["type_robust"] = "certificate"
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


    for net in ["ThreeLayer", "OneLayer"]:
        for dataset in range(20, 66): #315 experiments per dataset!

            restart = False
            standarize = False
            multiplier = 255.0
            batch_size = 256

            #Linf approx
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["restart"] = restart
                    config["training_batch_size"] = batch_size
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


    for net in ["ThreeLayer", "OneLayer"]:
        for dataset in range(20, 66): #315 experiments per dataset!

            restart = False
            standarize = False
            multiplier = 255.0
            batch_size = 256

            #L1 approx
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = batch_size
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["type_robust"] = "l1"
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


    for net in ["ThreeLayer", "OneLayer"]:
        for dataset in range(20, 66): #315 experiments per dataset!

            restart = False
            standarize = False
            multiplier = 255.0
            batch_size = 256

            #grad
            for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                    config = base_config.copy()
                    config["data_set"] = dataset
                    config["model_name"] = str(id)
                    config["training_batch_size"] = batch_size
                    config["restart"] = restart
                    config["backbone"] = net
                    config["initial_learning_rate"] = lr
                    config["epsilon"] = epsilon
                    config["max_num_training_steps"] = 5000
                    config["robust_training"] = True
                    config["type_robust"] = "grad"
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

    for experiment in experiments_list[28980:]:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testingg.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check train completed")
    '''
    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check test completed")
    
    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing_bound.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check bound completed")
    '''

def get_missing(results_dir, experiments_list):

    missing = []
    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing_approx_bound.done'):
            missing.append(experiment["model_name"])
    with open("missing", "w") as output:
        for row in missing:
            output.write(str(row) + '\n')
    print("\n Check train completed")
    print(len(missing))
    '''
    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check test completed")
    
    
    for experiment in experiments_list:
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/testing_bound.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check bound completed")
    '''