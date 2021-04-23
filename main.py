import os
import json
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--experiment_name", type=str, default="",
                            help="config instruction")
parser.add_argument("--experiment_id", type=int, default="0",
                            help="id of json file")
parser.add_argument("--run", type=str, default="idle",
                            help="run to perform on experiment <id>")
parser.add_argument("--filesystem", type=str, default="local",
                            help="filesystem")
parser.add_argument("--gpu_id", type=str, default="",
                            help="gpu IDs")
parser.add_argument("--config", type=str, default="",
                            help="config instruction")

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.filesystem == 'local':
    results_dir = './results/'
elif args.filesystem == 'om':
    results_dir = '/vast/robustness/'
elif args.filesystem == 'dgx1':
    results_dir = '/raid/poggio/home/xboix/results/Robustness/'

full_results_dir = results_dir + args.experiment_name + '/'

if not args.run == 'config':

    with open(full_results_dir + 'configs/' + str(args.experiment_id) + '.json') as config_file:
        config = json.load(config_file)

    config['model_dir'] = full_results_dir + config['model_name']
    config['results_dir'] = results_dir

    with open(results_dir + 'configs_datasets/' + str(config["data_set"]) + '.json') as config_file:
        config_dataset = json.load(config_file)
    config["num_classes"] = config_dataset["num_classes"]  # This is going to be needed to define the architecture

    if args.run == 'train':
        import runs.train as run
        run.train(config)
    elif args.run == 'test':
        import runs.test as run
        run.test(config)
    elif args.run == 'test_bound':
        import runs.test_bound as run
        run.test_bound(config)

else:
    if args.experiment_name == 'mnist':
        import runs.config_experiments_mnist as run_exp
    elif args.experiment_name == 'mnist_std':
        import runs.config_experiments_mnist_standarized as run_exp
    elif args.experiment_name == 'cifar':
        import runs.config_experiments_cifar as run_exp
    elif args.experiment_name == 'synthetic':
        import runs.config_experiments_synthetic as run_exp
    elif args.experiment_name == 'uci':
        import runs.config_experiments_uci_old as run_exp
    elif args.experiment_name == 'fashion':
        import runs.config_experiments_fashion as run_exp
    elif args.experiment_name == 'onelayer':
        import runs.config_experiments_onelayer as run_exp
    elif args.experiment_name == 'vision':
        import runs.config_experiments_vision as run_exp


    if args.config == 'generate':
        run_exp.config_experiments(full_results_dir)

    elif args.config == 'generate_datasets':
        import runs.config_datasets as run
        run.config_datasets(results_dir)

    elif args.config == 'check':
        experiment_list = run_exp.config_experiments(full_results_dir, create_json=False)
        run_exp.check_uncompleted(full_results_dir, experiment_list)
