import os
import json
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--experiment_id", type=int, default="0",
                            help="id of json file")
parser.add_argument("--run", type=str, default="idle",
                            help="run to perform on epxeriment <id>")
parser.add_argument("--filesystem", type=str, default="local",
                            help="filesystem")
parser.add_argument("--gpu_id", type=str, default="",
                            help="gpu IDs")
parser.add_argument("--config", type=str, default="",
                            help="config instruction")
args = parser.parse_args()
print(args)

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.filesystem == 'local':
    results_dir = './results/'
elif args.filesystem == 'om':
    results_dir = '/vast/robustness/'

if not args.run == 'config':

    with open(results_dir + 'configs/' + str(args.experiment_id) + '.json') as config_file:
        config = json.load(config_file)

    config['model_dir'] = results_dir + config['model_name']

    if args.run == 'train':
        import runs.train as run
        run.train(config)
    elif args.run == 'test':
        import runs.test as run
        run.test(config)

else:
    if args.config == 'generate':
        import runs.config_experiments as run
        run.config_experiments(results_dir)
    elif args.config == 'check':
        import runs.config_experiments as run
        experiment_list = run.config_experiments(results_dir, create_json=False)
        run.check_uncompleted(results_dir, experiment_list)
