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
parser.add_argument("--gpus", type=str, default="",
                            help="gpu IDs")
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

with open('./configs/' + str(args.experiment_id) + '.json') as config_file:
    config = json.load(config_file)

if args.filesystem == 'local':
    config['model_dir'] = './results/' + config['model_name']
elif args.filesystem == 'om':
    config['model_dir'] = '/vast/robustness/' + config['model_name']


if args.run == 'train':
    import runs.train as run
    run.train(config)
elif args.run == 'test':
    import runs.test as run
    run.test(config)