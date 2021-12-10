# A Robust Optimization Approach to Deep Learning
This repository is the official implementation of the paper "A Robust Optimization Approach to Deep Learning" by [D. Bertsimas](https://dbertsim.mit.edu/), [X. Boix](https://www.mit.edu/~xboix/), [K. Villalobos Carballo](https://github.com/kimvc7) and [D. den Hertog](https://www.uva.nl/en/profile/h/e/d.denhertog/d.den-hertog.html). 

>We develop two new methods for training neural networks that are robust against input perturbations. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an approximate upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes instead an exact upper bound of the adversarial loss by extending state-of-the-art tools from Robust Optimization to neural networks with ReLU activation function. Across a variety of tabular and vision data sets we present the following three results. Regarding adversarial accuracies, we find that for large perturbations RUB has a performance edge, while for smaller perturbations aRUB matches the performance of state-of-the-art methods. In terms of computational speed, we observe that both aRUB and RUB run faster than adversarial training at the expense of an increase in memory. Finally, with regard to natural accuracy, we demonstrate that training based on robust optimization has an edge over standard training. 

## Requirements 

Docker needs to be installed in your system. Pull the following docker container:
```
docker pull xboixbosch/tf
```
Then, run bash in the container and execute all the code there.
```
docker run -v <CODE PATH>:/home/neuro/Robustness -it xboixbosch/tf bash
```

## Preparing and running the experiments

1. Generate dataset configuration files:

```
python3 main.py --run=config --config=generate_datasets
```
This generate a json file to configure each dataset.

2. Generate the experiment configuration files:

The configurations of all trained networks are prepared with `runs/config_experiment_uci_all.py` and `runs/config_experiment_vision.py`. These scripts generate one configuration file per network to train and evaluate.
To generate all the configuration files run the following:
```
python3 main.py --run=config --experiment_name=vision --config=generate
python3 main.py --run=config --experiment_name=uci_all --config=generate
```
These command will create ~5K json files for the vision datasets and ~35K json files for the UCI datasets. Each file
describes the network, hyperparameters, dataset, etc. of an experiment. The name of the file is a number that corresponds
to the `experiment_id`.

3. Run the training and testing:

To train, test, and evaluate the bound use the following commands:
```
python3 main.py --run=train --experiment_name=vision --experiment_id=<experiment_id> --gpu_id=0
python3 main.py --run=test --experiment_name=vision --experiment_id=<experiment_id> --gpu_id=0
python3 main.py --run=test_bound --experiment_name=vision --experiment_id=<experiment_id> --gpu_id=0
```
where `experiment_id` corresponds to the number of the experiment and `gpu_id` indicates the ID of the GPUs to run use 
(the current version of the code does not support multi-GPU).
To run the UCI experiment just replace `vision` by `uci_all`.

4. Analyze the results:

Use the jupter notebooks in `notebooks` folder.
