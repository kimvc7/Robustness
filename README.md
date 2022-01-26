# A Robust Optimization Approach to Deep Learning
This repository is the official implementation of the paper ["A Robust Optimization Approach to Deep Learning"](https://arxiv.org/pdf/2112.09279v1.pdf) by [D. Bertsimas](https://dbertsim.mit.edu/), [X. Boix](https://www.mit.edu/~xboix/), [K. Villalobos Carballo](https://github.com/kimvc7) and [D. den Hertog](https://www.uva.nl/en/profile/h/e/d.denhertog/d.den-hertog.html). 

>Many state-of-the-art adversarial training methods leverage upper bounds of the adversarial loss to provide security guarantees. Yet, these methods require  computations at each training step that can not be incorporated in the gradient for backpropagation. We introduce a new, more principled approach to adversarial training based on a closed form solution of an upper bound of the adversarial loss, which can be effectively trained with backpropagation. This bound is facilitated by state-of-the-art tools from robust optimization. We derive two new methods with our approach.  The first method (\textit{Approximated Robust Upper Bound} or aRUB) uses the first order approximation of the network as well as basic tools from linear robust optimization to obtain an approximate upper bound of the adversarial loss that can be easily implemented. The second method (\textit{Robust Upper Bound} or RUB), computes an exact upper bound of the adversarial loss. Across a variety of tabular and vision data sets we demonstrate the effectiveness of our more principled approach ---RUB is substantially more robust than state-of-the-art methods for larger perturbations, while aRUB matches the performance of state-of-the-art methods for small perturbations. Also, both RUB and aRUB run faster than standard adversarial training (at the expense of an increase in memory).

## Demo
Check out `demo.ipynb` in google colaboratory for an illustrative example on how to use the code to train a network:

<a href="https://colab.research.google.com/github/kimvc7/Robustness/blob/main/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Demo In Colab!"/></a>


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

`runs/config_experiment_uci_all.py` and `runs/config_experiment_vision.py` generate one configuration file per network to train and evaluate.
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

Use the jupter notebooks in `notebooks` folder. `UCI.ipynb` and `Vision.ipynb` do the cross-validation for each attack and rho and save all relevant information in a csv file. `plots.ipynb` displays the results as in the paper (the code pulls the csv files for the experiments we run for the paper from a google drive shared folder).


