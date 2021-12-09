# A Robust Optimization Approach to Deep Learning
This repository is the official implementation of the paper "A Robust Optimization Approach to Deep Learning" by [D. Bertsimas](https://dbertsim.mit.edu/), [X. Boix](https://www.mit.edu/~xboix/), [K. Villalobos Carballo](https://github.com/kimvc7) and [D. den Hertog](https://www.uva.nl/en/profile/h/e/d.denhertog/d.den-hertog.html). 

>We develop two new methods for training neural networks that are robust against input perturbations. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an approximate upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes instead an exact upper bound of the adversarial loss by extending state-of-the-art tools from Robust Optimization to neural networks with ReLU activation function. Across a variety of tabular and vision data sets we present the following three results. Regarding adversarial accuracies, we find that for large perturbations RUB has a performance edge, while for smaller perturbations aRUB matches the performance of state-of-the-art methods. In terms of computational speed, we observe that both aRUB and RUB run faster than adversarial training at the expense of an increase in memory. Finally, with regard to natural accuracy, we demonstrate that training based on robust optimization has an edge over standard training. 

## Requirements 

Docker needs to be installed in your system.


Pull the following docker container:
```
docker pull xboixbosch/tf
```
Then, run bash in the container and execute all the code there.
```
docker run -v <CODE PATH>:/home/neuro/Robustness -it xboixbosch/tf bash
```

## Preparing the datasets and the experiments

Generate dataset configuration files:
```
python3 main.py --run=config --config=generate_datasets
```


```
python3 main.py --run=config --experiment_name=vision --config=generate
python3 main.py --run=config --experiment_name=uci_all --config=generate
```



```
python3 main.py --run=train --experiment_name=vision --experiment_id=0
```