# mRNA_pred
**mRNA Degradation Prediction System**

## Research
This code is part of the study Hyperparameter Optimization of Graph Neural Networks for mRNA Degradation Prediction.

### Dataset
The dataset used in this study is from The Stanford Eterna Dataset of mRNA molecules. <br />
https://www.kaggle.com/competitions/stanford-covid-vaccine/data

### Methods
#### Models
- GAT
- GCN

#### Optimization Algorithms
- Random Search
- Bayesian Search
- Hill Climb
- Simulated Annealing
- Genetic Algorithm
- Particle Swarm Optimization
- Artificial Bee Colony

## Usage

### Installation
```
python -m venv .venv
pip install -r requirements.txt
```

### Hyperparameter Optimization
For more details on HPO options refer to the HPO documentation in [tuning/README.md](tuning/README.md).

### GNN Experiment
```
python -m scripts.experiment --config_path <path_to_json> --train true --test true --log true --save true
```
