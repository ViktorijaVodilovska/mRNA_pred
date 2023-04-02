# Hyperparameter Optimization

## Algorithms

from niapy

- HillClimbAlgorithm,
- SimulatedAnnealing,
- ParticleSwarmAlgorithm,
- ArtificialBeeColony,
- GeneticAlgorithm

from wandb

- Random
- Bayes

## How to use

Each experiment (sweep) is configured by a YAML file in tuning/hpo_configs/. The configuration file contains:

```
parameters:
    <parameter_name>:
        values: [<val 1>, ..., <val n>]
```

### Niapy optimizers

To start a hyperparameter optimization experiment run:
```
python -m tuning.hpo --model <name of model> --algorithm <name of algorithm> --timeout 3600
```

### Wandb optimizers

To create a hyperparameter optimization sweep experiment run:
```
python -m tuning.sweep --model <name of model> --method <name of method> --timeout 3600
```

This project uses wandb for tracking all experiments. For details refer to https://docs.wandb.ai/


