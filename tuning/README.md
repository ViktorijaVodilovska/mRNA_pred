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
# (for wandb)
program: <path to program>
command:
    <command arguments>
method: <search method>
metric:
    goal: <minimize or optimize>
    name: <metric name>

# (for both)
parameters:
    <parameter_name>:
        values: [<val 1>, ..., <val n>]
```

### Niapy optimizers

To start a hyperparameter optimization experiment run:
```
python -m tuning.hpo --model <name of model> --algorithm <name of algorithm> --count <num explored solutions>
```

### Wandb optimizers

To create a hyperparameter optimization sweep experiment run:
```
wandb sweep --project <project_name> <path_to_config_yml>
```

To start the agent for sampling configurations and running training for a created sweep run 
```
wandb agent <sweep_id> --count <num_runs>
```

This project uses wandb for tracking all experiments. For details refer to https://docs.wandb.ai/


