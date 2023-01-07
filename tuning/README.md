# Hyperparameter Optimization

## How to use

This project uses wandb sweeps for tracking of hyperparametr optimization experiments.

Each experiment (sweep) is configured by a YAML file in tuning/hpo_configs/.

To create a hyperparameter optimization experiment run:
```
wandb sweep --project <project_name> <path_to_config_yml>
```

To start the agent for sampling configurations and running training for a created sweep run 
```
wandb agent <sweep_id> --count <num_runs>
```

For details refer to https://docs.wandb.ai/

## Algorithms

from wandb

- Random
- Bayes

other

- ...
- ...
