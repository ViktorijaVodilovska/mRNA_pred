from pathlib import Path
from niapy.algorithms import Algorithm
from niapy.algorithms.other import (
    HillClimbAlgorithm,
    SimulatedAnnealing,
)
from niapy.algorithms.basic import (
    ParticleSwarmAlgorithm,
    ArtificialBeeColonyAlgorithm,
    GeneticAlgorithm,
)
from typing import Dict

# dictionary with the model hpo configurations as yaml files
model_hpo_configs: Dict[str, Path] = {
    'GAT': Path('tuning/hpo_configs/gat_10_narrow.yaml'),
    'GCN': Path('tuning/hpo_configs/gcn_10_narrow.yaml'),
    # 'HAN': Path('tuning/hpo_configs/han_random.yaml'),
}

# TODO: add params
# TODO: meta hpo?
optimizers: Dict[str, Algorithm] = {
    'hill': HillClimbAlgorithm(),
    'simulated': SimulatedAnnealing(delta=0.3, starting_temperature=200, delta_temperature=0.9),
    'pso': ParticleSwarmAlgorithm(),
    'abc': ArtificialBeeColonyAlgorithm(),
    'genetic': GeneticAlgorithm()
}
