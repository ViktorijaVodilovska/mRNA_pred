from niapy.algorithms import Algorithm
from niapy.algorithms.other import (
    HillClimbAlgorithm,
    SimulatedAnnealing,
)
from niapy.algorithms.basic import (
    ParticleSwarmAlgorithm,
    # ArtificialBeeColony,
    GeneticAlgorithm,
)
from typing import Dict


# TODO: add params
optimizers: Dict[str, Algorithm] = {
    'hill': HillClimbAlgorithm(),
    'simulated': SimulatedAnnealing(),
    'pso': ParticleSwarmAlgorithm(),
    # 'abc': ArtificialBeeColony(),
    'genetic': GeneticAlgorithm()
}

# TODO: sweep across optimizer param options?
