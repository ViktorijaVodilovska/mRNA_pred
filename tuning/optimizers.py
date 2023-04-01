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


# TODO: add params
# TODO: meta hpo?
optimizers: Dict[str, Algorithm] = {
    'hill': HillClimbAlgorithm(),
    'simulated': SimulatedAnnealing(delta=0.3, starting_temperature=200, delta_temperature=0.9),
    'pso': ParticleSwarmAlgorithm(),
    'abc': ArtificialBeeColonyAlgorithm(),
    'genetic': GeneticAlgorithm()
}
