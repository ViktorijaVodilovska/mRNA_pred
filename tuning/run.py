import sys
from configs.predictor_config import GraphConfig, PredictorConfig, TrainConfig
from scripts.experiments import run_training
import wandb
from configs.base_config import BaseSettings as settings

def run(**kwargs):
    """
    Casts arg params to a config dictionairy and calls the training function for the sweep
    """
    conf = kwargs

    res_conf = {}
    res_conf.update(PredictorConfig.from_dict(conf))
    res_conf.update(GraphConfig.from_dict(conf))
    res_conf.update(TrainConfig.from_dict(conf))

    print(res_conf)

    run = wandb.init() # gets everything from the sweep agent
    run_training(res_conf)
        

if __name__ == "__main__":
    run(**dict(arg.replace("--","").split('=') for arg in sys.argv[1:]))