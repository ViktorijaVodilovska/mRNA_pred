import sys
from configs.predictor_config import GraphConfig, PredictorConfig, TrainConfig
from scripts.experiments import run_training

def run(**kwargs):
    """
    Casts arg params to config params
    """
    conf = kwargs

    res_conf = {}
    res_conf.update(PredictorConfig.from_dict(conf))
    res_conf.update(GraphConfig.from_dict(conf))
    res_conf.update(TrainConfig.from_dict(conf))
    
    run_training(res_conf)


if __name__ == "__main__":
    run(**dict(arg.replace("--","").split('=') for arg in sys.argv[1:]))